const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');
const { Redis } = require('@upstash/redis');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const NIM_API_KEY = process.env.NIM_API_KEY;
const NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions';

const DRAFT_MODEL = process.env.DRAFT_MODEL || 'z-ai/glm-5.2';
const POLISH_MODEL = process.env.POLISH_MODEL || 'z-ai/glm-5.2';
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || DRAFT_MODEL;

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});

// ---------- Chat ID marker (zero-width encoding) ----------
// Invisible marker embedded in the assistant's FIRST reply of a chat so we can
// recognize and reuse the same chat ID on every subsequent turn, without
// JanitorAI ever needing to give us an explicit conversation ID.
const DELIM = '\u200D';
const BIT0 = '\u200B';
const BIT1 = '\u200C';

function generateChatId() {
  return crypto.randomBytes(8).toString('hex');
}

function encodeMarker(id) {
  const bits = id
    .split('')
    .map(ch => parseInt(ch, 16).toString(2).padStart(4, '0'))
    .join('');
  const encoded = bits.split('').map(b => (b === '0' ? BIT0 : BIT1)).join('');
  return `${DELIM}${encoded}${DELIM}`;
}

function decodeMarker(text) {
  const match = text.match(new RegExp(`${DELIM}([${BIT0}${BIT1}]+)${DELIM}`));
  if (!match) return null;
  const bits = match[1].split('').map(ch => (ch === BIT0 ? '0' : '1')).join('');
  let id = '';
  for (let i = 0; i < bits.length; i += 4) {
    id += parseInt(bits.slice(i, i + 4), 2).toString(16);
  }
  return id;
}

function stripMarker(text) {
  return text.replace(new RegExp(`${DELIM}[${BIT0}${BIT1}]+${DELIM}`), '');
}

// Finds the most recent assistant message in the incoming (confirmed) history,
// along with its index in the array. Only messages that survive here are ones
// JanitorAI actually kept and replayed back to us — a rerolled/rejected reply
// never shows up in a future request, so it never reaches this function.
function getLastAssistantEntry(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant' && typeof messages[i].content === 'string') {
      return { index: i, content: stripMarker(messages[i].content) };
    }
  }
  return null;
}

function findExistingChatId(messages) {
  for (const m of messages) {
    if (m.role === 'assistant' && typeof m.content === 'string') {
      const id = decodeMarker(m.content);
      if (id) return id;
    }
  }
  return null;
}

// ---------- Character ID ----------
// Hash of the character's system/persona prompt, so the same character always
// maps to the same key even without JanitorAI giving us an explicit ID.
function getCharacterId(messages) {
  const systemMsg = messages.find(m => m.role === 'system');
  const basis = systemMsg?.content || messages[0]?.content || '';
  return crypto.createHash('sha256').update(basis).digest('hex').slice(0, 16);
}

// ---------- Notebook load/save ----------
async function loadNotebook(key) {
  try {
    const raw = await redis.get(key);
    if (!raw) return { summary: '', recent: [], lastRecordedIndex: -1 };
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    return {
      summary: parsed.summary || '',
      recent: parsed.recent || [],
      lastRecordedIndex: typeof parsed.lastRecordedIndex === 'number' ? parsed.lastRecordedIndex : -1,
    };
  } catch (err) {
    console.error('Redis load error:', err.message);
    return { summary: '', recent: [], lastRecordedIndex: -1 };
  }
}

async function saveNotebook(key, notebook) {
  try {
    await redis.set(key, JSON.stringify(notebook));
  } catch (err) {
    console.error('Redis save error:', err.message);
  }
}

// ---------- Notebook summarization ----------
const SUMMARY_TRIGGER_COUNT = 6; // condense every N accumulated exchanges

const SUMMARIZE_INSTRUCTION =
  "You maintain a running memory summary for an ongoing roleplay chat. " +
  "You will be given the CURRENT summary (may be empty) and a list of RECENT " +
  "exchanges since the last update. Merge them into a single updated summary.\n\n" +
  "Rules:\n" +
  "- Keep only what matters for future continuity: established facts, " +
  "relationships, ongoing plot threads, settings, promises made, unresolved " +
  "tension, character states/injuries/emotions.\n" +
  "- Drop moment-to-moment dialogue and flourish — keep the substance, not the prose.\n" +
  "- Write in concise third-person notes, not narrative prose. Bullet-style " +
  "fragments are fine.\n" +
  "- Target length: 4-8 short sentences/fragments MAX, regardless of how much " +
  "input you're given. If old and new info conflict, prefer the newer info.\n" +
  "- Do not invent anything not present in the input.\n\n" +
  "Return only the updated summary text, no preamble.";

async function summarizeNotebook(notebook) {
  const recentText = notebook.recent
    .map(r => `Turn ${r.turn}: ${r.snippet}`)
    .join('\n');

  const prompt =
    `${SUMMARIZE_INSTRUCTION}\n\n` +
    `CURRENT SUMMARY:\n${notebook.summary || '(empty, this is the first summarization)'}\n\n` +
    `RECENT EXCHANGES:\n${recentText}`;

  try {
    const updated = await callNim(SUMMARY_MODEL, [{ role: 'user', content: prompt }], 300);
    return updated.trim();
  } catch (err) {
    console.error('Summarization error:', err.message);
    // On failure, keep the old summary rather than losing it.
    return notebook.summary;
  }
}

// ---------- NIM API call ----------
async function callNim(model, messages, max_tokens = 1024, enableThinking = false) {
  const body = { model, messages, max_tokens };
  if (enableThinking) {
    body.chat_template_kwargs = { enable_thinking: true };
    body.reasoning_effort = 'high';
  }
  const response = await axios.post(
    NIM_BASE_URL,
    body,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
      },
      timeout: 120000,
    }
  );
  // Only the final answer is returned — reasoning_content (if present) is
  // internal scratch work and is never surfaced to the user.
  return response.data.choices[0].message.content;
}

// Strict in-character enforcement pass. Unlike a copyedit pass, this one is
// specifically hunting for character breaks and correcting them — but it's
// still bounded: it corrects breaks, it doesn't rewrite the scene.
const POLISH_INSTRUCTION =
  "You are a strict in-character consistency reviewer for a roleplay draft " +
  "written by another model. Your job is to catch and fix any place where the " +
  "draft broke character, then hand back a corrected version — not to rewrite " +
  "the scene.\n\n" +
  "Check the draft against the character's established personality, voice, " +
  "speech patterns, knowledge, and the conversation history, and fix any of " +
  "the following if present:\n" +
  "1. POV violations: the draft has the USER's character speak, act, or " +
  "decide something on its own that wasn't given to it in the conversation " +
  "history. Correct this so that character only reacts to what it was given — " +
  "never invents its own initiative. This is the most common and most serious " +
  "mistake to catch.\n" +
  "2. Tone/personality breaks: the roleplay character suddenly acting far " +
  "outside its established personality, speech style, or knowledge with no " +
  "narrative reason (e.g. a cold character being suddenly warm, a character " +
  "knowing something it has no way of knowing, dropping an established accent " +
  "or verbal tic, contradicting a previously stated fact about itself).\n" +
  "3. Meta breaks: the draft slipping out of the fiction — narrator asides, " +
  "AI-assistant-style disclaimers, apologizing out of character, safety " +
  "boilerplate, or any acknowledgment that this is an AI/roleplay/story. " +
  "Remove these entirely and replace with an in-fiction equivalent if needed.\n" +
  "4. Continuity errors: contradicting an established fact from the " +
  "conversation history (a prop, a name, an injury, a location, a promise " +
  "made earlier).\n\n" +
  "If NONE of the above are present, make no changes beyond trivial word " +
  "choice/flow polish — do not rewrite dialogue or actions that are already " +
  "in-character and consistent, and do not add new plot events, props, or " +
  "decisions that the draft didn't already contain.\n\n" +
  "Use the full conversation history below only as your source of truth for " +
  "what's already been established — do not respond to it or continue the " +
  "conversation yourself.\n\n" +
  "Hard constraint: your output must be roughly the same length as the draft " +
  "(within about 20%) — you are correcting specific breaks, not padding or " +
  "expanding the scene. Return only the corrected response, no preamble, no " +
  "explanation of what you changed, no meta-commentary about the review itself.";

app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    draft_model: DRAFT_MODEL,
    polish_model: POLISH_MODEL,
    summary_model: SUMMARY_MODEL,
    key_configured: Boolean(NIM_API_KEY),
    redis_configured: Boolean(process.env.UPSTASH_REDIS_REST_URL),
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { messages, stream } = req.body;

    if (!NIM_API_KEY) {
      return res.status(500).json({ error: { message: 'NIM_API_KEY not configured on server' } });
    }
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: { message: 'messages array is required' } });
    }

    // --- Identify character + chat, load notebook ---
    const characterId = getCharacterId(messages);
    let chatId = findExistingChatId(messages);
    const isNewChat = !chatId;
    if (isNewChat) chatId = generateChatId();

    const notebookKey = `notebook:${characterId}:${chatId}`;
    const notebook = await loadNotebook(notebookKey);

    // --- Record any newly-CONFIRMED assistant message from incoming history ---
    // This runs on the incoming `messages` array (what JanitorAI has already
    // accepted and is replaying back to us), NOT on anything we're about to
    // generate. A rerolled/rejected reply never reappears in a future request,
    // so it never gets recorded here — only what the user actually kept does.
    const lastAssistant = getLastAssistantEntry(messages);
    if (lastAssistant && lastAssistant.index > notebook.lastRecordedIndex) {
      notebook.recent.push({ turn: lastAssistant.index, snippet: lastAssistant.content.slice(0, 300) });
      notebook.lastRecordedIndex = lastAssistant.index;

      if (notebook.recent.length >= SUMMARY_TRIGGER_COUNT) {
        notebook.summary = await summarizeNotebook(notebook);
        notebook.recent = [];
      }

      await saveNotebook(notebookKey, notebook);
    }

    // --- Inject persistent memory into the draft prompt ---
    const notebookContext = notebook.summary
      ? `\n\n[Persistent memory for this chat: ${notebook.summary}]`
      : '';
    const messagesWithMemory = notebookContext
      ? [
          ...messages.slice(0, 1), // keep original system/first message first if present
          { role: 'system', content: notebookContext },
          ...messages.slice(1),
        ]
      : messages;

    // Step 1: draft model stays in character using the full conversation history
    const draft = await callNim(DRAFT_MODEL, messagesWithMemory);

    // Step 2: polish model does a light copyedit pass, grounded on the full
    // conversation history, length-capped relative to the draft.
    const polishMessages = [
      {
        role: 'user',
        content:
          `${POLISH_INSTRUCTION}\n\n` +
          `Full conversation history (for grounding/consistency only — do not respond to it):\n` +
          `${JSON.stringify(messages)}\n\n` +
          `Draft to copyedit:\n${draft}`,
      },
    ];

    const estimatedDraftTokens = Math.ceil(draft.length / 4);
    // Extra headroom beyond the normal length guard, since reasoning tokens
    // (the model's internal "thinking") count against max_tokens too, even
    // though only the final content is ever returned to the user.
    const polishMaxTokens = Math.max(256, Math.ceil(estimatedDraftTokens * 1.3)) + 1024;

    let finalText = await callNim(POLISH_MODEL, polishMessages, polishMaxTokens, true);

    // Safety net: if the polish pass still bloats the response well past the
    // draft, prefer the draft over a padded rewrite.
    if (finalText.length > draft.length * 1.6) {
      console.warn('Polish output exceeded length guard, falling back to draft.');
      finalText = draft;
    }

    // Note: we do NOT record `finalText` here. It only gets written to the
    // notebook once it comes back to us as confirmed history on a future
    // request (handled by the block above, near the top of this handler).
    // This is what makes rerolled/rejected replies invisible to memory.

    // --- Embed chat ID marker on EVERY reply, not just the first ---
    // This is deliberate: if old messages ever fall out of what JanitorAI
    // sends us (context truncation, forking into a new chat, etc.), we only
    // need ONE surviving assistant message with the marker to correctly
    // recognize and continue the same memory. Embedding it every time is
    // cheap (same ID, tiny overhead) and makes the whole system resilient
    // to history being cut or duplicated.
    finalText += encodeMarker(chatId);

    // --- Respond ---
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const chunk = {
        id: 'chatcmpl-proxy',
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: { role: 'assistant', content: finalText }, finish_reason: null }],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);

      const doneChunk = {
        id: 'chatcmpl-proxy',
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      };
      res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      res.json({
        id: 'chatcmpl-proxy',
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: finalText },
            finish_reason: 'stop',
          },
        ],
      });
    }
  } catch (err) {
    console.error('Proxy error:', err.response?.data || err.message);
    res.status(500).json({
      error: { message: err.response?.data?.error?.message || err.message || 'Proxy request failed' },
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
  console.log(`Draft model: ${DRAFT_MODEL}`);
  console.log(`Polish model: ${POLISH_MODEL}`);
  console.log(`Summary model: ${SUMMARY_MODEL}`);
});
