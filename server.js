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
// Two GLM calls total per message now: DRAFT_MODEL writes the actual reply
// (sent to the user directly, no extra review pass), SUMMARY_MODEL handles
// memory compression in the background. This trades away the in-character
// enforcement/review pass for real latency savings.
const SUMMARY_MODEL = process.env.SUMMARY_MODEL || DRAFT_MODEL;
// Only used for the (background, non-blocking) memory summarization calls
// if you choose to enable thinking there — off by default for speed.
const REASONING_EFFORT = process.env.REASONING_EFFORT || 'medium';

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
// notebook.tiers is an array of arrays of strings. tiers[0] holds individual
// per-message compressed notes waiting to be grouped; tiers[1], tiers[2], ...
// hold progressively more condensed, progressively older memory. Higher tier
// = older + more compressed. Lower tier = more recent + more detailed.
async function loadNotebook(key) {
  try {
    const raw = await redis.get(key);
    if (!raw) return { tiers: [[]], lastRecordedIndex: -1 };
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
    return {
      tiers: Array.isArray(parsed.tiers) && parsed.tiers.length ? parsed.tiers : [[]],
      lastRecordedIndex: typeof parsed.lastRecordedIndex === 'number' ? parsed.lastRecordedIndex : -1,
    };
  } catch (err) {
    console.error('Redis load error:', err.message);
    return { tiers: [[]], lastRecordedIndex: -1 };
  }
}

async function saveNotebook(key, notebook) {
  try {
    await redis.set(key, JSON.stringify(notebook));
  } catch (err) {
    console.error('Redis save error:', err.message);
  }
}

// ---------- Tiered memory compression ----------
// Tier 0 groups by 5 (individual message notes -> first condensed note).
// Every tier above that groups by 3 (condensed notes -> more condensed notes),
// cascading upward with no fixed ceiling.
const TIER_BASE_GROUP_SIZE = 5;
const TIER_UPPER_GROUP_SIZE = 3;
function groupSizeForTier(tierIndex) {
  return tierIndex === 0 ? TIER_BASE_GROUP_SIZE : TIER_UPPER_GROUP_SIZE;
}

const MICRO_SUMMARY_INSTRUCTION =
  "You are compressing a single roleplay message into a compact note for " +
  "long-term memory. Preserve EVERY concrete detail: actions taken, decisions " +
  "made, facts revealed, objects/props introduced, physical or emotional " +
  "states, injuries, promises, and the substance of dialogue (paraphrased, " +
  "not verbatim). You may cut purely stylistic flourish, scene-setting " +
  "description, and repeated wording that carries no new information — but " +
  "do not omit any plot-relevant detail. Write concise third-person notes, " +
  "not narrative prose.\n\n" +
  "HARD LENGTH BUDGET: your entire output must fit within roughly 500 tokens " +
  "(about 350-375 words), no matter how long the input is. Compress more " +
  "aggressively rather than running out of room partway through — always " +
  "finish within budget, prioritizing the most plot-relevant details if you " +
  "must choose what to keep.\n\n" +
  "Return only the compressed note, no preamble.";

const MERGE_SUMMARY_INSTRUCTION =
  "You are merging several already-compressed memory notes (given in " +
  "chronological order) into one combined note for long-term memory. Remove " +
  "redundancy across them, but keep every distinct concrete detail: facts, " +
  "decisions, relationships, ongoing threads, states, promises. If entries " +
  "conflict, prefer the later one. Write concise third-person notes, not " +
  "narrative prose.\n\n" +
  "HARD LENGTH BUDGET: your entire output must fit within roughly 500 tokens " +
  "(about 350-375 words), no matter how many notes you're merging or how much " +
  "detail they contain. This is not optional — if there's a lot to merge, " +
  "compress MORE AGGRESSIVELY so the finished note still fits the budget in " +
  "full, rather than running out of room partway through. Never let your " +
  "output get cut off mid-thought — always finish within the budget, " +
  "prioritizing the most plot-relevant and most recent details if you must " +
  "choose what to keep.\n\n" +
  "Return only the merged note, no preamble.";

async function condenseTexts(instruction, texts) {
  const joined = texts.map((t, i) => `[${i + 1}] ${t}`).join('\n\n');
  const prompt = `${instruction}\n\nINPUT:\n${joined}`;
  try {
    // max_tokens is set a bit above the instructed 500-token target, purely
    // as a safety ceiling — the model is told to self-compress and land
    // under ~500 on its own. This headroom just avoids a hard API-level cut
    // in the rare case it runs slightly over while still finishing its
    // thought.
    const result = await callNim(SUMMARY_MODEL, [{ role: 'user', content: prompt }], 650);
    return result.trim();
  } catch (err) {
    console.error('Condense error:', err.message);
    // On failure, fall back to a raw truncated join rather than losing the
    // content entirely.
    return texts.join(' ').slice(0, 800);
  }
}

// Pushes an already-produced micro-note into tier 0 (no API call here — the
// note is expected to already be made, e.g. by the combined polish+memory
// call), then cascades merges upward through any tiers that hit threshold.
async function pushNoteAndCascade(notebook, microNote) {
  notebook.tiers[0] = notebook.tiers[0] || [];
  notebook.tiers[0].push(microNote);

  let tier = 0;
  while (notebook.tiers[tier] && notebook.tiers[tier].length >= groupSizeForTier(tier)) {
    const size = groupSizeForTier(tier);
    const group = notebook.tiers[tier].splice(0, size); // take the oldest N, remove them
    const merged = await condenseTexts(MERGE_SUMMARY_INSTRUCTION, group);
    notebook.tiers[tier + 1] = notebook.tiers[tier + 1] || [];
    notebook.tiers[tier + 1].push(merged);
    tier++; // check whether the tier above now also hit its threshold
  }
}

// Builds the memory text injected into the draft prompt: oldest/broadest
// (highest tier) first, down to most recent/most detailed (tier 0) last.
function buildMemoryContext(notebook) {
  const parts = [];
  for (let i = notebook.tiers.length - 1; i >= 0; i--) {
    if (notebook.tiers[i] && notebook.tiers[i].length > 0) {
      parts.push(notebook.tiers[i].join(' '));
    }
  }
  return parts.join('\n');
}

// ---------- NIM API call ----------
async function callNim(model, messages, max_tokens = 1024, enableThinking = false) {
  const body = { model, messages, max_tokens };
  if (enableThinking) {
    body.chat_template_kwargs = { enable_thinking: true };
    body.reasoning_effort = REASONING_EFFORT;
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
  const message = response.data?.choices?.[0]?.message;
  const content = message?.content;
  // With thinking mode, a model can burn its entire max_tokens budget on
  // reasoning and return content: null before ever writing the actual
  // answer (finish_reason will typically be "length" when this happens).
  // Fail loudly here instead of letting `null` silently propagate and crash
  // later wherever `.length` gets called on it.
  if (typeof content !== 'string') {
    const finishReason = response.data?.choices?.[0]?.finish_reason;
    throw new Error(
      `NIM returned no content for model "${model}" (finish_reason: ${finishReason || 'unknown'}). ` +
      `This usually means it ran out of max_tokens before finishing` +
      `${enableThinking ? ' (likely spent the whole budget on reasoning)' : ''}.`
    );
  }
  // Only the final answer is returned — reasoning_content (if present) is
  // internal scratch work and is never surfaced to the user.
  return content;
}

app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    draft_model: DRAFT_MODEL,
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

    // --- Inject persistent memory (as of before this turn) into the draft prompt ---
    // We build context from the notebook as it currently stands, BEFORE
    // recording this turn's newly-confirmed message. That's fine: the raw
    // message is already present in `messages` sent to the draft model
    // directly, so nothing is lost — it just hasn't been folded into the
    // compressed long-term notes yet, which happens in the background below.
    const memoryText = buildMemoryContext(notebook);
    const notebookContext = memoryText
      ? `\n\n[Persistent memory for this chat, oldest/broadest first, most recent/detailed last: ${memoryText}]`
      : '';
    const messagesWithMemory = notebookContext
      ? [
          ...messages.slice(0, 1), // keep original system/first message first if present
          { role: 'system', content: notebookContext },
          ...messages.slice(1),
        ]
      : messages;

    // --- Determine if a memory note is needed this turn ---
    // This runs on the incoming `messages` array (what JanitorAI has already
    // accepted and is replaying back to us), NOT on anything we're about to
    // generate. A rerolled/rejected reply never reappears in a future request,
    // so it never gets recorded — only what the user actually kept does.
    const lastAssistant = getLastAssistantEntry(messages);
    const needsMemoryNote = Boolean(lastAssistant && lastAssistant.index > notebook.lastRecordedIndex);

    // Step 1: draft model writes the actual reply, sent to the user as-is —
    // no second review/correction pass. This is the only call the response
    // has to wait on.
    let finalText = await callNim(DRAFT_MODEL, messagesWithMemory);

    // --- Memory: fully separate call, in the BACKGROUND, not blocking the reply ---
    // Deliberately not awaited: this involves its own model call (micro-
    // summary + possible tier cascades) that would otherwise add real
    // latency to every single reply, for no benefit to THIS turn's response
    // (the raw message is already in `messages` above, so the draft model
    // sees it regardless). It just saves to Redis whenever it finishes.
    if (needsMemoryNote) {
      notebook.lastRecordedIndex = lastAssistant.index; // mark immediately so a concurrent request can't double-record
      condenseTexts(MICRO_SUMMARY_INSTRUCTION, [lastAssistant.content])
        .then(microNote => pushNoteAndCascade(notebook, microNote))
        .then(() => saveNotebook(notebookKey, notebook))
        .catch(err => console.error('Background memory recording failed:', err.message));
    }

    // Note: we do NOT record `finalText` (the roleplay reply) itself. Memory
    // is built from `lastAssistant.content` (already-confirmed history), so
    // a rerolled/rejected reply is never seen by the memory system at all.

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
  console.log(`Summary model: ${SUMMARY_MODEL}`);
});
