/**
 * Pilotage WhatsApp bridge — walking skeleton.
 *
 * Holds the WhatsApp connection in Node (Baileys is Node-only), and exposes a
 * loopback HTTP API to the Python runtime:
 *
 *   GET  /health    -> { status, pid, connected, me }
 *   GET  /messages  -> drains the inbound queue
 *   POST /typing    -> { chatId }
 *   POST /read      -> { key }
 *   POST /send      -> { chatId, message, replyTo? }
 *   POST /send-media -> { chatId, filePath, mediaType?, fileName? }
 *
 * The HTTP contract is ours. Everything below it — the reconnect scheduler, the
 * version resolver, the serialized send queue, the message unwrapping, the
 * media extraction, the identifier resolution — is the Hermes bridge's proven
 * code (scripts/whatsapp-bridge/{bridge,bridge_helpers,allowlist}.js), kept as
 * it stands rather than reimplemented.
 *
 * Inbound is complete. Outbound supports text plus native generated files;
 * polls, locations and reactions remain outside the production requirement.
 */

import process from 'node:process';
import path from 'node:path';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';

import express from 'express';
import pino from 'pino';
import qrcode from 'qrcode-terminal';

import {
  makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
} from '@whiskeysockets/baileys';

import {
  expandWhatsAppIdentifiers,
  matchesAllowedUser,
  normalizeWhatsAppIdentifier,
  parseAllowedUsers,
} from './allowlist.js';
import {
  buildTextSendPayload,
  createBoundedMessageStore,
  createReconnectScheduler,
  createVersionResolver,
  extractBridgeEvent,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  normalizeWhatsAppId,
} from './bridge_helpers.js';

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

function readArg(name, fallback) {
  const flag = `--${name}`;
  const index = process.argv.indexOf(flag);
  if (index !== -1 && index + 1 < process.argv.length) return process.argv[index + 1];
  const inline = process.argv.find((a) => a.startsWith(`${flag}=`));
  if (inline) return inline.slice(flag.length + 1);
  return fallback;
}

function readFlag(name, envName, fallback = false) {
  const raw = readArg(name, undefined) ?? process.env[envName];
  if (raw === undefined || raw === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(raw).toLowerCase());
}

const PORT = Number.parseInt(readArg('port', '8765'), 10);
const INSTANCE_TOKEN = readArg('instance-token', process.env.PILOTAGE_BRIDGE_TOKEN || '');
const SESSION_DIR = readArg('session', './session');
const PAIR_ONLY = process.argv.includes('--pair-only');
// Inbound media is written here, outside the session directory: re-pairing
// deletes the session, and a cached voice note should not depend on that.
const MEDIA_DIR = readArg('media', './media');
// Read receipts (the blue ticks) are off unless the operator asks for them, so
// an agent watching a chat does not silently mark everything as read. (Hermes)
const SEND_READ_RECEIPTS = readFlag('read-receipts', 'PILOTAGE_SEND_READ_RECEIPTS', false);
const ANSWER_GROUPS = readFlag('answer-groups', 'PILOTAGE_ANSWER_GROUPS', false);
const ALLOWED_USERS = parseAllowedUsers(process.env.PILOTAGE_ALLOWED_SENDERS || '');
const ALLOWED_GROUPS = parseAllowedUsers(process.env.PILOTAGE_ALLOWED_GROUPS || '');

if (!PAIR_ONLY && !INSTANCE_TOKEN) {
  console.error('[bridge] --instance-token is required');
  process.exit(2);
}

const CACHE_DIRS = {
  image: path.join(MEDIA_DIR, 'images'),
  document: path.join(MEDIA_DIR, 'documents'),
  audio: path.join(MEDIA_DIR, 'audio'),
};

const MAX_QUEUE_SIZE = 100;
const MAX_MESSAGE_LENGTH = 4096;
const CHUNK_DELAY_MS = 300;
const SEND_TIMEOUT_MS = 60_000;
const VERSION_FETCH_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 3_000;
const RESTART_DELAY_MS = 1_000;

const logger = pino({ level: 'silent' });

mkdirSync(SESSION_DIR, { recursive: true });
for (const dir of Object.values(CACHE_DIRS)) mkdirSync(dir, { recursive: true });

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let sock = null;
let connected = false;
let meId = null;
const messageQueue = [];

function log(...args) {
  console.log(`[bridge] ${args.join(' ')}`);
}

// ---------------------------------------------------------------------------
// Connection guards (Hermes bridge_helpers.js)
// ---------------------------------------------------------------------------

// Both guards exist because Baileys can hang rather than fail: startSocket()
// awaits network I/O before it registers any handler, and the version fetch
// has no AbortSignal. See bridge_helpers.js for the full reasoning.
const scheduleReconnect = createReconnectScheduler(() => startSocket(), { log });
const getWAVersion = createVersionResolver(fetchLatestBaileysVersion, {
  timeoutMs: VERSION_FETCH_TIMEOUT_MS,
  log,
});

// ---------------------------------------------------------------------------
// Outbound
// ---------------------------------------------------------------------------

/**
 * Split a long reply on a line break, falling back to a word break and then to
 * a hard cut. A line break is only worth splitting on when it lands in the
 * second half of the chunk, otherwise the chunk comes out mostly empty.
 */
export function splitLongMessage(message, maxLength = MAX_MESSAGE_LENGTH) {
  const text = String(message || '');
  if (!text) return [];
  if (!Number.isFinite(maxLength) || maxLength < 1 || text.length <= maxLength) {
    return [text];
  }

  const chunks = [];
  let remaining = text;
  while (remaining.length > maxLength) {
    let splitAt = remaining.lastIndexOf('\n', maxLength);
    if (splitAt < Math.floor(maxLength / 2)) {
      splitAt = remaining.lastIndexOf(' ', maxLength);
    }
    if (splitAt < 1) splitAt = maxLength;

    chunks.push(remaining.slice(0, splitAt).trimEnd());
    remaining = remaining.slice(splitAt).trimStart();
  }
  if (remaining) chunks.push(remaining);
  return chunks;
}

/**
 * The recent messages, kept so a reply can quote the message it answers.
 * Baileys needs the whole original message to build a quote, not just its id,
 * and it is bounded because a long-running agent would otherwise hold every
 * message it ever saw. (Hermes)
 */
const messageStore = createBoundedMessageStore(512);

/**
 * Every send goes through this single promise chain. Overlapping sends on one
 * Baileys socket can misdeliver: the protocol-level routing does not survive
 * two sendMessage() promises racing on the same socket.
 */
let sendQueue = Promise.resolve();

function enqueueSend(fn) {
  const task = sendQueue.then(() => fn(), () => fn());
  sendQueue = task.catch(() => {});
  return task;
}

/**
 * Baileys can hang forever on a send; never let an HTTP handler wait on that.
 * The clock starts before the queue, so a stuck send ahead of us counts.
 */
function sendWithTimeout(chatId, payload, options = {}, timeoutMs = SEND_TIMEOUT_MS) {
  let timer;
  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`sendMessage timed out after ${timeoutMs / 1000}s`)),
      timeoutMs,
    );
  });
  return enqueueSend(() =>
    Promise.race([sock.sendMessage(chatId, payload, options), timeoutPromise])
      .finally(() => clearTimeout(timer))
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Inbound
// ---------------------------------------------------------------------------

function digitsOf(value) {
  const normalized = normalizeWhatsAppIdentifier(value);
  return /^\d+$/.test(normalized) ? normalized : '';
}

/**
 * Turn a raw Baileys message into the event the Python runtime consumes.
 *
 * The body, the media download and every non-text form — voice notes,
 * documents, stickers, locations, contacts, reactions, polls — come from
 * Hermes' extractBridgeEvent unchanged. What is added here is ours: the
 * identifier expansion the allowlist needs, and a plain-number timestamp
 * (Baileys hands back a protobuf Long, which does not survive JSON).
 */
async function buildEvent(msg) {
  const chatId = msg.key?.remoteJid;
  if (!chatId || chatId === 'status@broadcast') return null;

  const isGroup = chatId.endsWith('@g.us');
  const senderId = (isGroup ? msg.key?.participant : chatId) || chatId;
  // `senderPn` carries the real phone number when the jid is a @lid alias.
  const senderPn = msg.key?.senderPn || msg.key?.participantPn || null;
  const senderNumber = digitsOf(senderPn) || digitsOf(senderId);

  // Every form this person is known by — phone and LID — resolved from the
  // session's mapping files, so the allowlist matches whichever one arrives.
  const identities = new Set();
  for (const candidate of [senderId, senderPn]) {
    for (const alias of expandWhatsAppIdentifiers(candidate, SESSION_DIR)) {
      identities.add(alias);
    }
  }

  const event = await extractBridgeEvent({
    msg,
    chatId,
    senderId,
    senderNumber,
    // Hermes carries both forms because WhatsApp may mention or quote either
    // the phone JID or the linked-identity JID.
    botIds: Array.from(new Set([
      normalizeWhatsAppId(sock.user?.id),
      normalizeWhatsAppId(sock.user?.lid),
    ].filter(Boolean))),
    isGroup,
    downloadMedia: async (mediaMsg) => downloadMediaMessage(
      mediaMsg,
      'buffer',
      {},
      { logger, reuploadRequest: sock.updateMediaMessage },
    ),
    cacheDirs: CACHE_DIRS,
  });

  return {
    ...event,
    senderNumber,
    identities: Array.from(identities),
    pushName: msg.pushName || '',
    timestamp: Number(msg.messageTimestamp) || 0,
  };
}

// ---------------------------------------------------------------------------
// Socket
// ---------------------------------------------------------------------------

async function startSocket() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const version = await getWAVersion();

  sock = makeWASocket({
    ...(version ? { version } : {}),
    auth: state,
    logger,
    printQRInTerminal: false,
    browser: ['Pilotage Agent', 'Chrome', '120.0'],
    syncFullHistory: false,
    markOnlineOnConnect: false,
    // Required on Baileys 7.x: without it, inbound messages that need an E2EE
    // session re-establishment arrive with `message === null` and are lost.
    getMessage: async () => ({ conversation: '' }),
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      log('scan this QR code with WhatsApp > Linked devices');
      qrcode.generate(qr, { small: true });
    }

    if (connection === 'open') {
      connected = true;
      meId = sock.user?.id || null;
      log(`connected as ${meId}`);
      if (PAIR_ONLY) {
        log('pairing complete; credentials saved');
        setTimeout(() => process.exit(0), 2000);
      }
      return;
    }

    if (connection === 'close') {
      connected = false;
      // Hermes wraps this in Boom; a plain read reaches the same two branches,
      // since an unwrapped or missing error falls through to the reconnect.
      const reason = lastDisconnect?.error?.output?.statusCode;
      if (reason === DisconnectReason.loggedOut) {
        log('logged out on the phone — delete the session directory and pair again');
        process.exit(1);
      }
      // 515: WhatsApp asks for a restart right after pairing.
      const delay = reason === 515 ? RESTART_DELAY_MS : RECONNECT_DELAY_MS;
      log(`connection closed (${reason ?? 'unknown'}) — reconnecting in ${delay}ms`);
      scheduleReconnect(delay);
    }
  });

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify' && type !== 'append') return;

    for (const msg of messages || []) {
      try {
        if (!msg.message) continue;
        // The agent runs on its own number. Anything this account sent is
        // either our own reply echoing back or a message the operator typed on
        // the linked device — never something to answer.
        if (msg.key?.fromMe) continue;

        // Reject before extractBridgeEvent can download or cache media. Python
        // repeats this gate before dispatch, but unauthorized content should
        // never cross the bridge boundary in the first place.
        const chatId = msg.key?.remoteJid;
        if (!chatId || chatId === 'status@broadcast') continue;
        const isGroup = chatId.endsWith('@g.us');
        const senderId = (isGroup ? msg.key?.participant : chatId) || chatId;
        const senderPn = msg.key?.senderPn || msg.key?.participantPn || null;
        if (isGroup) {
          // Group access is keyed by the chat, independently from the DM
          // sender allowlist. Python repeats this gate and applies mentions.
          if (
            !ANSWER_GROUPS
            || !matchesAllowedUser(chatId, ALLOWED_GROUPS, SESSION_DIR)
          ) continue;
        } else if (
          !matchesAllowedUser(senderId, ALLOWED_USERS, SESSION_DIR)
          && !matchesAllowedUser(senderPn, ALLOWED_USERS, SESSION_DIR)
        ) {
          continue;
        }

        messageStore.remember(msg);
        // eslint-disable-next-line no-await-in-loop
        const event = await buildEvent(msg);
        if (!event) continue;
        // Nothing to answer and nothing attached: a receipt, an empty
        // protocol message, or a form we do not read. (Hermes)
        if (!event.body && !event.hasMedia) continue;

        messageQueue.push(event);
        if (messageQueue.length > MAX_QUEUE_SIZE) messageQueue.shift();
      } catch (error) {
        log(`failed to read an inbound message: ${error.message}`);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------

const ALLOWED_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

const app = express();
app.use(express.json({ limit: '2mb' }));

// DNS-rebinding defence: a browser on this machine must not be able to reach
// the bridge through an attacker-controlled hostname (GHSA-ppp5-vxwm-4cf7).
app.use((req, res, next) => {
  const host = (req.headers.host || '').split(':')[0];
  if (!ALLOWED_HOSTS.has(host)) {
    res.status(403).json({ error: 'forbidden host' });
    return;
  }
  next();
});

// Every request proves which Python runtime owns this bridge. Besides keeping
// the loopback API private, this prevents a duplicate port from ever crossing
// profile identities.
app.use((req, res, next) => {
  if (req.get('x-pilotage-bridge-token') !== INSTANCE_TOKEN) {
    res.status(403).json({ error: 'forbidden bridge instance' });
    return;
  }
  next();
});

app.get('/health', (_req, res) => {
  res.json({
    status: connected ? 'connected' : 'connecting',
    connected,
    pid: process.pid,
    me: meId,
    queued: messageQueue.length,
    sendReadReceipts: SEND_READ_RECEIPTS,
  });
});

app.get('/messages', (_req, res) => {
  res.json(messageQueue.splice(0, messageQueue.length));
});

app.post('/shutdown', (_req, res) => {
  res.json({ success: true });
  setImmediate(shutdown);
});

// Presence, not a message: deliberately outside the send queue, so a slow
// reply in flight never delays the indicator that explains the wait. (Hermes)
app.post('/typing', async (req, res) => {
  const { chatId } = req.body || {};
  if (!connected || !sock) {
    res.status(503).json({ error: 'not connected' });
    return;
  }
  if (!chatId) {
    res.status(400).json({ error: 'chatId is required' });
    return;
  }
  try {
    await sock.sendPresenceUpdate('composing', chatId);
    res.json({ success: true });
  } catch (error) {
    res.json({ success: false });
  }
});

// Mark an inbound message as read, but only once the runtime has accepted it
// through the allowlist — a blocked sender must not learn the agent is
// watching. (Hermes)
app.post('/read', async (req, res) => {
  if (!connected || !sock) {
    res.status(503).json({ error: 'not connected' });
    return;
  }
  const receiptKeys = inboundReadReceiptKeys({
    key: req.body?.key,
    enabled: SEND_READ_RECEIPTS,
  });
  if (receiptKeys.length === 0) {
    res.json({ success: true, marked: false });
    return;
  }
  try {
    await sock.readMessages(receiptKeys);
    res.json({ success: true, marked: true });
  } catch (error) {
    log(`read receipt failed: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

app.post('/send', async (req, res) => {
  const { chatId, message, replyTo } = req.body || {};
  if (!connected || !sock) {
    res.status(503).json({ error: 'not connected' });
    return;
  }
  if (!chatId || typeof message !== 'string' || !message.trim()) {
    res.status(400).json({ error: 'chatId and a non-empty message are required' });
    return;
  }

  try {
    const chunks = splitLongMessage(message);
    const messageIds = [];
    for (let i = 0; i < chunks.length; i += 1) {
      // Only the first chunk quotes: a split answer that quoted the same
      // message three times would read as three separate replies.
      const { content, options } = buildTextSendPayload(chunks[i], {
        replyTo: i === 0 ? replyTo : undefined,
        messageStore,
      });
      // eslint-disable-next-line no-await-in-loop
      const sent = await sendWithTimeout(chatId, content, options);
      if (sent?.key?.id) messageIds.push(sent.key.id);
      // Remembered so the person can quote our answer back at us.
      if (sent) messageStore.remember(sent);
      if (chunks.length > 1 && i < chunks.length - 1) {
        // eslint-disable-next-line no-await-in-loop
        await sleep(CHUNK_DELAY_MS);
      }
    }
    res.json({
      success: true,
      messageId: messageIds[messageIds.length - 1],
      messageIds,
    });
  } catch (error) {
    log(`send failed: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// Hermes' native-file handoff. Python has already resolved the path and
// confined it to this profile's workspace; the private, token-authenticated
// bridge only turns those bytes into the correct Baileys payload.
app.post('/send-media', async (req, res) => {
  const { chatId, filePath, mediaType, fileName } = req.body || {};
  if (!connected || !sock) {
    res.status(503).json({ error: 'not connected' });
    return;
  }
  if (!chatId || typeof filePath !== 'string' || !filePath) {
    res.status(400).json({ error: 'chatId and filePath are required' });
    return;
  }
  if (mediaType && !['image', 'video', 'document'].includes(mediaType)) {
    res.status(400).json({ error: 'unsupported mediaType' });
    return;
  }

  try {
    if (!existsSync(filePath)) {
      res.status(404).json({ error: 'file not found' });
      return;
    }
    const content = mediaPayloadForFile({
      buffer: readFileSync(filePath),
      filePath,
      mediaType,
      fileName,
    });
    if (!content) {
      res.status(400).json({ error: 'unsupported media file' });
      return;
    }
    const sent = await sendWithTimeout(chatId, content);
    if (sent) messageStore.remember(sent);
    res.json({ success: true, messageId: sent?.key?.id });
  } catch (error) {
    log(`media send failed: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

let server = null;
if (PAIR_ONLY) {
  log(`pairing mode (session: ${SESSION_DIR})`);
} else {
  server = app.listen(PORT, '127.0.0.1', () => {
    log(`listening on http://127.0.0.1:${PORT} (session: ${SESSION_DIR}, media: ${MEDIA_DIR})`);
  });
}

function shutdown() {
  try { server?.close(); } catch { /* already down */ }
  try { sock?.end?.(undefined); } catch { /* already down */ }
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

startSocket().catch((error) => {
  log(`failed to start: ${error.message}`);
  process.exit(1);
});
