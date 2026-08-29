/**
 * Pilotage WhatsApp bridge — walking skeleton.
 *
 * Holds the WhatsApp connection in Node (Baileys is Node-only), and exposes a
 * loopback HTTP API to the Python runtime:
 *
 *   GET  /health    -> { status, pid, connected, me }
 *   GET  /messages  -> durably claims inbound work
 *   POST /messages/ack -> completes claimed work
 *   POST /messages/release -> returns failed work for retry
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
  BufferJSON,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  initAuthCreds,
  proto,
} from '@whiskeysockets/baileys';

import {
  expandWhatsAppIdentifiers,
  matchesAllowedUser,
  normalizeWhatsAppIdentifier,
  parseAllowedUsers,
} from './allowlist.js';
import {
  buildEditSendPayload,
  buildTextSendPayload,
  createBoundedMessageStore,
  createCredentialSaveCoordinator,
  createDurableInboundQueue,
  createIdentityRedactor,
  createReconnectScheduler,
  createVersionResolver,
  drainTasksForShutdown,
  extractBridgeEvent,
  flushCredentialSavesForShutdown,
  hasDownloadableMedia,
  INBOUND_MEDIA_LIMIT_BYTES,
  inboundReadReceiptKeys,
  mediaPayloadForFile,
  normalizeWhatsAppId,
  resolveInboundClaimIdentities,
  useAtomicMultiFileAuthState,
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
const LOG_KEY_FILE = readArg('log-key', '');
const INBOUND_QUEUE_DIR = readArg('inbound-queue', '');
const PAIR_ONLY = process.argv.includes('--pair-only');
// Inbound media is written here, outside the session directory: re-pairing
// deletes the session, and a cached voice note should not depend on that.
const MEDIA_DIR = readArg('media', './media');
// Read receipts (the blue ticks) are off unless the operator asks for them, so
// an agent watching a chat does not silently mark everything as read. (Hermes)
const SEND_READ_RECEIPTS = readFlag('read-receipts', 'PILOTAGE_SEND_READ_RECEIPTS', false);
const ALLOWED_USERS = parseAllowedUsers(process.env.PILOTAGE_ALLOWED_SENDERS || '');

if (!PAIR_ONLY && !INSTANCE_TOKEN) {
  console.error('[bridge] --instance-token is required');
  process.exit(2);
}
if (!PAIR_ONLY && !INBOUND_QUEUE_DIR) {
  console.error('[bridge] --inbound-queue is required');
  process.exit(2);
}
if (!LOG_KEY_FILE) {
  console.error('[bridge] --log-key is required');
  process.exit(2);
}

let identityRedactor;
try {
  identityRedactor = createIdentityRedactor(readFileSync(LOG_KEY_FILE));
} catch {
  console.error('[bridge] identity-log key is missing or corrupt');
  process.exit(2);
}

const CACHE_DIRS = {
  image: path.join(MEDIA_DIR, 'images'),
  document: path.join(MEDIA_DIR, 'documents'),
  audio: path.join(MEDIA_DIR, 'audio'),
};

const QUEUE_HIGH_WATER_MARK = 100;
const QUEUE_CLAIM_BATCH_SIZE = 25;
const MAX_MESSAGE_LENGTH = 4096;
const CHUNK_DELAY_MS = 300;
const SEND_TIMEOUT_MS = 60_000;
const VERSION_FETCH_TIMEOUT_MS = 15_000;
const RECONNECT_DELAY_MS = 3_000;
const RESTART_DELAY_MS = 1_000;
const PAIR_SETTLE_MS = 2_000;
const SHUTDOWN_CREDENTIAL_FLUSH_MS = 5_000;
const SHUTDOWN_INBOUND_DRAIN_MS = 4_000;

const logger = pino({ level: 'silent' });

mkdirSync(SESSION_DIR, { recursive: true });
for (const dir of Object.values(CACHE_DIRS)) mkdirSync(dir, { recursive: true });

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let sock = null;
let connected = false;
let meId = null;
let activeSaveCreds = null;
let shuttingDown = false;
let inboundIntakeClosed = false;
let shutdownExitCode = 0;
const activeInboundTasks = new Set();

function log(...args) {
  console.log(`[bridge] ${identityRedactor.redact(args.join(' '))}`);
}

const inboundQueue = PAIR_ONLY ? null : createDurableInboundQueue(
  INBOUND_QUEUE_DIR,
  {
    highWaterMark: QUEUE_HIGH_WATER_MARK,
    claimBatchSize: QUEUE_CLAIM_BATCH_SIZE,
    log,
  },
);
const inboundQueueReady = inboundQueue?.initialize() || Promise.resolve();
const pendingMediaFences = new Map();
let mediaFenceSequence = 0;

const credentialSaves = createCredentialSaveCoordinator({ log });
let pairingFinalizing = false;

async function finishPairOnly(saveCreds, pairingSocket) {
  if (pairingFinalizing) return;
  pairingFinalizing = true;

  try {
    // Preserve Hermes' proven post-connect settling window for Baileys' other
    // auth-state work, then put a real save barrier after it.
    await new Promise((resolve) => setTimeout(resolve, PAIR_SETTLE_MS));
    await credentialSaves.flush(saveCreds);
  } catch (error) {
    log(`pairing failed: credentials could not be saved (${error.message})`);
    process.exitCode = 1;
    try { await pairingSocket?.end?.(undefined); } catch { /* already down */ }
    return;
  }

  log('pairing complete; credentials saved');
  process.exitCode = 0;
  try {
    await pairingSocket?.end?.(undefined);
  } catch (error) {
    log(`pairing shutdown failed: ${error.message}`);
    process.exitCode = 1;
  }
}

// ---------------------------------------------------------------------------
// Connection guards (Hermes bridge_helpers.js)
// ---------------------------------------------------------------------------

// Both guards exist because Baileys can hang rather than fail: startSocket()
// awaits network I/O before it registers any handler, and the version fetch
// has no AbortSignal. See bridge_helpers.js for the full reasoning.
const scheduleReconnect = createReconnectScheduler(() => startSocket(), {
  log,
  shouldReconnect: () => !shuttingDown,
});
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

function describeInbound(msg) {
  const chatId = msg.key?.remoteJid;
  if (!chatId || chatId === 'status@broadcast') return null;

  const isGroup = chatId.endsWith('@g.us');
  const senderId = (isGroup ? msg.key?.participant : chatId) || chatId;
  const senderPn = msg.key?.senderPn || msg.key?.participantPn || null;
  const senderNumber = digitsOf(senderPn) || digitsOf(senderId);
  const identities = new Set();
  for (const candidate of [senderId, senderPn]) {
    for (const alias of expandWhatsAppIdentifiers(candidate, SESSION_DIR)) {
      identities.add(alias);
    }
  }
  return {
    chatId,
    claimIdentities: resolveInboundClaimIdentities({
      senderId,
      senderPn,
      identities: Array.from(identities),
    }),
    identities: Array.from(identities),
    isGroup,
    senderId,
    senderNumber,
    senderPn,
  };
}

function registerMediaFence(msg, description) {
  if (!hasDownloadableMedia(msg)) return null;
  const messageId = String(msg.key?.id || '');
  if (!messageId) return null;
  mediaFenceSequence += 1;
  const fenceId = `${process.pid}-${mediaFenceSequence}`;
  pendingMediaFences.set(fenceId, {
    fenceId,
    messageId,
    chatId: description.chatId,
    senderId: description.senderId,
    senderNumber: description.senderNumber,
    identities: description.identities,
    isGroup: description.isGroup,
  });
  return fenceId;
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
async function buildEvent(msg, description = describeInbound(msg)) {
  if (!description) return null;
  const {
    chatId,
    claimIdentities,
    identities,
    isGroup,
    senderId,
    senderNumber,
    senderPn,
  } = description;

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
      'stream',
      {},
      { logger, reuploadRequest: sock.updateMediaMessage },
    ),
    cacheDirs: CACHE_DIRS,
    mediaLimits: INBOUND_MEDIA_LIMIT_BYTES,
    log,
  });

  return {
    ...event,
    _pilotageClaimIdentities: claimIdentities,
    senderNumber,
    senderPn,
    identities,
    pushName: msg.pushName || '',
    timestamp: Number(msg.messageTimestamp) || 0,
  };
}

function requestFatalShutdown(message) {
  log(message);
  shuttingDown = true;
  inboundIntakeClosed = true;
  shutdownExitCode = 1;
  setImmediate(() => { void shutdown(1); });
}

async function handleMessagesUpsert({ messages, type }) {
  if (type !== 'notify' && type !== 'append') return;

  const accepted = [];
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
      const description = describeInbound(msg);
      if (!description) continue;
      const {
        senderId,
        senderPn,
      } = description;
      const senderAllowed = (
        matchesAllowedUser(senderId, ALLOWED_USERS, SESSION_DIR)
        || matchesAllowedUser(senderPn, ALLOWED_USERS, SESSION_DIR)
      );
      if (!senderAllowed) continue;
      // The person allowlist is the complete access policy: an authorized
      // person may use DMs or any group. Python repeats the sender gate and
      // applies the optional direct-mention rule.

      accepted.push({ description, fenceId: null, msg });
    } catch (error) {
      log(`failed to inspect an inbound message: ${error.message}`);
    }
  }

  // Register the entire accepted media burst before the first fsync or
  // download can yield back to the HTTP poller. Python can then hold an
  // already-durable text sibling for one bounded grace window.
  for (const item of accepted) {
    item.fenceId = registerMediaFence(item.msg, item.description);
  }

  try {
    for (const { description, fenceId, msg } of accepted) {
      try {
        messageStore.remember(msg);
        // eslint-disable-next-line no-await-in-loop
        const event = await buildEvent(msg, description);
        if (!event) continue;
        // Nothing to answer and nothing attached: a receipt, an empty
        // protocol message, or a form we do not read. (Hermes)
        if (!event.body && !event.hasMedia) continue;

        if (inboundQueue) {
          try {
            // The event is accepted only after its fsync+atomic-rename barrier.
            // Duplicate Baileys replays resolve to the same durable identity.
            // eslint-disable-next-line no-await-in-loop
            await inboundQueue.enqueue(event);
          } catch (error) {
            requestFatalShutdown(
              `durable inbound enqueue failed; stopping the bridge: ${error.message}`,
            );
            return;
          }
        }
      } catch (error) {
        log(`failed to read an inbound message: ${error.message}`);
      } finally {
        if (fenceId) pendingMediaFences.delete(fenceId);
      }
    }
  } finally {
    // A fatal queue write returns early; no unprocessed media fence may remain
    // visible while shutdown drains the other already-started upserts.
    for (const { fenceId } of accepted) {
      if (fenceId) pendingMediaFences.delete(fenceId);
    }
  }
}

function trackMessagesUpsert(update) {
  if (inboundIntakeClosed) return;
  const task = handleMessagesUpsert(update);
  activeInboundTasks.add(task);
  void task.then(
    () => activeInboundTasks.delete(task),
    (error) => {
      activeInboundTasks.delete(task);
      requestFatalShutdown(
        `inbound processing failed; stopping the bridge: ${error.message}`,
      );
    },
  );
}

// ---------------------------------------------------------------------------
// Socket
// ---------------------------------------------------------------------------

async function startSocket() {
  if (shuttingDown) return;
  await inboundQueueReady;
  if (shuttingDown) return;
  const { state, saveCreds } = await useAtomicMultiFileAuthState(
    SESSION_DIR,
    { BufferJSON, initAuthCreds, proto },
  );
  if (shuttingDown) return;
  activeSaveCreds = saveCreds;
  const version = await getWAVersion();
  if (shuttingDown) return;

  const socket = makeWASocket({
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
  sock = socket;

  sock.ev.on('creds.update', () => {
    void credentialSaves.queue(saveCreds).catch(() => {
      // The atomic writer preserved the last good state, but continuing with
      // credentials that are known not to be durable makes reconnect unsafe.
      log('credential persistence failed; stopping the bridge');
      if (shuttingDown) {
        process.exitCode = 1;
        return;
      }
      process.exit(1);
    });
  });

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      log('scan this QR code with WhatsApp > Linked devices');
      qrcode.generate(qr, { small: true });
    }

    if (connection === 'open') {
      connected = true;
      meId = socket.user?.id || null;
      log(`connected as ${meId}`);
      if (PAIR_ONLY) {
        void finishPairOnly(saveCreds, socket);
      }
      return;
    }

    if (connection === 'close') {
      connected = false;
      if (shuttingDown || pairingFinalizing) return;
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

  sock.ev.on('messages.upsert', trackMessagesUpsert);
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

app.get('/health', async (_req, res) => {
  try {
    await inboundQueueReady;
    const queue = await inboundQueue.status();
    const payload = {
      status: queue.healthy ? (connected ? 'connected' : 'connecting') : 'unhealthy',
      connected,
      pid: process.pid,
      me: meId,
      queued: queue.depth,
      queue,
      sendReadReceipts: SEND_READ_RECEIPTS,
    };
    if (!queue.storageHealthy) {
      res.status(503).json(payload);
      return;
    }
    res.json(payload);
  } catch (error) {
    log(`durable inbound queue health check failed: ${error.message}`);
    res.status(503).json({ error: 'durable inbound queue unavailable' });
  }
});

app.get('/messages', async (_req, res) => {
  try {
    await inboundQueueReady;
    const messages = await inboundQueue.claim();
    const queue = await inboundQueue.status();
    const mediaFences = Array.from(pendingMediaFences.values());
    res.json({ messages, mediaFences, queue });
  } catch (error) {
    log(`durable inbound claim failed: ${error.message}`);
    res.status(503).json({ error: 'durable inbound queue unavailable' });
  }
});

app.post('/messages/ack', async (req, res) => {
  const claims = req.body?.claims;
  if (!Array.isArray(claims)) {
    res.status(400).json({ error: 'claims must be an array' });
    return;
  }
  try {
    const settled = await inboundQueue.ack(claims);
    res.json({ success: true, settled });
  } catch (error) {
    log(`durable inbound acknowledgement failed: ${error.message}`);
    res.status(503).json({ error: 'durable inbound queue unavailable' });
  }
});

app.post('/messages/release', async (req, res) => {
  const claims = req.body?.claims;
  if (!Array.isArray(claims)) {
    res.status(400).json({ error: 'claims must be an array' });
    return;
  }
  try {
    const released = await inboundQueue.release(claims);
    res.json({ success: true, released });
  } catch (error) {
    log(`durable inbound release failed: ${error.message}`);
    res.status(503).json({ error: 'durable inbound queue unavailable' });
  }
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

// Edit one previously sent text message. Progress updates use this route so a
// long turn owns one bubble instead of appending a new one every interval.
app.post('/edit', async (req, res) => {
  const { chatId, messageId, message } = req.body || {};
  if (!connected || !sock) {
    res.status(503).json({ error: 'not connected' });
    return;
  }
  if (
    !chatId
    || typeof messageId !== 'string'
    || !messageId
    || typeof message !== 'string'
    || !message.trim()
  ) {
    res.status(400).json({
      error: 'chatId, messageId, and a non-empty message are required',
    });
    return;
  }

  try {
    const chunks = splitLongMessage(message);
    if (chunks.length !== 1) {
      res.status(400).json({ error: 'edited message must fit one chunk' });
      return;
    }
    await sendWithTimeout(
      chatId,
      buildEditSendPayload(chatId, messageId, chunks[0]),
    );
    res.json({ success: true, messageId });
  } catch (error) {
    log(`edit failed: ${error.message}`);
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

let shutdownPromise = null;

async function shutdown(requestedExitCode = 0) {
  shutdownExitCode = Math.max(shutdownExitCode, Number(requestedExitCode) || 0);
  if (shutdownPromise) return shutdownPromise;
  shuttingDown = true;
  const closingSocket = sock;
  const closingSaveCreds = activeSaveCreds;
  // Freeze the accepted-work set synchronously. Once shutdown starts, no
  // event may enter between the task snapshot and the transport close.
  inboundIntakeClosed = true;
  try {
    closingSocket?.ev?.off?.('messages.upsert', trackMessagesUpsert);
  } catch (error) {
    log(`inbound event-source fence failed during shutdown: ${error.message}`);
    shutdownExitCode = 1;
  }
  shutdownPromise = (async () => {
    try { server?.close(); } catch { /* already down */ }
    const inboundDrained = await drainTasksForShutdown(activeInboundTasks, {
      timeoutMs: SHUTDOWN_INBOUND_DRAIN_MS,
    });
    if (!inboundDrained) {
      log(
        `shutdown inbound drain timed out with ${activeInboundTasks.size} task(s)`,
      );
      shutdownExitCode = 1;
    }
    try {
      await flushCredentialSavesForShutdown(
        credentialSaves,
        closingSaveCreds,
        {
          timeoutMs: SHUTDOWN_CREDENTIAL_FLUSH_MS,
          closeSocket: async () => {
            try {
              await closingSocket?.end?.(undefined);
            } catch (error) {
              log(`socket close failed during shutdown: ${error.message}`);
              shutdownExitCode = 1;
            }
          },
        },
      );
    } catch (error) {
      log(`credential flush failed during shutdown: ${error.message}`);
      shutdownExitCode = 1;
    }
    process.exit(shutdownExitCode);
  })();
  return shutdownPromise;
}

process.on('SIGINT', () => { void shutdown(); });
process.on('SIGTERM', () => { void shutdown(); });

startSocket().catch((error) => {
  if (shuttingDown) return;
  log(`failed to start: ${error.message}`);
  process.exit(1);
});
