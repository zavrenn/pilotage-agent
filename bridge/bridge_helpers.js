import path from 'path';
import {
  chmod,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  stat,
  unlink,
} from 'fs/promises';
import { createHash, createHmac, randomBytes } from 'crypto';

const WHATSAPP_JID_RE = /(?:\d[\d:+-]{4,})@(?:s\.whatsapp\.net|g\.us|lid)(?![a-z0-9.])/gi;
const MEBIBYTE = 1024 * 1024;

// Fixed production ceilings. Metadata is checked before the CDN download and
// the decrypted stream is bounded again while it is written to disk.
export const INBOUND_MEDIA_LIMIT_BYTES = Object.freeze({
  image: 20 * MEBIBYTE,
  sticker: 20 * MEBIBYTE,
  audio: 25 * MEBIBYTE,
  ptt: 25 * MEBIBYTE,
  document: 50 * MEBIBYTE,
  video: 100 * MEBIBYTE,
  gif: 100 * MEBIBYTE,
});

export function mediaLengthBytes(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : null;
  }
  let written;
  if (typeof value === 'bigint' || typeof value === 'string') {
    written = String(value).trim();
  } else if (value && typeof value.toString === 'function') {
    written = String(value.toString()).trim();
  } else {
    return null;
  }
  if (!/^\d+$/.test(written)) return null;
  try {
    const parsed = BigInt(written);
    if (parsed <= 0n) return null;
    if (parsed > BigInt(Number.MAX_SAFE_INTEGER)) return Number.POSITIVE_INFINITY;
    return Number(parsed);
  } catch {
    return null;
  }
}

export function createIdentityRedactor(key) {
  const secret = Buffer.from(key || []);
  if (secret.length !== 32) {
    throw new Error('identity-log key must contain exactly 32 bytes');
  }

  function alias(value, namespace = 'wa') {
    const safeNamespace = String(namespace).toLowerCase().replace(/[^a-z0-9_-]/g, '-') || 'id';
    const digest = createHmac('sha256', secret)
      .update(`${safeNamespace}\0${String(value || '')}`)
      .digest('hex')
      .slice(0, 12);
    return `[${safeNamespace}:${digest}]`;
  }

  function redact(value) {
    return String(value || '').replace(WHATSAPP_JID_RE, (identity) => alias(identity));
  }

  return { alias, redact };
}

const atomicPathLocks = new Map();
const DEFAULT_ATOMIC_FILE_OPS = {
  chmod,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  stat,
  unlink,
};

async function withAtomicPathLock(filePath, work) {
  const normalized = path.resolve(filePath);
  const previous = atomicPathLocks.get(normalized) || Promise.resolve();
  let release;
  const tail = new Promise((resolve) => { release = resolve; });
  atomicPathLocks.set(normalized, tail);
  await previous.catch(() => {});
  try {
    return await work();
  } finally {
    release();
    if (atomicPathLocks.get(normalized) === tail) atomicPathLocks.delete(normalized);
  }
}

async function syncDirectory(directory, fileOps) {
  let handle;
  try {
    handle = await fileOps.open(directory, 'r');
    await handle.sync();
  } catch (error) {
    // Linux is the production contract and supports directory fsync. Some
    // development hosts do not; only those hosts may degrade this barrier.
    if (process.platform === 'linux') throw error;
  } finally {
    await handle?.close().catch(() => {});
  }
}

export function createAtomicJsonFileStore(folder, { fileOps: overrides = {} } = {}) {
  const fileOps = { ...DEFAULT_ATOMIC_FILE_OPS, ...overrides };

  async function initialize() {
    const info = await fileOps.stat(folder).catch((error) => {
      if (error?.code === 'ENOENT') return null;
      throw error;
    });
    if (info && !info.isDirectory()) {
      throw new Error(`WhatsApp authentication path is not a directory: ${folder}`);
    }
    if (!info) await fileOps.mkdir(folder, { recursive: true, mode: 0o700 });
    await fileOps.chmod(folder, 0o700).catch((error) => {
      if (process.platform !== 'win32') throw error;
    });
  }

  function safeName(file) {
    return String(file || '').replace(/\//g, '__').replace(/:/g, '-');
  }

  async function writeData(data, file, replacer) {
    const name = safeName(file);
    const target = path.join(folder, name);
    return withAtomicPathLock(target, async () => {
      const serialized = JSON.stringify(data, replacer);
      const temporary = path.join(
        folder,
        `.${name}.${process.pid}.${randomBytes(8).toString('hex')}.tmp`,
      );
      let handle;
      try {
        handle = await fileOps.open(temporary, 'wx', 0o600);
        await handle.writeFile(serialized, { encoding: 'utf8' });
        await handle.sync();
        await handle.close();
        handle = null;
        await fileOps.rename(temporary, target);
        await fileOps.chmod(target, 0o600).catch((error) => {
          if (process.platform !== 'win32') throw error;
        });
        await syncDirectory(folder, fileOps);
      } finally {
        await handle?.close().catch(() => {});
        await fileOps.unlink(temporary).catch((error) => {
          if (error?.code !== 'ENOENT') throw error;
        });
      }
    });
  }

  async function readData(file, reviver) {
    const name = safeName(file);
    const target = path.join(folder, name);
    return withAtomicPathLock(target, async () => {
      let serialized;
      try {
        serialized = await fileOps.readFile(target, { encoding: 'utf8' });
      } catch (error) {
        if (error?.code === 'ENOENT') return null;
        throw error;
      }
      try {
        return JSON.parse(serialized, reviver);
      } catch (error) {
        throw new Error(`Corrupt WhatsApp authentication file: ${name}`, { cause: error });
      }
    });
  }

  async function removeData(file) {
    const name = safeName(file);
    const target = path.join(folder, name);
    return withAtomicPathLock(target, async () => {
      try {
        await fileOps.unlink(target);
      } catch (error) {
        if (error?.code === 'ENOENT') return;
        throw error;
      }
      await syncDirectory(folder, fileOps);
    });
  }

  return { initialize, readData, removeData, writeData };
}

export async function useAtomicMultiFileAuthState(
  folder,
  { BufferJSON, initAuthCreds, proto, fileOps } = {},
) {
  if (!BufferJSON?.replacer || !BufferJSON?.reviver || typeof initAuthCreds !== 'function') {
    throw new TypeError('Baileys auth serialization dependencies are required');
  }
  const files = createAtomicJsonFileStore(folder, { fileOps });
  await files.initialize();
  const creds = (await files.readData('creds.json', BufferJSON.reviver)) || initAuthCreds();

  return {
    state: {
      creds,
      keys: {
        get: async (type, ids) => {
          const data = {};
          await Promise.all(ids.map(async (id) => {
            let value = await files.readData(`${type}-${id}.json`, BufferJSON.reviver);
            if (type === 'app-state-sync-key' && value) {
              value = proto.Message.AppStateSyncKeyData.fromObject(value);
            }
            data[id] = value;
          }));
          return data;
        },
        set: async (data) => {
          const tasks = [];
          for (const category of Object.keys(data || {})) {
            for (const [id, value] of Object.entries(data[category] || {})) {
              const file = `${category}-${id}.json`;
              tasks.push(
                value
                  ? files.writeData(value, file, BufferJSON.replacer)
                  : files.removeData(file),
              );
            }
          }
          await Promise.all(tasks);
        },
      },
    },
    saveCreds: () => files.writeData(creds, 'creds.json', BufferJSON.replacer),
  };
}

function bareInboundIdentity(value) {
  return String(value || '')
    .trim()
    .replace(/:.*@/, '@')
    .replace(/@.*/, '')
    .replace(/^\+/, '');
}

function typedInboundIdentity(value, assumedType = '') {
  const written = String(value || '').trim().toLowerCase().replace(/:.*@/, '@');
  const explicit = /^(pn|lid):(\d+)$/.exec(written);
  if (explicit) return `${explicit[1]}:${explicit[2]}`;
  const jid = /^(\d+)@(s\.whatsapp\.net|lid)$/.exec(written);
  if (jid) return `${jid[2] === 'lid' ? 'lid' : 'pn'}:${jid[1]}`;
  if (/^\d+$/.test(written) && ['pn', 'lid'].includes(assumedType)) {
    return `${assumedType}:${written}`;
  }
  return '';
}

export function resolveInboundClaimIdentities({
  senderId,
  senderPn = null,
  identities = [],
} = {}) {
  const resolved = new Set();
  const sender = typedInboundIdentity(senderId);
  if (sender) resolved.add(sender);
  const phone = typedInboundIdentity(senderPn, 'pn');
  if (phone) resolved.add(phone);

  const senderType = sender.startsWith('pn:')
    ? 'pn'
    : (sender.startsWith('lid:') ? 'lid' : '');
  const counterpartType = senderType === 'pn' ? 'lid' : 'pn';
  const knownValues = new Set(
    Array.from(resolved).map((value) => value.slice(value.indexOf(':') + 1)),
  );
  for (const value of Array.isArray(identities) ? identities : []) {
    const bare = bareInboundIdentity(value);
    if (!/^\d+$/.test(bare) || knownValues.has(bare) || !senderType) continue;
    // expandWhatsAppIdentifiers only adds values read from Baileys' durable
    // PN<->LID mapping files. That mapping is the evidence for this type link.
    resolved.add(`${counterpartType}:${bare}`);
  }
  return Array.from(resolved).sort();
}

function inboundIdentityAliases(event) {
  const aliases = new Set();
  const sender = typedInboundIdentity(event?.senderId);
  if (sender) aliases.add(sender);
  const senderPn = typedInboundIdentity(event?.senderPn, 'pn');
  if (senderPn) aliases.add(senderPn);
  for (const value of Array.isArray(event?._pilotageClaimIdentities)
    ? event._pilotageClaimIdentities
    : []) {
    const typed = typedInboundIdentity(value);
    if (typed) aliases.add(typed);
  }
  return Array.from(aliases)
    .sort((left, right) => {
      const leftRank = left.startsWith('pn:') ? 0 : 1;
      const rightRank = right.startsWith('pn:') ? 0 : 1;
      return leftRank - rightRank || left.localeCompare(right);
    })
    .slice(0, 8);
}

function hashedInboundIdentity(value) {
  return createHash('sha256').update(value).digest('hex');
}

function claimIdsFor(event) {
  const messageId = String(event?.messageId || '').trim();
  const chatId = String(event?.chatId || '').trim();
  if (!messageId || !chatId) {
    throw new Error('Inbound WhatsApp event has no stable message identity');
  }

  const aliases = inboundIdentityAliases(event);
  if (aliases.length === 0) {
    throw new Error('Inbound WhatsApp event has no stable sender identity');
  }
  const isGroup = Boolean(event?.isGroup) || chatId.endsWith('@g.us');
  const groupId = bareInboundIdentity(chatId);
  if (isGroup && !groupId) {
    throw new Error('Inbound WhatsApp group event has no stable chat identity');
  }

  const canonical = aliases[0];
  const scopedId = (participant) => hashedInboundIdentity(
    isGroup
      ? `pilotage-wa-inbound-v2\0group\0${groupId}\0${participant}\0${messageId}`
      : `pilotage-wa-inbound-v2\0direct\0${participant}\0${messageId}`,
  );
  const primary = scopedId(canonical);
  const compatible = new Set([primary]);

  // If a mapping file appears after an event was first accepted, either alias
  // must still find the earlier v2 claim.
  for (const alias of aliases) compatible.add(scopedId(alias));

  // Before v2 the filename hashed raw chatId+senderId+messageId. Check the
  // current representation plus only typed aliases backed by a WhatsApp
  // mapping. Equal local parts in PN and LID namespaces are not proof that two
  // identities belong to the same person.
  const legacy = (legacyChat, legacySender) => hashedInboundIdentity(
    `${legacyChat}\0${legacySender}\0${messageId}`,
  );
  compatible.add(legacy(chatId, String(event?.senderId || '').trim()));
  for (const alias of aliases) {
    const separator = alias.indexOf(':');
    const type = alias.slice(0, separator);
    const value = alias.slice(separator + 1);
    const participant = `${value}@${type === 'lid' ? 'lid' : 's.whatsapp.net'}`;
    compatible.add(
      legacy(isGroup ? chatId : participant, participant),
    );
  }
  return { primary, compatible: Array.from(compatible) };
}

export function createDurableInboundQueue(
  root,
  {
    fileOps: overrides = {},
    highWaterMark = 100,
    claimBatchSize = 25,
    doneMaxEntries = 10_000,
    now = () => Date.now(),
    log = () => {},
  } = {},
) {
  const fileOps = { ...DEFAULT_ATOMIC_FILE_OPS, ...overrides };
  const pendingDir = path.join(root, 'pending');
  const claimedDir = path.join(root, 'claimed');
  const doneDir = path.join(root, 'done');
  const pending = createAtomicJsonFileStore(pendingDir, { fileOps });
  const claimed = createAtomicJsonFileStore(claimedDir, { fileOps });
  const done = createAtomicJsonFileStore(doneDir, { fileOps });
  let tail = Promise.resolve();
  let storageFailed = false;
  let overflowed = false;
  let doneCount = 0;
  const parsedDoneMaxEntries = Number(doneMaxEntries);
  const boundedDoneMaxEntries = (
    Number.isFinite(parsedDoneMaxEntries) && parsedDoneMaxEntries >= 1
      ? Math.floor(parsedDoneMaxEntries)
      : 10_000
  );

  function serialized(work) {
    const current = tail.catch(() => {}).then(work);
    tail = current;
    return current;
  }

  function fileName(claimId) {
    return `${claimId}.json`;
  }

  async function exists(directory, name) {
    try {
      await fileOps.stat(path.join(directory, name));
      return true;
    } catch (error) {
      if (error?.code === 'ENOENT') return false;
      throw error;
    }
  }

  async function names(directory) {
    return (await fileOps.readdir(directory))
      .filter((name) => /^[a-f0-9]{64}\.json$/.test(name));
  }

  async function pruneDoneUnlocked() {
    const completedNames = await names(doneDir);
    doneCount = completedNames.length;
    if (doneCount <= boundedDoneMaxEntries) return;

    const lowWaterMark = Math.max(
      1,
      Math.floor(boundedDoneMaxEntries * 0.9),
    );
    const completed = await Promise.all(completedNames.map(async (name) => {
      const record = await done.readData(name);
      const completedAt = Number(record?.completedAt);
      if (!Number.isFinite(completedAt)) {
        throw new Error(`Corrupt durable inbound completion: ${name}`);
      }
      return { name, completedAt };
    }));
    completed.sort((left, right) => (
      left.completedAt - right.completedAt
      || left.name.localeCompare(right.name)
    ));
    const remove = completed.slice(0, doneCount - lowWaterMark);
    for (let offset = 0; offset < remove.length; offset += 50) {
      await Promise.all(remove.slice(offset, offset + 50).map(async ({ name }) => {
        try {
          await fileOps.unlink(path.join(doneDir, name));
        } catch (error) {
          if (error?.code !== 'ENOENT') throw error;
        }
      }));
    }
    if (remove.length) await syncDirectory(doneDir, fileOps);
    doneCount -= remove.length;
    log(`compacted durable inbound completions to ${doneCount} entries`);
  }

  async function depthUnlocked() {
    const [waiting, inFlight] = await Promise.all([
      names(pendingDir),
      names(claimedDir),
    ]);
    return { pending: waiting.length, claimed: inFlight.length };
  }

  async function updateOverflowUnlocked() {
    const counts = await depthUnlocked();
    const overloaded = counts.pending + counts.claimed > highWaterMark;
    if (overloaded && !overflowed) {
      log(`durable inbound queue exceeded its ${highWaterMark}-message high-water mark`);
    } else if (!overloaded && overflowed) {
      log('durable inbound queue recovered below its high-water mark');
    }
    overflowed = overloaded;
    return counts;
  }

  async function initialize() {
    await fileOps.mkdir(root, { recursive: true, mode: 0o700 });
    await fileOps.chmod(root, 0o700).catch((error) => {
      if (process.platform !== 'win32') throw error;
    });
    await Promise.all([pending.initialize(), claimed.initialize(), done.initialize()]);

    await serialized(async () => {
      for (const name of await names(claimedDir)) {
        if (await exists(doneDir, name)) {
          await claimed.removeData(name);
          continue;
        }
        if (await exists(pendingDir, name)) {
          throw new Error(`Duplicate durable inbound claim: ${name}`);
        }
        await fileOps.rename(path.join(claimedDir, name), path.join(pendingDir, name));
      }
      await syncDirectory(claimedDir, fileOps);
      await syncDirectory(pendingDir, fileOps);
      await pruneDoneUnlocked();
      await updateOverflowUnlocked();
    });
  }

  async function enqueue(event) {
    try {
      return await serialized(async () => {
        const { primary: claimId, compatible } = claimIdsFor(event);
        const name = fileName(claimId);
        const occupied = await Promise.all(compatible.map(async (compatibleId) => {
          const compatibleName = fileName(compatibleId);
          const locations = await Promise.all([
            exists(doneDir, compatibleName),
            exists(pendingDir, compatibleName),
            exists(claimedDir, compatibleName),
          ]);
          return locations.some(Boolean) ? compatibleId : null;
        }));
        const duplicate = occupied.find(Boolean);
        if (duplicate) {
          return { status: 'duplicate', claimId: duplicate };
        }
        await pending.writeData(
          {
            version: 1,
            identity: claimId,
            acceptedAt: now(),
            event,
          },
          name,
        );
        await updateOverflowUnlocked();
        return { status: 'accepted', claimId };
      });
    } catch (error) {
      storageFailed = true;
      throw error;
    }
  }

  async function claim(limit = claimBatchSize) {
    try {
      return await serialized(async () => {
        const records = [];
        for (const name of await names(pendingDir)) {
          if (await exists(doneDir, name)) {
            await pending.removeData(name);
            continue;
          }
          const record = await pending.readData(name);
          if (!record || record.identity !== name.slice(0, -5) || !record.event) {
            throw new Error(`Corrupt durable inbound event: ${name}`);
          }
          records.push({ name, record });
        }
        records.sort((left, right) => (
          Number(left.record.acceptedAt) - Number(right.record.acceptedAt)
          || left.name.localeCompare(right.name)
        ));

        const output = [];
        for (const { name, record } of records.slice(0, Math.max(0, Number(limit) || 0))) {
          await fileOps.rename(path.join(pendingDir, name), path.join(claimedDir, name));
          output.push({
            ...record.event,
            _pilotageClaimId: record.identity,
          });
        }
        if (output.length) {
          await syncDirectory(pendingDir, fileOps);
          await syncDirectory(claimedDir, fileOps);
        }
        return output;
      });
    } catch (error) {
      storageFailed = true;
      throw error;
    }
  }

  function validClaimIds(values) {
    return Array.from(new Set((values || []).map(String)))
      .filter((value) => /^[a-f0-9]{64}$/.test(value));
  }

  async function ack(values) {
    try {
      return await serialized(async () => {
        let settled = 0;
        for (const claimId of validClaimIds(values)) {
          const name = fileName(claimId);
          if (await exists(doneDir, name)) {
            settled += 1;
            continue;
          }
          const known = await exists(claimedDir, name) || await exists(pendingDir, name);
          if (!known) continue;
          await done.writeData({ completedAt: now() }, name);
          doneCount += 1;
          await claimed.removeData(name);
          await pending.removeData(name);
          settled += 1;
        }
        if (doneCount > boundedDoneMaxEntries) await pruneDoneUnlocked();
        await updateOverflowUnlocked();
        return settled;
      });
    } catch (error) {
      storageFailed = true;
      throw error;
    }
  }

  async function release(values) {
    try {
      return await serialized(async () => {
        let released = 0;
        for (const claimId of validClaimIds(values)) {
          const name = fileName(claimId);
          if (await exists(doneDir, name) || await exists(pendingDir, name)) {
            released += 1;
            continue;
          }
          if (!await exists(claimedDir, name)) continue;
          await fileOps.rename(path.join(claimedDir, name), path.join(pendingDir, name));
          released += 1;
        }
        if (released) {
          await syncDirectory(claimedDir, fileOps);
          await syncDirectory(pendingDir, fileOps);
        }
        return released;
      });
    } catch (error) {
      storageFailed = true;
      throw error;
    }
  }

  async function status() {
    return serialized(async () => {
      const counts = await updateOverflowUnlocked();
      return {
        ...counts,
        depth: counts.pending + counts.claimed,
        healthy: !storageFailed && !overflowed,
        storageHealthy: !storageFailed,
        overflowed,
        highWaterMark,
        completed: doneCount,
        completedMaxEntries: boundedDoneMaxEntries,
      };
    });
  }

  return { ack, claim, enqueue, initialize, release, status };
}

/**
 * Track Baileys credential writes and provide a real completion barrier.
 * Baileys owns file ordering through its per-path mutex, so every save must be
 * invoked immediately to enter that mutex before a reconnect can read state.
 */
export function createCredentialSaveCoordinator({ log = () => {} } = {}) {
  const pending = new Set();
  let firstError = null;
  let revision = 0;

  function queue(saveCreds) {
    if (typeof saveCreds !== 'function') {
      throw new TypeError('saveCreds must be a function');
    }

    revision += 1;
    let write;
    try {
      write = Promise.resolve(saveCreds());
    } catch (error) {
      write = Promise.reject(error);
    }

    let tracked;
    tracked = write.catch((error) => {
      const normalized = error instanceof Error ? error : new Error(String(error));
      firstError ||= normalized;
      log(`credential save failed: ${normalized.message}`);
    }).finally(() => {
      pending.delete(tracked);
    });
    pending.add(tracked);
    return write;
  }

  async function drain() {
    while (pending.size) {
      await Promise.all([...pending]);
    }
  }

  async function flush(saveCreds) {
    while (true) {
      await drain();
      queue(saveCreds);
      const finalRevision = revision;
      await drain();
      if (revision === finalRevision) break;
    }
    if (firstError) throw firstError;
  }

  return { queue, flush };
}

export async function flushCredentialSavesForShutdown(
  coordinator,
  saveCreds,
  { timeoutMs = 5_000, closeSocket = null } = {},
) {
  if (closeSocket !== null && typeof closeSocket !== 'function') {
    throw new TypeError('closeSocket must be a function');
  }
  const hasSaveCreds = typeof saveCreds === 'function';
  if (!hasSaveCreds && closeSocket === null) return;
  if (hasSaveCreds && (!coordinator || typeof coordinator.flush !== 'function')) {
    throw new TypeError('credential save coordinator is unavailable');
  }
  const boundedTimeout = Number.isFinite(timeoutMs) ? Math.max(1, timeoutMs) : 5_000;
  let timer;
  try {
    await Promise.race([
      (async () => {
        if (closeSocket) await closeSocket();
        if (hasSaveCreds) await coordinator.flush(saveCreds);
      })(),
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error('socket close or credential flush timed out during shutdown')),
          boundedTimeout,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

export async function drainTasksForShutdown(
  tasks,
  { timeoutMs = 4_000 } = {},
) {
  const active = Array.from(tasks || []).filter(
    (task) => task && typeof task.then === 'function',
  );
  if (active.length === 0) return true;

  const parsedTimeout = Number(timeoutMs);
  const boundedTimeout = Number.isFinite(parsedTimeout)
    ? Math.max(1, parsedTimeout)
    : 4_000;
  let timer;
  try {
    return await Promise.race([
      Promise.allSettled(active).then(() => true),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(false), boundedTimeout);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

export const MIME_MAP = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  webp: 'image/webp', gif: 'image/gif',
  mp4: 'video/mp4', mov: 'video/quicktime', avi: 'video/x-msvideo',
  mkv: 'video/x-matroska', '3gp': 'video/3gpp',
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

export function normalizeWhatsAppId(value) {
  if (!value) return '';
  return String(value).replace(':', '@');
}

export function getMessageContent(msg) {
  const content = msg?.message || {};
  if (content.ephemeralMessage?.message) return content.ephemeralMessage.message;
  if (content.viewOnceMessage?.message) return content.viewOnceMessage.message;
  if (content.viewOnceMessageV2?.message) return content.viewOnceMessageV2.message;
  if (content.documentWithCaptionMessage?.message) return content.documentWithCaptionMessage.message;
  if (content.templateMessage?.hydratedTemplate) return content.templateMessage.hydratedTemplate;
  if (content.buttonsMessage) return content.buttonsMessage;
  if (content.listMessage) return content.listMessage;
  return content;
}

export function hasDownloadableMedia(msg) {
  const content = getMessageContent(msg);
  return Boolean(
    content.imageMessage
    || content.videoMessage
    || content.audioMessage
    || content.pttMessage
    || content.documentMessage
    || content.stickerMessage
  );
}

export function getContextInfo(messageContent) {
  if (!messageContent || typeof messageContent !== 'object') return {};
  for (const value of Object.values(messageContent)) {
    if (value && typeof value === 'object' && value.contextInfo) {
      return value.contextInfo;
    }
  }
  return {};
}

export function createBoundedMessageStore(limit = 512) {
  const byId = new Map();

  function remember(msg) {
    const id = msg?.key?.id;
    if (!id) return;
    byId.delete(id);
    byId.set(id, msg);
    while (byId.size > limit) {
      const oldest = byId.keys().next().value;
      byId.delete(oldest);
    }
  }

  function get(id) {
    if (!id || !byId.has(id)) return null;
    const msg = byId.get(id);
    byId.delete(id);
    byId.set(id, msg);
    return msg;
  }

  return { remember, get };
}

export function pollCreationMessageSecret(pollCreation) {
  return pollCreation?.message?.messageContextInfo?.messageSecret
    || pollCreation?.messageContextInfo?.messageSecret
    || null;
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values || []) {
    const text = String(value || '').trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function pollUpdateForAggregation({
  pollUpdateMessage,
  pollUpdateMessageKey,
  pollCreation,
  decryptPollVote,
  getKeyAuthor,
  meId = 'me',
  pollCreatorJids = [],
  voterJids = [],
}) {
  if (!pollUpdateMessage) return null;
  const updateKey = pollUpdateMessage.pollUpdateMessageKey
    || pollUpdateMessageKey
    || pollUpdateMessage.key;
  if (!updateKey) return null;

  if (pollUpdateMessage.vote?.selectedOptions) {
    return {
      pollUpdateMessageKey: updateKey,
      vote: pollUpdateMessage.vote,
      senderTimestampMs: pollUpdateMessage.senderTimestampMs,
    };
  }

  const creationKey = pollUpdateMessage.pollCreationMessageKey;
  const secret = pollCreationMessageSecret(pollCreation);
  if (
    !creationKey?.id
    || !secret
    || !pollUpdateMessage.vote?.encPayload
    || !pollUpdateMessage.vote?.encIv
    || typeof decryptPollVote !== 'function'
    || typeof getKeyAuthor !== 'function'
  ) {
    return null;
  }

  // Baileys poll decryption keys include both creator and voter JIDs.  On
  // WhatsApp LID chats, the poll creator can be the linked-device LID even
  // when sock.user.id is the classic @s.whatsapp.net JID.  Try the exact
  // candidates the live bridge knows before falling back to the generic helper.
  const creatorCandidates = uniqueStrings([
    ...pollCreatorJids,
    getKeyAuthor(creationKey, meId),
  ]);
  const voterCandidates = uniqueStrings([
    ...voterJids,
    getKeyAuthor(updateKey, meId),
  ]);

  let lastError = null;
  for (const pollCreatorJid of creatorCandidates) {
    for (const voterJid of voterCandidates) {
      try {
        const vote = decryptPollVote(pollUpdateMessage.vote, {
          pollCreatorJid,
          pollMsgId: creationKey.id,
          pollEncKey: secret,
          voterJid,
        });
        return {
          pollUpdateMessageKey: updateKey,
          vote,
          senderTimestampMs: pollUpdateMessage.senderTimestampMs,
        };
      } catch (err) {
        lastError = err;
      }
    }
  }
  if (lastError) throw lastError;
  return null;
}

export function buildTextSendPayload(text, { replyTo, messageStore } = {}) {
  const content = { text };
  const options = {};
  const quoted = messageStore?.get(replyTo);
  if (quoted?.key && quoted?.message) {
    // Baileys expects quoted messages as sendMessage options, not inside the
    // message content payload. Keeping this split avoids silently sending a
    // literal/ignored `quoted` field instead of a native WhatsApp reply.
    options.quoted = quoted;
  }
  return { content, options };
}

export function buildLocationPayload({ latitude, longitude, name, address } = {}) {
  const lat = Number(latitude);
  const lon = Number(longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error('latitude and longitude must be numbers');
  }
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error('latitude/longitude out of range');
  }

  const location = {
    degreesLatitude: lat,
    degreesLongitude: lon,
  };
  if (name) location.name = String(name);
  if (address) location.address = String(address);
  return { location };
}

function textFromQuotedMessage(quotedMessage) {
  if (!quotedMessage) return '';
  if (quotedMessage.conversation) return quotedMessage.conversation;
  if (quotedMessage.extendedTextMessage?.text) return quotedMessage.extendedTextMessage.text;
  if (quotedMessage.imageMessage?.caption) return quotedMessage.imageMessage.caption;
  if (quotedMessage.videoMessage?.caption) return quotedMessage.videoMessage.caption;
  if (quotedMessage.documentMessage?.caption) return quotedMessage.documentMessage.caption;
  if (quotedMessage.documentMessage?.fileName) return `[Document: ${quotedMessage.documentMessage.fileName}]`;
  if (quotedMessage.locationMessage) return formatLocationText(quotedMessage.locationMessage, false);
  if (quotedMessage.contactMessage) return formatContactText(quotedMessage.contactMessage);
  if (quotedMessage.pollCreationMessage) return formatPollText(quotedMessage.pollCreationMessage);
  return '';
}

function mediaExtForMime(mime, fallback) {
  const normalized = String(mime || '').split(';', 1)[0].toLowerCase();
  const extMap = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/x-matroska': '.mkv',
    'audio/ogg': '.ogg',
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'application/pdf': '.pdf',
  };
  return extMap[normalized] || fallback;
}

class InboundMediaLimitError extends Error {
  constructor() {
    super('inbound media exceeds its configured limit');
    this.code = 'PILOTAGE_INBOUND_MEDIA_LIMIT';
  }
}

function isAsyncIterable(value) {
  return !!value && typeof value[Symbol.asyncIterator] === 'function';
}

async function writeWholeChunk(handle, chunk, position) {
  const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
  let offset = 0;
  while (offset < bytes.byteLength) {
    const result = await handle.write(
      bytes,
      offset,
      bytes.byteLength - offset,
      position + offset,
    );
    if (!result.bytesWritten) throw new Error('inbound media write made no progress');
    offset += result.bytesWritten;
  }
  return position + bytes.byteLength;
}

async function defaultWriteMediaFile({ buffer, stream, dir, prefix, ext, fileName, limit }) {
  await mkdir(dir, { recursive: true });
  let safeName = fileName ? `_${path.basename(fileName).replace(/[^a-zA-Z0-9._-]/g, '_')}` : '';
  if (safeName && ext && !path.extname(safeName)) {
    safeName = `${safeName}${ext}`;
  }
  const filePath = path.join(dir, `${prefix}_${randomBytes(6).toString('hex')}${safeName || ext}`);
  let handle;
  try {
    handle = await open(filePath, 'wx', 0o600);
    let written = 0;
    if (isAsyncIterable(stream)) {
      for await (const chunk of stream) {
        const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        if (limit > 0 && written + bytes.byteLength > limit) {
          if (typeof stream.destroy === 'function') stream.destroy();
          throw new InboundMediaLimitError();
        }
        written = await writeWholeChunk(handle, bytes, written);
      }
    } else {
      const length = Number(buffer?.byteLength ?? buffer?.length ?? 0);
      if (limit > 0 && length > limit) throw new InboundMediaLimitError();
      written = await writeWholeChunk(handle, buffer || Buffer.alloc(0), written);
    }
    await handle.close();
    handle = null;
    return filePath;
  } catch (error) {
    await handle?.close().catch(() => {});
    await unlink(filePath).catch(() => {});
    throw error;
  }
}

function formatLocationText(location, isLive) {
  const name = location.name || location.address || '';
  const lat = location.degreesLatitude ?? location.latitude;
  const lng = location.degreesLongitude ?? location.longitude;
  const kind = isLive ? 'Live location' : 'Location';
  const coords = lat !== undefined && lng !== undefined ? `${lat},${lng}` : '';
  return `[${kind}: ${[name, coords].filter(Boolean).join(' ')}]`;
}

function locationMetadata(location, isLive) {
  return {
    name: location.name || '',
    address: location.address || '',
    latitude: location.degreesLatitude ?? location.latitude ?? null,
    longitude: location.degreesLongitude ?? location.longitude ?? null,
    isLive,
  };
}

function formatContactText(contact) {
  const name = contact.displayName || contact.vcard?.match(/FN:(.+)/)?.[1] || 'unknown';
  const phone = contact.vcard?.match(/TEL[^:]*:(.+)/)?.[1] || '';
  return `[Contact: ${[name, phone].filter(Boolean).join(' ')}]`;
}

function formatContactsText(contacts) {
  const names = contacts.map(c => c.displayName).filter(Boolean);
  return `[Contacts: ${names.join(', ') || contacts.length}]`;
}

function formatReactionText(reaction) {
  const emoji = reaction.text || '';
  const target = reaction.key?.id || '';
  return `[Reaction: ${emoji}${target ? ` to ${target}` : ''}]`;
}

function pollOptions(poll) {
  return (poll.options || [])
    .map(option => option.optionName || option.name)
    .filter(Boolean);
}

function formatPollText(poll) {
  const question = poll.name || poll.title || 'poll';
  const options = pollOptions(poll);
  return `[Poll: ${question}${options.length ? ` Options: ${options.join(', ')}` : ''}]`;
}

function formatPollUpdateText(update) {
  const target = update.pollCreationMessageKey?.id || update.key?.id || '';
  return `[Poll update${target ? `: ${target}` : ''}]`;
}

/**
 * Append a visible note for media that failed to download, so the agent knows
 * something was sent rather than silently losing the attachment. Returns
 * `content` unchanged when nothing failed. (Port of nanoclaw#2895.)
 */
export function appendMediaFailureNote(content, failures) {
  if (!failures || failures.length === 0) return content;
  const note = failures.map((t) => `[${t} could not be downloaded]`).join(' ');
  return content ? `${content}\n${note}` : note;
}

export function appendMediaLimitNote(content, rejections) {
  if (!rejections || rejections.length === 0) return content;
  const note = rejections.map(({ type, limit }) => {
    const megabytes = Math.floor(limit / MEBIBYTE);
    return `[The ${type} was rejected because it exceeds the ${megabytes} MB limit.]`;
  }).join(' ');
  return content ? `${content}\n${note}` : note;
}

export async function extractBridgeEvent({
  msg,
  chatId,
  senderId,
  senderNumber,
  botIds = [],
  isGroup = false,
  downloadMedia,
  writeMediaFile,
  cacheDirs = {},
  mediaLimits = INBOUND_MEDIA_LIMIT_BYTES,
  log = () => {},
}) {
  const messageContent = getMessageContent(msg);
  const contextInfo = getContextInfo(messageContent);
  const mentionedIds = Array.from(new Set((contextInfo?.mentionedJid || []).map(normalizeWhatsAppId).filter(Boolean)));
  const quotedMessageId = contextInfo?.stanzaId || null;
  const quotedParticipant = normalizeWhatsAppId(contextInfo?.participant || '') || null;
  const quotedRemoteJid = normalizeWhatsAppId(contextInfo?.remoteJid || '') || null;
  const hasQuotedMessage = !!contextInfo?.quotedMessage;
  const quotedText = textFromQuotedMessage(contextInfo?.quotedMessage);

  let body = '';
  let hasMedia = false;
  let mediaType = '';
  let mime = '';
  let fileName = '';
  let nativeType = '';
  const mediaUrls = [];
  const nativeMetadata = {};

  const mediaFailures = [];
  const mediaRejections = [];

  const saveMedia = async ({ mediaMessage, dir, prefix, fallbackExt, fileName: name, type }) => {
    if (!downloadMedia) return;
    const limit = Number(mediaLimits?.[type] || 0);
    const advertisedSize = mediaLengthBytes(mediaMessage?.fileLength);
    if (limit > 0 && advertisedSize !== null && advertisedSize > limit) {
      mediaRejections.push({ type, limit });
      log(`rejected inbound ${type}: advertised size exceeds ${Math.floor(limit / MEBIBYTE)} MB`);
      return;
    }
    try {
      const downloaded = await downloadMedia(msg);
      const streamed = isAsyncIterable(downloaded);
      const downloadedSize = streamed
        ? null
        : Number(downloaded?.byteLength ?? downloaded?.length ?? 0);
      if (limit > 0 && downloadedSize !== null && downloadedSize > limit) {
        mediaRejections.push({ type, limit });
        log(`rejected inbound ${type}: downloaded size exceeds ${Math.floor(limit / MEBIBYTE)} MB`);
        return;
      }
      const ext = mediaExtForMime(mediaMessage?.mimetype, fallbackExt);
      const writer = writeMediaFile || defaultWriteMediaFile;
      const saved = await writer({
        buffer: streamed ? undefined : downloaded,
        stream: streamed ? downloaded : undefined,
        dir,
        prefix,
        ext,
        fileName: name,
        limit,
      });
      if (saved) mediaUrls.push(saved);
    } catch (err) {
      if (err?.code === 'PILOTAGE_INBOUND_MEDIA_LIMIT') {
        mediaRejections.push({ type, limit });
        log(`rejected inbound ${type}: streamed size exceeds ${Math.floor(limit / MEBIBYTE)} MB`);
        return;
      }
      // A failed CDN fetch (expired media URL, transient network error) must
      // never reject out of extractBridgeEvent — that would drop this message
      // AND every remaining message in the same upsert batch. Record the
      // failure so the agent is told media was sent instead of losing it
      // silently. (Port of nanoclaw#2895's never-silently-drop guarantee; the
      // reuploadRequest recovery half is already wired in bridge.js.)
      mediaFailures.push(type || 'media');
      try {
        log(`failed to download inbound ${type || 'media'}: ${err?.message || err}`);
      } catch {}
    }
  };

  if (messageContent.conversation) {
    body = messageContent.conversation;
    nativeType = 'conversation';
  } else if (messageContent.extendedTextMessage?.text) {
    body = messageContent.extendedTextMessage.text;
    nativeType = 'extendedTextMessage';
  } else if (messageContent.imageMessage) {
    const item = messageContent.imageMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'image';
    nativeType = 'imageMessage';
    mime = item.mimetype || 'image/jpeg';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.image, prefix: 'img', fallbackExt: '.jpg', type: 'image' });
  } else if (messageContent.videoMessage) {
    const item = messageContent.videoMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = item.gifPlayback ? 'gif' : 'video';
    nativeType = 'videoMessage';
    mime = item.mimetype || 'video/mp4';
    nativeMetadata.video = { gifPlayback: !!item.gifPlayback };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'vid', fallbackExt: '.mp4', type: mediaType });
  } else if (messageContent.audioMessage || messageContent.pttMessage) {
    const item = messageContent.pttMessage || messageContent.audioMessage;
    hasMedia = true;
    mediaType = item.ptt || messageContent.pttMessage ? 'ptt' : 'audio';
    nativeType = messageContent.pttMessage ? 'pttMessage' : 'audioMessage';
    mime = item.mimetype || 'audio/ogg';
    nativeMetadata.audio = { ptt: mediaType === 'ptt' };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.audio, prefix: 'aud', fallbackExt: '.ogg', type: 'audio' });
  } else if (messageContent.documentMessage) {
    const item = messageContent.documentMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'document';
    nativeType = 'documentMessage';
    mime = item.mimetype || 'application/octet-stream';
    fileName = item.fileName || 'document';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'doc', fallbackExt: '.bin', fileName, type: 'document' });
  } else if (messageContent.stickerMessage) {
    hasMedia = true;
    mediaType = 'sticker';
    nativeType = 'stickerMessage';
    mime = messageContent.stickerMessage.mimetype || 'image/webp';
    body = '[Sticker]';
    nativeMetadata.sticker = {
      animated: !!messageContent.stickerMessage.isAnimated,
      mimetype: mime,
    };
    await saveMedia({ mediaMessage: messageContent.stickerMessage, dir: cacheDirs.image, prefix: 'sticker', fallbackExt: '.webp', type: 'sticker' });
  } else if (messageContent.locationMessage || messageContent.liveLocationMessage) {
    const isLive = !!messageContent.liveLocationMessage;
    const item = messageContent.liveLocationMessage || messageContent.locationMessage;
    mediaType = isLive ? 'live_location' : 'location';
    nativeType = isLive ? 'liveLocationMessage' : 'locationMessage';
    body = formatLocationText(item, isLive);
    nativeMetadata.location = locationMetadata(item, isLive);
  } else if (messageContent.contactMessage) {
    mediaType = 'contact';
    nativeType = 'contactMessage';
    body = formatContactText(messageContent.contactMessage);
    nativeMetadata.contact = {
      displayName: messageContent.contactMessage.displayName || '',
      vcard: messageContent.contactMessage.vcard || '',
    };
  } else if (messageContent.contactsArrayMessage) {
    const contacts = messageContent.contactsArrayMessage.contacts || [];
    mediaType = 'contacts';
    nativeType = 'contactsArrayMessage';
    body = formatContactsText(contacts);
    nativeMetadata.contacts = contacts.map(contact => ({
      displayName: contact.displayName || '',
      vcard: contact.vcard || '',
    }));
  } else if (messageContent.reactionMessage) {
    mediaType = 'reaction';
    nativeType = 'reactionMessage';
    body = formatReactionText(messageContent.reactionMessage);
    nativeMetadata.reaction = {
      text: messageContent.reactionMessage.text || '',
      messageId: messageContent.reactionMessage.key?.id || '',
      remoteJid: normalizeWhatsAppId(messageContent.reactionMessage.key?.remoteJid || ''),
      participant: normalizeWhatsAppId(messageContent.reactionMessage.key?.participant || ''),
    };
  } else if (messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3) {
    const item = messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3;
    mediaType = 'poll';
    nativeType = messageContent.pollCreationMessage ? 'pollCreationMessage' : messageContent.pollCreationMessageV2 ? 'pollCreationMessageV2' : 'pollCreationMessageV3';
    body = formatPollText(item);
    nativeMetadata.poll = {
      question: item.name || item.title || '',
      options: pollOptions(item),
      selectableCount: item.selectableOptionsCount || item.selectableCount || 1,
    };
  } else if (messageContent.pollUpdateMessage) {
    mediaType = 'poll_update';
    nativeType = 'pollUpdateMessage';
    body = formatPollUpdateText(messageContent.pollUpdateMessage);
    nativeMetadata.pollUpdate = messageContent.pollUpdateMessage;
  }

  // Surface failed downloads to the agent instead of silently losing the
  // attachment. Applied before the generic "[<type> received]" fallback so an
  // uncaptioned message whose download failed reads "[image could not be
  // downloaded]" rather than claiming the media arrived.
  body = appendMediaFailureNote(body, mediaFailures);
  body = appendMediaLimitNote(body, mediaRejections);

  if (hasMedia && !body) {
    body = `[${mediaType} received]`;
  }

  return {
    messageId: msg.key.id,
    chatId,
    senderId,
    senderName: msg.pushName || senderNumber,
    chatName: isGroup ? (chatId.split('@')[0]) : (msg.pushName || senderNumber),
    isGroup,
    body,
    hasMedia,
    mediaType,
    mime,
    fileName,
    nativeType,
    nativeMetadata,
    mediaUrls,
    mentionedIds,
    quotedMessageId,
    quotedParticipant,
    quotedRemoteJid,
    quotedText,
    hasQuotedMessage,
    botIds,
    readReceiptKey: {
      remoteJid: msg.key.remoteJid || chatId,
      id: msg.key.id,
      // Baileys 7.0.0-rc13 silently ignores DM receipts that carry the
      // group-only participant field. Groups need the original participant.
      ...(isGroup ? { participant: msg.key.participant || senderId } : {}),
      fromMe: Boolean(msg.key.fromMe),
    },
    timestamp: msg.messageTimestamp,
  };
}

export function inferMediaType(ext) {
  if (['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'avi', 'mkv', '3gp'].includes(ext)) return 'video';
  if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';
  return 'document';
}

export function inboundReadReceiptKeys({ key, enabled }) {
  if (!enabled || !key || key.fromMe || !key.id || !key.remoteJid) return [];
  // Repair durable events created by older bridge code too: a DM key must not
  // look group-shaped, while a real group keeps its original participant.
  if (String(key.remoteJid).endsWith('@g.us')) return [key];
  const { participant: _ignored, ...directKey } = key;
  return [directKey];
}

export function mediaPayloadForFile({ buffer, filePath, mediaType, caption, fileName }) {
  const ext = filePath.toLowerCase().split('.').pop();
  const type = mediaType || inferMediaType(ext);
  if (type === 'image' && ext === 'gif') {
    // Pure helper fallback: do not lie and label raw GIF bytes as mp4.
    // The live bridge tries ffmpeg conversion to WhatsApp gifPlayback video
    // before it falls back to this regular image payload.
    return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/gif' };
  }
  switch (type) {
    case 'image':
      return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/jpeg' };
    case 'video':
      return { video: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'video/mp4' };
    case 'document':
      return {
        document: buffer,
        fileName: fileName || path.basename(filePath),
        caption: caption || undefined,
        mimetype: MIME_MAP[ext] || 'application/octet-stream',
      };
    default:
      return null;
  }
}

export function buildPollPayload({ question, options, selectableCount = 1 }) {
  const cleanQuestion = String(question || '').trim();
  const cleanOptions = (options || []).map(option => String(option || '').trim()).filter(Boolean);
  if (!cleanQuestion) throw new Error('question is required');
  if (cleanOptions.length < 2) throw new Error('at least two poll options are required');
  if (cleanOptions.length > 12) throw new Error('at most 12 poll options are supported');
  const count = Math.max(1, Math.min(Number(selectableCount) || 1, cleanOptions.length));
  return {
    poll: {
      name: cleanQuestion,
      values: cleanOptions,
      selectableCount: count,
      messageSecret: randomBytes(32),
    },
  };
}

export function pollCreationMessageFromPayload(payload) {
  const poll = payload?.poll;
  if (!poll) return null;
  const values = Array.isArray(poll.values) ? poll.values : [];
  const options = values.map(value => String(value || '').trim()).filter(Boolean);
  if (!poll.name || options.length < 2) return null;
  const selectableOptionsCount = Math.max(1, Math.min(Number(poll.selectableCount) || 1, options.length));
  const message = {};
  if (poll.messageSecret) {
    message.messageContextInfo = { messageSecret: poll.messageSecret };
  }
  message[selectableOptionsCount === 1 ? 'pollCreationMessageV3' : 'pollCreationMessage'] = {
    name: String(poll.name),
    options: options.map(optionName => ({ optionName })),
    selectableOptionsCount,
  };
  return message;
}

/**
 * Reconnect scheduling guard. startSocket() awaits network I/O before it
 * creates a socket or registers event handlers, so a bare
 * `setTimeout(startSocket, ...)` has two unrecoverable failure modes: a
 * rejection is unhandled (crashes the process on modern Node), and a hang
 * leaves the bridge permanently disconnected with nothing left to retry.
 * Every (re)connect must go through the scheduler this returns.
 */
export function createReconnectScheduler(startFn, {
  retryDelayMs = 5000,
  log = console.log,
  setTimeoutFn = setTimeout,
  shouldReconnect = () => true,
} = {}) {
  function scheduleReconnect(delayMs) {
    if (!shouldReconnect()) return;
    setTimeoutFn(() => {
      if (!shouldReconnect()) return;
      Promise.resolve()
        .then(startFn)
        .catch((err) => {
          if (!shouldReconnect()) return;
          log(`⚠️  Reconnect failed (${err?.message || err}). Retrying in ${Math.round(retryDelayMs / 1000)}s...`);
          scheduleReconnect(retryDelayMs);
        });
    }, delayMs);
  }
  return scheduleReconnect;
}

/**
 * Version resolution guard. fetchLatestBaileysVersion() is a plain fetch to
 * raw.githubusercontent.com with no AbortSignal; a stalled connection can
 * pend forever and wedge the reconnect path (the scheduler above cannot
 * retry past an await that never settles). Bound the fetch and fall back to
 * the last known-good version, or the Baileys default before first success.
 */
export function createVersionResolver(fetchVersionFn, {
  timeoutMs = 15000,
  log = console.log,
} = {}) {
  let cachedVersion = null;
  return async function resolveVersion() {
    let timer = null;
    try {
      const { version } = await Promise.race([
        fetchVersionFn(),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('version fetch timed out')), timeoutMs);
        }),
      ]);
      cachedVersion = version;
    } catch (err) {
      log(`⚠️  Baileys version fetch failed (${err?.message || err}); using ${cachedVersion ? 'cached version' : 'library default'}.`);
    } finally {
      if (timer) clearTimeout(timer);
    }
    return cachedVersion;
  };
}
