import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  createDurableInboundQueue,
  resolveInboundClaimIdentities,
} from './bridge_helpers.js';


async function queueDirectory(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'pilotage-inbound-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

function event(id, body = 'hello') {
  return {
    messageId: id,
    chatId: '212600123456@s.whatsapp.net',
    senderId: '212600123456@s.whatsapp.net',
    body,
  };
}

function aliasedEvent(id, {
  chatId = '212600123456@s.whatsapp.net',
  senderId = '212600123456@s.whatsapp.net',
  senderNumber = '212600123456',
  identities = ['212600123456', '778899001122334'],
  claimIdentities = ['pn:212600123456', 'lid:778899001122334'],
  isGroup = false,
} = {}) {
  return {
    messageId: id,
    chatId,
    senderId,
    senderNumber,
    identities,
    _pilotageClaimIdentities: claimIdentities,
    isGroup,
    body: 'hello',
  };
}

test('claim aliases require typed sender or mapping evidence', () => {
  assert.deepEqual(
    resolveInboundClaimIdentities({
      senderId: '212600123456@s.whatsapp.net',
      identities: ['212600123456', '778899001122334'],
    }),
    ['lid:778899001122334', 'pn:212600123456'],
  );
  assert.deepEqual(
    resolveInboundClaimIdentities({
      senderId: '555@lid',
      identities: ['555'],
    }),
    ['lid:555'],
  );
});

test('a claim survives bridge restart and is replayed with the same identity', async (t) => {
  const directory = await queueDirectory(t);
  const first = createDurableInboundQueue(directory);
  await first.initialize();
  await first.enqueue(event('message-1'));
  const claimed = await first.claim();

  assert.equal(claimed.length, 1);
  const claimId = claimed[0]._pilotageClaimId;

  const restarted = createDurableInboundQueue(directory);
  await restarted.initialize();
  const replayed = await restarted.claim();

  assert.equal(replayed.length, 1);
  assert.equal(replayed[0]._pilotageClaimId, claimId);
  assert.equal(replayed[0].messageId, 'message-1');
});

test('ack creates a durable duplicate tombstone', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();
  await queue.enqueue(event('message-2'));
  const [claimed] = await queue.claim();
  assert.equal(await queue.ack([claimed._pilotageClaimId]), 1);

  const duplicate = await queue.enqueue(event('message-2'));
  assert.equal(duplicate.status, 'duplicate');
  assert.deepEqual(await queue.claim(), []);

  const restarted = createDurableInboundQueue(directory);
  await restarted.initialize();
  assert.equal((await restarted.enqueue(event('message-2'))).status, 'duplicate');
  assert.deepEqual(await restarted.claim(), []);
});

test('PN and LID forms of one direct message share a durable claim', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();
  const pn = aliasedEvent('alias-message');
  const lid = aliasedEvent('alias-message', {
    chatId: '778899001122334@lid',
    senderId: '778899001122334@lid',
    senderNumber: '212600123456',
    identities: ['778899001122334', '212600123456'],
  });

  const first = await queue.enqueue(pn);
  const duplicate = await queue.enqueue(lid);

  assert.equal(first.status, 'accepted');
  assert.equal(duplicate.status, 'duplicate');
  assert.equal(duplicate.claimId, first.claimId);
  const [claimed] = await queue.claim();
  assert.equal(claimed._pilotageClaimId, first.claimId);
  assert.equal((await queue.enqueue(lid)).status, 'duplicate');
  assert.equal(await queue.ack([claimed._pilotageClaimId]), 1);
  assert.equal((await queue.enqueue(lid)).status, 'duplicate');
});

test('equal PN and LID local parts stay distinct without mapping evidence', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();
  const pn = aliasedEvent('unmapped-same-local', {
    chatId: '555@s.whatsapp.net',
    senderId: '555@s.whatsapp.net',
    senderNumber: '555',
    identities: ['555'],
    claimIdentities: [],
  });
  const lid = aliasedEvent('unmapped-same-local', {
    chatId: '555@lid',
    senderId: '555@lid',
    senderNumber: '555',
    identities: ['555'],
    claimIdentities: [],
  });

  const first = await queue.enqueue(pn);
  const second = await queue.enqueue(lid);

  assert.equal(first.status, 'accepted');
  assert.equal(second.status, 'accepted');
  assert.notEqual(first.claimId, second.claimId);
});

test('group durable identity isolates both participant and group chat', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();

  const first = await queue.enqueue(aliasedEvent('shared-message-id', {
    chatId: '111111111111@g.us',
    senderId: '212600000001@s.whatsapp.net',
    senderNumber: '212600000001',
    identities: ['212600000001'],
    claimIdentities: ['pn:212600000001'],
    isGroup: true,
  }));
  const otherParticipant = await queue.enqueue(aliasedEvent('shared-message-id', {
    chatId: '111111111111@g.us',
    senderId: '212600000002@s.whatsapp.net',
    senderNumber: '212600000002',
    identities: ['212600000002'],
    claimIdentities: ['pn:212600000002'],
    isGroup: true,
  }));
  const otherGroup = await queue.enqueue(aliasedEvent('shared-message-id', {
    chatId: '222222222222@g.us',
    senderId: '212600000001@s.whatsapp.net',
    senderNumber: '212600000001',
    identities: ['212600000001'],
    claimIdentities: ['pn:212600000001'],
    isGroup: true,
  }));

  assert.deepEqual(
    new Set([first.claimId, otherParticipant.claimId, otherGroup.claimId]).size,
    3,
  );
  assert.equal((await queue.claim()).length, 3);
});

test('v2 enqueue honors a completion written by the legacy identity scheme', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();
  const inbound = aliasedEvent('legacy-message');
  const legacyClaim = createHash('sha256')
    .update(`${inbound.chatId}\0${inbound.senderId}\0${inbound.messageId}`)
    .digest('hex');
  await writeFile(
    path.join(directory, 'done', `${legacyClaim}.json`),
    JSON.stringify({ completedAt: Date.now() }),
    'utf8',
  );

  const duplicate = await queue.enqueue(inbound);

  assert.deepEqual(duplicate, { status: 'duplicate', claimId: legacyClaim });
  assert.deepEqual(await queue.claim(), []);
});

test('high-water overflow is visible without dropping accepted events', async (t) => {
  const directory = await queueDirectory(t);
  const logs = [];
  const queue = createDurableInboundQueue(directory, {
    highWaterMark: 2,
    log: (message) => logs.push(message),
  });
  await queue.initialize();

  await queue.enqueue(event('one'));
  await queue.enqueue(event('two'));
  await queue.enqueue(event('three'));

  const status = await queue.status();
  assert.equal(status.healthy, false);
  assert.equal(status.storageHealthy, true);
  assert.equal(status.overflowed, true);
  assert.equal(status.depth, 3);
  assert.equal((await queue.claim(10)).length, 3);
  assert.match(logs.join('\n'), /exceeded/);
});

test('release makes a failed claim available for retry', async (t) => {
  const directory = await queueDirectory(t);
  const queue = createDurableInboundQueue(directory);
  await queue.initialize();
  await queue.enqueue(event('message-3'));
  const [claimed] = await queue.claim();

  assert.equal(await queue.release([claimed._pilotageClaimId]), 1);
  const [retried] = await queue.claim();
  assert.equal(retried._pilotageClaimId, claimed._pilotageClaimId);
});

test('completed tombstones compact to a bounded low-water mark', async (t) => {
  const directory = await queueDirectory(t);
  let clock = 0;
  const queue = createDurableInboundQueue(directory, {
    doneMaxEntries: 3,
    now: () => ++clock,
  });
  await queue.initialize();

  for (const id of ['one', 'two', 'three', 'four']) {
    await queue.enqueue(event(id));
    const [claimed] = await queue.claim();
    assert.equal(await queue.ack([claimed._pilotageClaimId]), 1);
  }

  const status = await queue.status();
  assert.equal(status.completed, 2);
  assert.equal(status.completedMaxEntries, 3);
  assert.equal((await readdir(path.join(directory, 'done'))).length, 2);
  assert.equal((await queue.enqueue(event('one'))).status, 'accepted');
  assert.equal((await queue.enqueue(event('four'))).status, 'duplicate');
});
