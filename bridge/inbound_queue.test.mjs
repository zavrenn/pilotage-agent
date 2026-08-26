import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { createDurableInboundQueue } from './bridge_helpers.js';


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
