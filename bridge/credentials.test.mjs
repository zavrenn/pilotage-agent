import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createCredentialSaveCoordinator,
  createReconnectScheduler,
  drainTasksForShutdown,
  flushCredentialSavesForShutdown,
} from './bridge_helpers.js';


const nextTurn = () => new Promise((resolve) => setImmediate(resolve));


test('credential saves enter the Baileys file queue before a reconnect read', async () => {
  const order = [];
  let fileTail = Promise.resolve();
  let releaseFirst;

  function fileOperation(name, wait = false) {
    const operation = fileTail.then(async () => {
      order.push(name);
      if (wait) {
        await new Promise((resolve) => { releaseFirst = resolve; });
      }
    });
    fileTail = operation;
    return operation;
  }

  const saves = createCredentialSaveCoordinator();
  saves.queue(() => fileOperation('first save', true));
  saves.queue(() => fileOperation('pair-success save'));
  const reconnectRead = fileOperation('reconnect read');
  await nextTurn();

  assert.deepEqual(order, ['first save']);
  releaseFirst();
  await reconnectRead;
  assert.deepEqual(order, [
    'first save',
    'pair-success save',
    'reconnect read',
  ]);
});


test('credential flush restores current state after a late stale write', async () => {
  const saves = createCredentialSaveCoordinator();
  let persisted = 'initial';
  let currentCalls = 0;
  let releaseCurrent;
  let releaseStale;
  let currentStarted;
  const firstCurrentStarted = new Promise((resolve) => { currentStarted = resolve; });

  function saveCurrent() {
    currentCalls += 1;
    if (currentCalls > 1) {
      persisted = 'current';
      return Promise.resolve();
    }
    currentStarted();
    return new Promise((resolve) => {
      releaseCurrent = () => {
        persisted = 'current';
        resolve();
      };
    });
  }

  const flushing = saves.flush(saveCurrent);
  await firstCurrentStarted;
  saves.queue(() => new Promise((resolve) => {
    releaseStale = () => {
      persisted = 'stale';
      resolve();
    };
  }));

  releaseCurrent();
  await nextTurn();
  releaseStale();
  await flushing;

  assert.equal(persisted, 'current');
  assert.equal(currentCalls, 2);
});


test('credential flush reports an earlier write failure', async () => {
  const messages = [];
  const saves = createCredentialSaveCoordinator({
    log: (message) => messages.push(message),
  });

  saves.queue(async () => {
    throw new Error('disk unavailable');
  });

  await assert.rejects(
    saves.flush(async () => {}),
    /disk unavailable/,
  );
  assert.deepEqual(messages, ['credential save failed: disk unavailable']);
});


test('normal shutdown waits for queued credential saves and a final snapshot', async () => {
  const saves = createCredentialSaveCoordinator();
  const order = [];
  let releaseQueued;
  saves.queue(() => new Promise((resolve) => {
    releaseQueued = () => {
      order.push('queued');
      resolve();
    };
  }));

  const flushing = flushCredentialSavesForShutdown(
    saves,
    async () => { order.push('final'); },
    { timeoutMs: 100 },
  );
  await nextTurn();
  assert.deepEqual(order, []);
  releaseQueued();
  await flushing;
  assert.deepEqual(order, ['queued', 'final']);
});


test('normal shutdown closes the socket before its final credential snapshot', async () => {
  const saves = createCredentialSaveCoordinator();
  const order = [];
  let releaseClose;

  const flushing = flushCredentialSavesForShutdown(
    saves,
    async () => { order.push('final snapshot'); },
    {
      timeoutMs: 100,
      closeSocket: async () => {
        order.push('close started');
        await new Promise((resolve) => { releaseClose = resolve; });
        order.push('close finished');
      },
    },
  );
  await nextTurn();
  assert.deepEqual(order, ['close started']);
  releaseClose();
  await flushing;
  assert.deepEqual(order, [
    'close started',
    'close finished',
    'final snapshot',
  ]);
});


test('shutdown deadline includes socket closure before credential flush', async () => {
  const saves = createCredentialSaveCoordinator();
  let snapshots = 0;

  await assert.rejects(
    flushCredentialSavesForShutdown(
      saves,
      async () => { snapshots += 1; },
      {
        timeoutMs: 5,
        closeSocket: () => new Promise(() => {}),
      },
    ),
    /timed out/,
  );
  assert.equal(snapshots, 0);
});


test('shutdown fence suppresses pending and new reconnect attempts', async () => {
  const timers = [];
  let reconnectAllowed = true;
  let starts = 0;
  const scheduleReconnect = createReconnectScheduler(
    async () => { starts += 1; },
    {
      shouldReconnect: () => reconnectAllowed,
      setTimeoutFn: (callback) => { timers.push(callback); },
    },
  );

  scheduleReconnect(3_000);
  assert.equal(timers.length, 1);
  reconnectAllowed = false;
  timers[0]();
  scheduleReconnect(3_000);
  await nextTurn();

  assert.equal(starts, 0);
  assert.equal(timers.length, 1);
});


test('normal shutdown credential flush has a hard deadline', async () => {
  const saves = createCredentialSaveCoordinator();
  await assert.rejects(
    flushCredentialSavesForShutdown(
      saves,
      () => new Promise(() => {}),
      { timeoutMs: 5 },
    ),
    /timed out/,
  );
});


test('shutdown waits for every accepted inbound task', async () => {
  let release;
  const task = new Promise((resolve) => { release = resolve; });
  const draining = drainTasksForShutdown(new Set([task]), { timeoutMs: 100 });

  await nextTurn();
  let finished = false;
  void draining.then(() => { finished = true; });
  await nextTurn();
  assert.equal(finished, false);

  release();
  assert.equal(await draining, true);
});


test('a rejected inbound task is still fully drained', async () => {
  const rejected = Promise.reject(new Error('spool failed'));
  assert.equal(
    await drainTasksForShutdown(new Set([rejected]), { timeoutMs: 100 }),
    true,
  );
});


test('shutdown inbound drain has a hard deadline', async () => {
  const never = new Promise(() => {});

  assert.equal(
    await drainTasksForShutdown(new Set([never]), { timeoutMs: 5 }),
    false,
  );
});
