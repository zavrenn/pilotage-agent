import test from 'node:test';
import assert from 'node:assert/strict';

import { createCredentialSaveCoordinator } from './bridge_helpers.js';


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
