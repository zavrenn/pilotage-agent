import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, open, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  createAtomicJsonFileStore,
  useAtomicMultiFileAuthState,
} from './bridge_helpers.js';


async function temporaryDirectory(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'pilotage-auth-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test('ENOSPC during a temp write preserves the last good credentials', async (t) => {
  const directory = await temporaryDirectory(t);
  const good = createAtomicJsonFileStore(directory);
  await good.initialize();
  await good.writeData({ registered: true, revision: 1 }, 'creds.json');

  const failing = createAtomicJsonFileStore(directory, {
    fileOps: {
      open: async (target, flags, mode) => {
        const handle = await open(target, flags, mode);
        if (flags !== 'wx') return handle;
        return {
          close: () => handle.close(),
          sync: () => handle.sync(),
          writeFile: async (value) => {
            await handle.writeFile(String(value).slice(0, 5));
            const error = new Error('disk full');
            error.code = 'ENOSPC';
            throw error;
          },
        };
      },
    },
  });

  await assert.rejects(
    failing.writeData({ registered: true, revision: 2 }, 'creds.json'),
    (error) => error.code === 'ENOSPC',
  );
  assert.deepEqual(JSON.parse(await readFile(path.join(directory, 'creds.json'), 'utf8')), {
    registered: true,
    revision: 1,
  });
});

test('corrupt credentials fail loudly instead of silently creating a new identity', async (t) => {
  const directory = await temporaryDirectory(t);
  await writeFile(path.join(directory, 'creds.json'), '{truncated', 'utf8');
  const BufferJSON = { replacer: (_key, value) => value, reviver: (_key, value) => value };

  await assert.rejects(
    useAtomicMultiFileAuthState(directory, {
      BufferJSON,
      initAuthCreds: () => ({ registered: false }),
      proto: {},
    }),
    /Corrupt WhatsApp authentication file: creds\.json/,
  );
});

test('concurrent writes to one auth path are committed in invocation order', async (t) => {
  const directory = await temporaryDirectory(t);
  const store = createAtomicJsonFileStore(directory);
  await store.initialize();

  await Promise.all([
    store.writeData({ revision: 1 }, 'creds.json'),
    store.writeData({ revision: 2 }, 'creds.json'),
  ]);

  assert.deepEqual(JSON.parse(await readFile(path.join(directory, 'creds.json'), 'utf8')), {
    revision: 2,
  });
});
