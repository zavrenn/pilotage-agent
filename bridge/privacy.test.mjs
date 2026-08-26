import assert from 'node:assert/strict';
import test from 'node:test';

import { createIdentityRedactor, extractBridgeEvent } from './bridge_helpers.js';


test('bridge logs replace WhatsApp JIDs with stable keyed aliases', () => {
  const redactor = createIdentityRedactor(Buffer.alloc(32, 7));
  const jid = '212600123456@s.whatsapp.net';

  const first = redactor.redact(`connected as ${jid}`);
  const second = redactor.redact(`send failed for ${jid}`);

  assert.equal(first.includes(jid), false);
  assert.equal(second.includes(jid), false);
  assert.match(first, /\[wa:[a-f0-9]{12}\]/);
  assert.equal(first.match(/\[wa:[a-f0-9]{12}\]/)[0], second.match(/\[wa:[a-f0-9]{12}\]/)[0]);
});

test('inbound media errors use the redacted bridge log path', async () => {
  const redactor = createIdentityRedactor(Buffer.alloc(32, 9));
  const jid = '212600654321@s.whatsapp.net';
  const logs = [];

  await extractBridgeEvent({
    msg: {
      key: { id: 'm1', remoteJid: jid },
      message: { imageMessage: { mimetype: 'image/jpeg' } },
    },
    chatId: jid,
    senderId: jid,
    senderNumber: '212600654321',
    downloadMedia: async () => {
      throw new Error(`expired media session for ${jid}`);
    },
    cacheDirs: { image: '.' },
    log: (message) => logs.push(redactor.redact(message)),
  });

  assert.equal(logs.length, 1);
  assert.equal(logs[0].includes(jid), false);
  assert.match(logs[0], /\[wa:[a-f0-9]{12}\]/);
});
