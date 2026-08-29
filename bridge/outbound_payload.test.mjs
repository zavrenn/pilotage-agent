import assert from 'node:assert/strict';
import test from 'node:test';

import { buildEditSendPayload } from './bridge_helpers.js';


test('progress edits target the exact process-owned WhatsApp message', () => {
  assert.deepEqual(
    buildEditSendPayload(
      '212600123456@s.whatsapp.net',
      'progress-1',
      'Je continue. (2 min)',
    ),
    {
      text: 'Je continue. (2 min)',
      edit: {
        id: 'progress-1',
        fromMe: true,
        remoteJid: '212600123456@s.whatsapp.net',
      },
    },
  );
});


test('progress edits reject incomplete identities and empty text', () => {
  assert.throws(() => buildEditSendPayload('', 'progress-1', 'working'));
  assert.throws(() => buildEditSendPayload('chat', '', 'working'));
  assert.throws(() => buildEditSendPayload('chat', 'progress-1', '   '));
});
