import test from 'node:test';
import assert from 'node:assert/strict';

import {
  extractBridgeEvent,
  inferMediaType,
  mediaPayloadForFile,
} from './bridge_helpers.js';


test('Hermes media type mapping keeps skill outputs native', () => {
  assert.equal(inferMediaType('png'), 'image');
  assert.equal(inferMediaType('mp4'), 'video');
  assert.equal(inferMediaType('xlsx'), 'document');
  assert.equal(inferMediaType('pdf'), 'document');
});


test('chart bytes become a WhatsApp image payload', () => {
  const bytes = Buffer.from('png');
  const payload = mediaPayloadForFile({
    buffer: bytes,
    filePath: '/workspace/chart.png',
    mediaType: 'image',
  });

  assert.equal(payload.image, bytes);
  assert.equal(payload.mimetype, 'image/png');
  assert.equal(payload.caption, undefined);
});


test('Excel output becomes a named WhatsApp document payload', () => {
  const bytes = Buffer.from('xlsx');
  const payload = mediaPayloadForFile({
    buffer: bytes,
    filePath: '/workspace/report.xlsx',
    mediaType: 'document',
    fileName: 'report.xlsx',
  });

  assert.equal(payload.document, bytes);
  assert.equal(payload.fileName, 'report.xlsx');
  assert.equal(
    payload.mimetype,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  );
});


test('group event carries Hermes mention, quote, and bot identity metadata', async () => {
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'm1',
        remoteJid: '120363001234567890@g.us',
        participant: '212600000000@s.whatsapp.net',
      },
      pushName: 'User',
      messageTimestamp: 1,
      message: {
        extendedTextMessage: {
          text: '@15551230000 hello',
          contextInfo: {
            mentionedJid: ['15551230000@s.whatsapp.net'],
            stanzaId: 'quoted-1',
            participant: '67427329167522@lid',
            quotedMessage: { conversation: 'earlier answer' },
          },
        },
      },
    },
    chatId: '120363001234567890@g.us',
    senderId: '212600000000@s.whatsapp.net',
    senderNumber: '212600000000',
    botIds: [
      '15551230000@10@s.whatsapp.net',
      '67427329167522@lid',
    ],
    isGroup: true,
  });

  assert.deepEqual(event.mentionedIds, ['15551230000@s.whatsapp.net']);
  assert.equal(event.quotedParticipant, '67427329167522@lid');
  assert.equal(event.quotedText, 'earlier answer');
  assert.deepEqual(
    event.botIds,
    ['15551230000@10@s.whatsapp.net', '67427329167522@lid'],
  );
});
