import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { Readable } from 'node:stream';

import {
  appendMediaLimitNote,
  extractBridgeEvent,
  hasDownloadableMedia,
  INBOUND_MEDIA_LIMIT_BYTES,
  inboundReadReceiptKeys,
  inferMediaType,
  mediaLengthBytes,
  mediaPayloadForFile,
} from './bridge_helpers.js';


test('Hermes media type mapping keeps skill outputs native', () => {
  assert.equal(inferMediaType('png'), 'image');
  assert.equal(inferMediaType('mp4'), 'video');
  assert.equal(inferMediaType('xlsx'), 'document');
  assert.equal(inferMediaType('pdf'), 'document');
});


test('download fences recognize wrapped media but not ordinary text', () => {
  assert.equal(hasDownloadableMedia({ message: { conversation: 'hello' } }), false);
  assert.equal(hasDownloadableMedia({ message: { imageMessage: {} } }), true);
  assert.equal(hasDownloadableMedia({
    message: {
      ephemeralMessage: {
        message: { documentMessage: {} },
      },
    },
  }), true);
});


test('inbound media ceilings match the production decision', () => {
  assert.equal(INBOUND_MEDIA_LIMIT_BYTES.image, 20 * 1024 * 1024);
  assert.equal(INBOUND_MEDIA_LIMIT_BYTES.audio, 25 * 1024 * 1024);
  assert.equal(INBOUND_MEDIA_LIMIT_BYTES.video, 100 * 1024 * 1024);
  assert.equal(INBOUND_MEDIA_LIMIT_BYTES.document, 50 * 1024 * 1024);
  assert.equal(
    mediaLengthBytes({ toString: () => String(INBOUND_MEDIA_LIMIT_BYTES.image + 1) }),
    INBOUND_MEDIA_LIMIT_BYTES.image + 1,
  );
  assert.equal(
    appendMediaLimitNote('', [{ type: 'video', limit: INBOUND_MEDIA_LIMIT_BYTES.video }]),
    '[The video was rejected because it exceeds the 100 MB limit.]',
  );
});


test('accepted inbound media is written through the asynchronous cache writer', async () => {
  const cacheDir = mkdtempSync(path.join(os.tmpdir(), 'pilotage-media-'));
  try {
    const event = await extractBridgeEvent({
      msg: {
        key: { id: 'small-image' },
        message: {
          imageMessage: {
            fileLength: '5',
            mimetype: 'image/jpeg',
          },
        },
      },
      chatId: '212600000000@s.whatsapp.net',
      senderId: '212600000000@s.whatsapp.net',
      senderNumber: '212600000000',
      cacheDirs: { image: cacheDir },
      downloadMedia: async () => Readable.from([Buffer.from('im'), Buffer.from('age')]),
    });

    assert.equal(event.mediaUrls.length, 1);
    assert.equal(readFileSync(event.mediaUrls[0], 'utf8'), 'image');
  } finally {
    rmSync(cacheDir, { recursive: true, force: true });
  }
});


test('streamed oversize media stops and removes its partial cache file', async () => {
  const cacheDir = mkdtempSync(path.join(os.tmpdir(), 'pilotage-media-limit-'));
  try {
    const event = await extractBridgeEvent({
      msg: {
        key: { id: 'streamed-oversize-image' },
        message: { imageMessage: { mimetype: 'image/jpeg' } },
      },
      chatId: '212600000000@s.whatsapp.net',
      senderId: '212600000000@s.whatsapp.net',
      senderNumber: '212600000000',
      cacheDirs: { image: cacheDir },
      mediaLimits: { image: 3 },
      downloadMedia: async () => Readable.from([
        Buffer.from('ab'),
        Buffer.from('cd'),
        Buffer.from('must-not-be-read'),
      ]),
    });

    assert.deepEqual(event.mediaUrls, []);
    assert.match(event.body, /rejected because it exceeds/);
    assert.deepEqual(readdirSync(cacheDir), []);
  } finally {
    rmSync(cacheDir, { recursive: true, force: true });
  }
});


test('advertised oversize media is rejected before download', async () => {
  let downloads = 0;
  let writes = 0;
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'too-large-image' },
      message: {
        imageMessage: {
          fileLength: String(INBOUND_MEDIA_LIMIT_BYTES.image + 1),
          mimetype: 'image/jpeg',
        },
      },
    },
    chatId: '212600000000@s.whatsapp.net',
    senderId: '212600000000@s.whatsapp.net',
    senderNumber: '212600000000',
    downloadMedia: async () => {
      downloads += 1;
      return Buffer.from('must not download');
    },
    writeMediaFile: async () => {
      writes += 1;
      return '/media/forbidden.jpg';
    },
  });

  assert.equal(downloads, 0);
  assert.equal(writes, 0);
  assert.deepEqual(event.mediaUrls, []);
  assert.match(event.body, /exceeds the 20 MB limit/);
});


test('downloaded oversize media is rejected when metadata is absent', async () => {
  let writes = 0;
  const oneMegabyte = 1024 * 1024;
  const event = await extractBridgeEvent({
    msg: {
      key: { id: 'unknown-size-audio' },
      message: { audioMessage: { mimetype: 'audio/ogg' } },
    },
    chatId: '212600000000@s.whatsapp.net',
    senderId: '212600000000@s.whatsapp.net',
    senderNumber: '212600000000',
    mediaLimits: { audio: oneMegabyte },
    downloadMedia: async () => Buffer.alloc(oneMegabyte + 1),
    writeMediaFile: async () => {
      writes += 1;
      return '/media/forbidden.ogg';
    },
  });

  assert.equal(writes, 0);
  assert.deepEqual(event.mediaUrls, []);
  assert.match(event.body, /exceeds the 1 MB limit/);
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
  assert.deepEqual(event.readReceiptKey, {
    remoteJid: '120363001234567890@g.us',
    id: 'm1',
    participant: '212600000000@s.whatsapp.net',
    fromMe: false,
  });
  assert.deepEqual(
    event.botIds,
    ['15551230000@10@s.whatsapp.net', '67427329167522@lid'],
  );
});


test('DM read receipts omit participant while groups preserve it', async () => {
  const event = await extractBridgeEvent({
    msg: {
      key: {
        id: 'dm-1',
        remoteJid: '212600000000@s.whatsapp.net',
      },
      pushName: 'User',
      messageTimestamp: 1,
      message: { conversation: 'hello' },
    },
    chatId: '212600000000@s.whatsapp.net',
    senderId: '212600000000@s.whatsapp.net',
    senderNumber: '212600000000',
    botIds: [],
    isGroup: false,
  });

  assert.deepEqual(event.readReceiptKey, {
    remoteJid: '212600000000@s.whatsapp.net',
    id: 'dm-1',
    fromMe: false,
  });

  const legacyDm = {
    remoteJid: '212600000000@lid',
    id: 'dm-old',
    participant: '212600000000@lid',
    fromMe: false,
  };
  assert.deepEqual(inboundReadReceiptKeys({ key: legacyDm, enabled: true }), [{
    remoteJid: '212600000000@lid',
    id: 'dm-old',
    fromMe: false,
  }]);
  assert.equal(legacyDm.participant, '212600000000@lid');

  const group = {
    remoteJid: '120363001234567890@g.us',
    id: 'group-1',
    participant: '212600000000@s.whatsapp.net',
    fromMe: false,
  };
  assert.deepEqual(inboundReadReceiptKeys({ key: group, enabled: true }), [group]);
});
