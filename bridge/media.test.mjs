import test from 'node:test';
import assert from 'node:assert/strict';

import { inferMediaType, mediaPayloadForFile } from './bridge_helpers.js';


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
