/**
 * WhatsApp identifier resolution.
 *
 * Copied from the Hermes bridge (scripts/whatsapp-bridge/allowlist.js). The
 * bridge gates before media extraction; Python repeats the check as defence in
 * depth before dispatch.
 *
 * WhatsApp now addresses people by two different identifiers — the phone number
 * (`34600111222@s.whatsapp.net`) and a LID alias (`67427329167522@lid`) — and
 * which one arrives on a message is not under our control. Baileys writes the
 * mapping to the session directory as `lid-mapping-<id>.json` (phone → LID) and
 * `lid-mapping-<id>_reverse.json` (LID → phone), so an operator can put either
 * form in the allowlist and still be recognised.
 */

import path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';

export function normalizeWhatsAppIdentifier(value) {
  return String(value || '')
    .trim()
    .replace(/:.*@/, '@')
    .replace(/@.*/, '')
    .replace(/^\+/, '');
}

export function parseAllowedUsers(rawValue) {
  return new Set(
    String(rawValue || '')
      .split(',')
      .map((value) => normalizeWhatsAppIdentifier(value))
      .filter(Boolean)
  );
}

function readMappingFile(sessionDir, identifier, suffix = '') {
  const filePath = path.join(sessionDir, `lid-mapping-${identifier}${suffix}.json`);
  if (!existsSync(filePath)) {
    return null;
  }

  try {
    const parsed = JSON.parse(readFileSync(filePath, 'utf8'));
    const normalized = normalizeWhatsAppIdentifier(parsed);
    return normalized || null;
  } catch {
    return null;
  }
}

export function expandWhatsAppIdentifiers(identifier, sessionDir) {
  const normalized = normalizeWhatsAppIdentifier(identifier);
  if (!normalized) {
    return new Set();
  }

  // Walk both phone->LID and LID->phone mapping files so allowlists can use
  // either form transparently.
  const resolved = new Set();
  const queue = [normalized];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || resolved.has(current)) {
      continue;
    }

    resolved.add(current);

    for (const suffix of ['', '_reverse']) {
      const mapped = readMappingFile(sessionDir, current, suffix);
      if (mapped && !resolved.has(mapped)) {
        queue.push(mapped);
      }
    }
  }

  return resolved;
}

export function matchesAllowedUser(senderId, allowedUsers, sessionDir) {
  if (!allowedUsers || allowedUsers.size === 0) {
    return false;
  }

  const aliases = expandWhatsAppIdentifiers(senderId, sessionDir);
  for (const alias of aliases) {
    if (allowedUsers.has(alias)) {
      return true;
    }
  }

  return false;
}
