"""Inbound attachments: what the bridge downloaded, and what the model can read.

The bridge is a separate process that writes inbound media to disk and hands
back absolute paths. Those paths are checked against the cache roots before
anything is opened — a buggy or compromised bridge must not be able to name
``/etc/passwd`` and have it read into a model request. (Hermes'
``_is_allowed_bridge_path``.)

What the model can actually take differs by kind:

* **Images** go into the request as a base64 data URL.
* **Text-readable documents** are inlined into the message, capped, as Hermes
  does for every one of its channels.
* **Voice notes, audio and video** cannot be read at all. ChatGPT's OAuth
  session does not carry the transcription API, so they are announced to the
  model rather than silently dropped.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Inlined document text, matching Hermes' cap across its channels.
MAX_TEXT_INJECT_BYTES = 100 * 1024
# A base64 image is a third larger than the file. Anything past this is not
# worth the request it would build; WhatsApp itself compresses photos well
# below it.
MAX_IMAGE_BYTES = 15 * 1024 * 1024

IMAGE_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif"}
)

TEXT_DOCUMENT_SUFFIXES = frozenset(
    {
        ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
        ".log", ".py", ".js", ".ts", ".html", ".css",
    }
)


@dataclass(frozen=True)
class Attachment:
    """One file the bridge downloaded for an inbound message."""

    path: Path
    mime: str
    media_type: str
    file_name: str = ""

    @property
    def is_image(self) -> bool:
        return self.media_type in {"image", "sticker"} and self.mime in IMAGE_MIME_TYPES

    @property
    def is_text_document(self) -> bool:
        return self.media_type == "document" and self.path.suffix.lower() in TEXT_DOCUMENT_SUFFIXES

    def display_name(self) -> str:
        """The name the sender knows, not the cache file name.

        The bridge writes ``doc_<hex>_original-name.pdf``; strip the prefix so
        the model quotes the file the way the sender sees it.
        """
        if self.file_name:
            return self.file_name
        parts = self.path.name.split("_", 2)
        return parts[2] if len(parts) >= 3 else self.path.name


def collect(event: Dict[str, Any], roots: Sequence[Path]) -> List[Attachment]:
    """Read the media paths out of a bridge event, rejecting any path outside *roots*."""
    raw_paths = event.get("mediaUrls")
    if not isinstance(raw_paths, list):
        return []

    mime = str(event.get("mime") or "")
    media_type = str(event.get("mediaType") or "")
    file_name = str(event.get("fileName") or "")

    attachments: List[Attachment] = []
    for raw in raw_paths:
        path = _accept_path(str(raw), roots)
        if path is None:
            logger.warning("Rejected a media path the bridge reported outside its cache: %s", raw)
            continue
        attachments.append(
            Attachment(path=path, mime=mime, media_type=media_type, file_name=file_name)
        )
    return attachments


def _accept_path(raw: str, roots: Sequence[Path]) -> Optional[Path]:
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve()
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        return resolved
    return None


def inline_documents(text: str, attachments: Sequence[Attachment]) -> str:
    """Prepend the content of text-readable documents to the message. (Hermes)"""
    for attachment in attachments:
        if not attachment.is_text_document:
            continue
        try:
            size = attachment.path.stat().st_size
            if size > MAX_TEXT_INJECT_BYTES:
                logger.info(
                    "Not inlining %s: %s bytes is over the %s byte cap",
                    attachment.display_name(), size, MAX_TEXT_INJECT_BYTES,
                )
                continue
            content = attachment.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s: %s", attachment.path, exc)
            continue
        injection = f"[Content of {attachment.display_name()}]:\n{content}"
        text = f"{injection}\n\n{text}" if text else injection
    return text


def image_parts(attachments: Sequence[Attachment]) -> List[Dict[str, Any]]:
    """Build the ``input_image`` parts for a Responses request."""
    parts: List[Dict[str, Any]] = []
    for attachment in attachments:
        if not attachment.is_image:
            continue
        try:
            size = attachment.path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                logger.warning("Skipping %s: %s bytes is too large to send", attachment.path, size)
                continue
            encoded = base64.b64encode(attachment.path.read_bytes()).decode("ascii")
        except OSError as exc:
            logger.warning("Could not read %s: %s", attachment.path, exc)
            continue
        parts.append(
            {"type": "input_image", "image_url": f"data:{attachment.mime};base64,{encoded}"}
        )
    return parts


def describe_unreadable(attachments: Sequence[Attachment]) -> str:
    """Name what arrived but cannot be read, so the model does not answer blind."""
    notes: List[str] = []
    for attachment in attachments:
        if attachment.is_image or attachment.is_text_document:
            continue
        if attachment.media_type in {"ptt", "audio"}:
            notes.append(
                "[The sender left a voice message. This agent cannot listen to it yet — "
                "say so, and ask them to write it out.]"
            )
        elif attachment.media_type in {"video", "gif"}:
            notes.append("[The sender sent a video. This agent cannot watch it.]")
        else:
            name = attachment.display_name()
            notes.append(f"[The sender sent a file this agent cannot read: {name}.]")
    # One note per kind is enough; three photos of the same album should not
    # produce three identical lines.
    seen: List[str] = []
    for note in notes:
        if note not in seen:
            seen.append(note)
    return "\n".join(seen)
