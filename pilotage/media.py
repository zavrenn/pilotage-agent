"""Media crossing the bridge boundary.

The bridge is a separate process that writes inbound media to disk and hands
back absolute paths. Those paths are checked against the cache roots before
anything is opened — a buggy or compromised bridge must not be able to name
``/etc/passwd`` and have it read into a model request. (Hermes'
``_is_allowed_bridge_path``.)

What the model can actually take differs by kind:

* **Images** go into the request as a base64 data URL.
* **Text-readable documents** are inlined into the message, capped, as Hermes
  does for every one of its channels.
* **Voice notes** are handed to the separate OpenAI transcription path before
  the turn. Uploaded audio and video are announced rather than silently
  dropped.

Outbound files take the reverse path. The model names a generated file with
Hermes' ``MEDIA:/absolute/path`` directive; the directive is removed from the
visible answer and the resolved file is accepted only when it is a regular
file inside this profile's workspace. The WhatsApp channel then hands that
validated path to the private bridge as a native attachment.
"""

from __future__ import annotations

import base64
import logging
import re
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


# Hermes' single media-delivery extension set. Keeping one list prevents the
# extractor and transport router from disagreeing about whether a directive is
# deliverable. Pilotage currently routes audio as a document; native voice
# output is outside this slice.
MEDIA_DELIVERY_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp",
    ".mp3", ".m2a", ".wav", ".ogg", ".opus", ".m4a", ".flac",
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".epub",
    ".xlsx", ".xls", ".ods", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    ".kmz", ".kml", ".geojson", ".gpx",
    ".pptx", ".ppt", ".odp", ".key",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".apk", ".ipa",
    ".html", ".htm",
)

_IMAGE_DELIVERY_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})
_VIDEO_DELIVERY_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_MEDIA_EXT_ALTERNATION = "|".join(
    sorted((suffix.lstrip(".") for suffix in MEDIA_DELIVERY_SUFFIXES), key=len, reverse=True)
)

# Ported from Hermes' MEDIA_TAG_CLEANUP_RE. It accepts ordinary, quoted,
# backticked and markdown-emphasized directives while anchoring every match to
# an absolute path and a deliverable extension.
MEDIA_TAG_RE = re.compile(
    r'''[`"'*_]{0,3}MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+?`|"[^"\n]+?"|'[^'\n]+?'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])\S+?(?:[^\S\n]+\S+?)*?\.(?:''' + _MEDIA_EXT_ALTERNATION + r'''))'''
    r'''(?=[\s`"'*_,;:)\]}\[]|MEDIA:|\.(?:\s|$)|$)[`"'*_]{0,3}\.?''',
    re.IGNORECASE,
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

    @property
    def is_voice_message(self) -> bool:
        """Native WhatsApp push-to-talk notes enter automatic STT. (Hermes)"""
        return self.media_type == "ptt"

    def display_name(self) -> str:
        """The name the sender knows, not the cache file name.

        The bridge writes ``doc_<hex>_original-name.pdf``; strip the prefix so
        the model quotes the file the way the sender sees it.
        """
        if self.file_name:
            return self.file_name
        parts = self.path.name.split("_", 2)
        return parts[2] if len(parts) >= 3 else self.path.name


@dataclass(frozen=True)
class OutboundAttachment:
    """One validated workspace file to deliver through WhatsApp."""

    path: Path
    media_type: str

    @property
    def file_name(self) -> str:
        return self.path.name


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


def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _accept_outbound_path(raw: str, roots: Sequence[Path]) -> Optional[Path]:
    try:
        expanded = Path(raw).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None
    return _accept_path(str(expanded), roots)


def _outbound_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_DELIVERY_SUFFIXES:
        return "image"
    if suffix in _VIDEO_DELIVERY_SUFFIXES:
        return "video"
    return "document"


def _mask_protected_media_spans(content: str, roots: Sequence[Path]) -> str:
    """Port Hermes' guard against treating examples as live attachments."""
    chars = list(content)
    spans: List[tuple[int, int]] = []

    for match in re.finditer(r"```[^\n]*\n.*?```", content, re.DOTALL):
        spans.append((match.start(), match.end()))

    for match in re.finditer(r"`[^`\n]+`", content):
        start = match.start()
        prefix = content[max(0, start - 20):start]
        if re.search(r"MEDIA:\s*$", prefix):
            continue
        inner = match.group(0)[1:-1].strip()
        if inner.upper().startswith("MEDIA:"):
            candidate = _normalize_media_tag_path(inner[6:])
            if candidate and _accept_outbound_path(candidate, roots):
                continue
        spans.append((start, match.end()))

    for match in re.finditer(r"^>.*$", content, re.MULTILINE):
        spans.append((match.start(), match.end()))

    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _mask_json_media_values(content: str) -> str:
    """Port Hermes' guard against replaying a stored MEDIA string."""
    if '"' not in content or "MEDIA:" not in content:
        return content
    chars = list(content)
    for match in re.finditer(r'(?<=[:,{\[])\s*"((?:[^"\\\n]|\\.)*)"', content):
        if re.search(r"MEDIA:\s*(?:~/|/|[A-Za-z]:[/\\])", match.group(1)):
            for index in range(match.start(1), match.end(1)):
                if chars[index] != "\n":
                    chars[index] = " "
    return "".join(chars)


def extract_outbound(
    content: str, roots: Sequence[Path]
) -> tuple[List[OutboundAttachment], str]:
    """Extract safe Hermes ``MEDIA:`` directives and clean the visible text."""
    if "MEDIA:" not in content:
        return [], content

    scan = _mask_protected_media_spans(content, roots)
    scan = _mask_json_media_values(scan)
    attachments: List[OutboundAttachment] = []
    seen: set[Path] = set()
    spans: List[tuple[int, int]] = []

    for match in MEDIA_TAG_RE.finditer(scan):
        spans.append(match.span())
        raw = _normalize_media_tag_path(match.group("path"))
        accepted = _accept_outbound_path(raw, roots)
        if accepted is None:
            logger.warning("Rejected an outbound MEDIA path outside the workspace")
            continue
        if accepted in seen:
            continue
        seen.add(accepted)
        attachments.append(
            OutboundAttachment(path=accepted, media_type=_outbound_media_type(accepted))
        )

    if not spans:
        return attachments, content

    chars = list(content)
    for start, end in reversed(spans):
        del chars[start:end]
    cleaned = re.sub(r"\n{3,}", "\n\n", "".join(chars)).strip()
    return attachments, cleaned


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


def image_parts_with_paths(
    attachments: Sequence[Attachment],
) -> tuple[List[Dict[str, Any]], List[Path]]:
    """Build image parts and Hermes' model-visible local path handles."""
    parts: List[Dict[str, Any]] = []
    paths: List[Path] = []
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
        paths.append(attachment.path.resolve())
    return parts, paths


def image_parts(attachments: Sequence[Attachment]) -> List[Dict[str, Any]]:
    """Build the ``input_image`` parts for a Responses request."""
    parts, _ = image_parts_with_paths(attachments)
    return parts


def describe_unreadable(attachments: Sequence[Attachment]) -> str:
    """Name what arrived but cannot be read, so the model does not answer blind."""
    notes: List[str] = []
    for attachment in attachments:
        if attachment.is_image or attachment.is_text_document:
            continue
        if attachment.is_voice_message:
            # The asynchronous channel handler replaces this with either a
            # transcript or Hermes' neutral transcription-failure marker.
            continue
        if attachment.media_type == "audio":
            notes.append(
                "[The sender sent an audio file. This agent cannot listen to it automatically.]"
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
