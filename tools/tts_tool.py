#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

TTS provider: OpenAI (needs OPENAI_API_KEY or VOICE_TOOLS_OPENAI_KEY),
also covering any OpenAI-compatible speech endpoint via tts.openai.base_url.

Output formats:
- Opus (.ogg) for Telegram/WhatsApp voice bubbles (written natively)
- MP3 (.mp3) for everything else (CLI, Discord, WhatsApp)

Configuration is loaded from ~/.pilotage/config.yaml under the 'tts:' key.
The user chooses the provider and voice; the model just sends text.

Usage:
    from tools.tts_tool import text_to_speech_tool, check_tts_requirements

    result = text_to_speech_tool(text="Hello world")
"""

import asyncio
import base64
import datetime
import importlib.util
import json
import logging
import os
import queue
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from pilotage_cli._subprocess_compat import windows_hide_flags
from pilotage_constants import display_pilotage_home

logger = logging.getLogger(__name__)
def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``pilotage_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so TTS does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from pilotage_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver.

    Delegates to ``tools.tool_backend_helpers.resolve_provider_secret`` —
    the single owner of STT/TTS key resolution (config > env/.env > the
    credential pool populated by ``pilotage auth add <provider_id>``).
    Resolved at call time so tests that reload the helpers module see the
    live function.
    """
    try:
        from tools.tool_backend_helpers import resolve_provider_secret
    except ImportError:  # pragma: no cover — helpers are in-repo
        return str(get_env_value(env_var) or "").strip()
    return resolve_provider_secret(env_var, provider_id, env_getter=get_env_value)

from tools.tool_backend_helpers import (
    resolve_openai_audio_api_key,
)

# ---------------------------------------------------------------------------
# Lazy imports -- providers are imported only when actually used to avoid
# crashing in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------


def _import_openai_client():
    """Lazy import OpenAI client. Returns the class or raises ImportError."""
    from openai import OpenAI as OpenAIClient
    return OpenAIClient


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
TTS_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024
TTS_RESPONSE_BODY_CHUNK_BYTES = 64 * 1024

def _get_default_output_dir() -> str:
    from pilotage_constants import get_pilotage_dir
    return str(get_pilotage_dir("cache/audio", "audio_cache"))

DEFAULT_OUTPUT_DIR = _get_default_output_dir()

# ---------------------------------------------------------------------------
# Per-provider input-character limits (from official provider docs).
# OpenAI caps a single request at 4096 characters. Users can override this
# via ``tts.<provider>.max_text_length`` in config.yaml.
# ---------------------------------------------------------------------------
PROVIDER_MAX_TEXT_LENGTH: Dict[str, int] = {
    "openai": 4096,       # https://platform.openai.com/docs/guides/text-to-speech
}


def _config_bool(value: Any, default: bool = False) -> bool:
    """Coerce common YAML/env bool spellings without treating random strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _response_has_explicit_stream(response: Any) -> bool:
    iter_content = getattr(response, "iter_content", None)
    if not callable(iter_content):
        return False
    response_type = type(response)
    if response_type.__module__.startswith("requests."):
        return True
    return "iter_content" in vars(response_type)


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _read_tts_response_bytes(
    response: Any,
    *,
    label: str,
    limit: Optional[int] = None,
) -> bytes:
    """Read an upstream TTS response with a hard byte cap."""
    limit = TTS_RESPONSE_BODY_LIMIT_BYTES if limit is None else limit
    chunks: list[bytes] = []
    total = 0
    try:
        if _response_has_explicit_stream(response):
            iterator = response.iter_content(chunk_size=TTS_RESPONSE_BODY_CHUNK_BYTES)
        else:
            content = vars(response).get("content", getattr(type(response), "content", b""))
            if isinstance(content, str):
                content = content.encode("utf-8", errors="replace")
            iterator = (content,) if isinstance(content, (bytes, bytearray)) else ()

        for chunk in iterator:
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            chunk = bytes(chunk)
            total += len(chunk)
            if total > limit:
                _close_response(response)
                raise RuntimeError(f"{label} response exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        _close_response(response)


def _read_tts_response_json(
    response: Any,
    *,
    label: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    raw = _read_tts_response_bytes(response, label=label, limit=limit)
    if raw:
        return json.loads(raw.decode("utf-8"))

    # Unit-test doubles often only provide `.json()`. Real requests.Response
    # objects use the streaming path above, so this fallback does not re-open
    # the production eager-buffering behavior.
    if not _response_has_explicit_stream(response):
        json_reader = getattr(response, "json", None)
        if callable(json_reader):
            parsed = json_reader()
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _write_tts_response_to_file(
    response: Any,
    output_path: str,
    *,
    label: str,
    limit: Optional[int] = None,
) -> None:
    audio_bytes = _read_tts_response_bytes(response, label=label, limit=limit)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

# Final fallback when provider isn't recognised at all.
FALLBACK_MAX_TEXT_LENGTH = 4000

# Back-compat alias. Prefer ``_resolve_max_text_length()`` for new code.
MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


def _resolve_max_text_length(
    provider: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the input-character cap for *provider*.

    Resolution order:
      1. ``tts.<provider>.max_text_length`` (user override in config.yaml)
      2. ``PROVIDER_MAX_TEXT_LENGTH`` default
      3. ``FALLBACK_MAX_TEXT_LENGTH`` (4000)

    Non-positive or non-integer overrides fall through to the default so a
    broken config can't accidentally disable truncation entirely.
    """
    if not provider:
        return FALLBACK_MAX_TEXT_LENGTH
    key = provider.lower().strip()
    cfg = tts_config or {}

    # Built-in-style override at tts.<provider>.max_text_length wins first,
    # matching historical behavior.
    prov_cfg = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
    override = prov_cfg.get("max_text_length") if prov_cfg else None
    if isinstance(override, bool):
        override = None
    if isinstance(override, int) and override > 0:
        return override

    if key in PROVIDER_MAX_TEXT_LENGTH:
        return PROVIDER_MAX_TEXT_LENGTH[key]

    return FALLBACK_MAX_TEXT_LENGTH


# ===========================================================================
# Long-form chunking and delivery packing
# ===========================================================================

@dataclass(frozen=True)
class AudioDeliveryProfile:
    """Destination-platform constraints for generated TTS audio."""

    platform: str
    max_file_bytes: int
    safety_ratio: float = 0.85

    @property
    def target_file_bytes(self) -> int:
        """Conservative packing target below the platform hard limit."""
        return max(1, int(self.max_file_bytes * self.safety_ratio))


_PLATFORM_AUDIO_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "discord": {
        "max_file_bytes": 10 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
    "telegram": {
        "max_file_bytes": 50 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
    "default": {
        "max_file_bytes": 10 * 1024 * 1024,
        "safety_ratio": 0.85,
    },
}


def _resolve_audio_delivery_profile(
    platform: Optional[str],
    tts_config: Optional[Dict[str, Any]] = None,
) -> AudioDeliveryProfile:
    """Resolve upload constraints, including optional per-platform overrides."""
    key = (platform or "default").lower().strip() or "default"
    defaults = dict(
        _PLATFORM_AUDIO_DEFAULTS.get(key) or _PLATFORM_AUDIO_DEFAULTS["default"]
    )
    cfg = tts_config or {}
    profiles = cfg.get("delivery_profiles")
    overrides = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    max_file_bytes = defaults.get("max_file_bytes")
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or max_file_bytes <= 0
    ):
        max_file_bytes = _PLATFORM_AUDIO_DEFAULTS["default"]["max_file_bytes"]

    safety_ratio = defaults.get("safety_ratio", 0.85)
    if (
        isinstance(safety_ratio, bool)
        or not isinstance(safety_ratio, (int, float))
        or not 0 < safety_ratio <= 1
    ):
        safety_ratio = 0.85

    return AudioDeliveryProfile(
        platform=key,
        max_file_bytes=max_file_bytes,
        safety_ratio=float(safety_ratio),
    )


def _split_oversized_sentence(sentence: str, max_chars: int) -> List[str]:
    """Split one over-limit sentence on word boundaries, then hard boundaries."""
    words = sentence.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split text under a provider cap without dropping normalized content."""
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;:,])\s+", normalized)
        if sentence.strip()
    ]
    expanded: List[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            expanded.append(sentence)
        else:
            expanded.extend(_split_oversized_sentence(sentence, max_chars))

    chunks: List[str] = []
    current = ""
    for sentence in expanded:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _pack_audio_files_for_delivery(
    audio_paths: List[str],
    profile: AudioDeliveryProfile,
) -> List[List[str]]:
    """Group already-final-encoded chunks under the conservative size target."""
    groups: List[List[str]] = []
    current: List[str] = []
    current_size = 0
    current_suffix = ""
    for path in audio_paths:
        size = Path(path).stat().st_size
        suffix = Path(path).suffix.lower()
        if current and (
            current_size + size > profile.target_file_bytes
            or suffix != current_suffix
        ):
            groups.append(current)
            current = []
            current_size = 0
            current_suffix = ""
        current.append(path)
        current_size += size
        current_suffix = suffix
    if current:
        groups.append(current)
    return groups


# ===========================================================================
# Config loader -- reads tts: section from ~/.pilotage/config.yaml
# ===========================================================================
def _load_tts_config() -> Dict[str, Any]:
    """
    Load TTS configuration from ~/.pilotage/config.yaml.

    Returns a dict with provider settings. Falls back to defaults
    for any missing fields.
    """
    try:
        from pilotage_cli.config import load_config
        config = load_config()
        return config.get("tts") or {}
    except ImportError:
        logger.debug("pilotage_cli.config not available, using default TTS config")
        return {}
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
        return {}


def _get_provider(tts_config: Dict[str, Any]) -> str:
    """Return the configured TTS provider name.

    ``openai`` is the only built-in backend; ``tts.provider`` pins it
    explicitly and any other value is rejected at synthesis time.
    """
    return (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()


# ===========================================================================
# ``tts.providers``, so user config can't silently shadow ``edge`` etc.
#
# Placeholder values are shell-quoted for their surrounding context
# (bare / single / double quote), so paths with spaces work transparently.

# Built-in provider names. Any ``tts.provider`` value NOT in this set is
# interpreted as a reference to ``tts.providers.<name>``.
BUILTIN_TTS_PROVIDERS = frozenset({"openai"})


# Platforms whose native voice-bubble delivery requires Ogg/Opus audio.
# Previously only Telegram was recognized, so Matrix/Feishu/WhatsApp/Signal
# voice replies were synthesized as MP3 and rendered as broken attachments
# (, and siblings).
OPUS_VOICE_PLATFORMS = frozenset({
    "telegram",
    "matrix",
    "feishu",
    "whatsapp",
    "signal",
})


def _get_provider_section(tts_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a provider config block if it's a dict, else an empty dict."""
    if not isinstance(tts_config, dict):
        return {}
    section = tts_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_named_provider_config(
    tts_config: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    """Return the config dict for a user-declared provider.

    Looks up ``tts.providers.<name>`` first (the canonical location), and
    falls back to ``tts.<name>`` so users who followed the built-in layout
    still work. Returns an empty dict when the provider is not declared.
    """
    providers = _get_provider_section(tts_config, "providers")
    section = providers.get(name) if isinstance(providers, dict) else None
    if isinstance(section, dict):
        return section
    # Back-compat: allow ``tts.<name>`` for user-declared providers too,
    # but only when the name is not a built-in (so a user's ``tts.openai``
    # block still means the OpenAI provider, not a custom command).
    if name.lower() not in BUILTIN_TTS_PROVIDERS:
        legacy = _get_provider_section(tts_config, name)
        if legacy:
            return legacy
    return {}


# ===========================================================================
# ffmpeg Opus conversion (MP3/WAV -> OGG Opus for voice bubbles)
# ===========================================================================
def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _convert_to_opus(mp3_path: str) -> Optional[str]:
    """
    Convert an audio file (MP3/WAV/anything ffmpeg reads) to OGG Opus
    format for Telegram voice bubbles.

    Args:
        mp3_path: Path to the input audio file.

    Returns:
        Path to the .ogg file, or None if conversion fails.
    """
    if not _has_ffmpeg():
        return None

    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
    return _ffmpeg_transcode_to_opus(mp3_path, ogg_path)


def _ffmpeg_transcode_to_opus(input_path: str, ogg_path: str) -> Optional[str]:
    """Transcode *input_path* to real Ogg/Opus at *ogg_path* via ffmpeg.

    Safe when ``input_path == ogg_path`` (writes to a temp file, then
    replaces). Returns the output path on success, None on failure.
    """
    if not _has_ffmpeg():
        return None

    in_place = os.path.abspath(input_path) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-acodec", "libopus",
             "-ac", "1", "-b:a", "48k", "-vbr", "on",
             "-application", "voip", "-compression_level", "10", "-f", "ogg",
             work_path, "-y"],
            capture_output=True, timeout=30,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s",
                          result.returncode, result.stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if os.path.exists(work_path) and os.path.getsize(work_path) > 0:
            if in_place:
                os.replace(work_path, ogg_path)
            return ogg_path
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg OGG conversion timed out after 30s")
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    finally:
        if in_place and os.path.exists(work_path):
            try:
                os.remove(work_path)
            except OSError:
                pass
    return None


# ---------------------------------------------------------------------------
# Container sniffing - guard against "MP3/WAV bytes in a .ogg file".
# Some OpenAI-compatible servers reject or ignore response_format="opus",
# which breaks native voice bubbles on Telegram/Matrix/Feishu/WhatsApp.
# Sniff the magic bytes once after synthesis and repair the container when
# it does not match the extension.
# ---------------------------------------------------------------------------

def _sniff_audio_container(path: str) -> str:
    """Return a container id ('ogg', 'wav', 'mp3', 'flac', ...) or 'unknown'.

    Delegates to the shared magic-byte sniffer in ``tools.audio_container``
    (one module owns container detection for both this outbound repair and
    the inbound gateway audio cache).
    """
    from tools.audio_container import sniff_container

    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return "unknown"
    return sniff_container(head) or "unknown"


def _repair_ogg_container(file_str: str) -> str:
    """Ensure a path claiming ``.ogg`` actually contains an Ogg container.

    When the bytes are MP3/WAV/FLAC (a backend ignored the opus request),
    transcode in place to real Ogg/Opus. On any failure, rename to the
    sniffed real extension so downstream players/platforms at least get an
    honest file instead of a 0-second voice bubble. Returns the (possibly
    updated) path.
    """
    if not file_str.endswith(".ogg"):
        return file_str
    container = _sniff_audio_container(file_str)
    if container in ("ogg", "unknown"):
        return file_str

    logger.info(
        "TTS wrote %s bytes into a .ogg path (%s) — transcoding to real Ogg/Opus",
        container, file_str,
    )
    repaired = _ffmpeg_transcode_to_opus(file_str, file_str)
    if repaired:
        return repaired

    # ffmpeg unavailable/failed: rename to the honest extension.
    honest = file_str[:-4] + "." + container
    try:
        os.replace(file_str, honest)
        logger.warning(
            "Could not transcode %s to Ogg/Opus — renamed to %s so the "
            "file is delivered with its real format", file_str, honest,
        )
        return honest
    except OSError:
        return file_str


# ===========================================================================
# Long-form audio combination and delivery packing
# ===========================================================================

def _concat_audio_files(
    audio_paths: List[str],
    output_path: str,
    *,
    voice_compatible: bool = False,
) -> Optional[str]:
    """Combine independently encoded chunks with ffmpeg.

    OGG/Opus is always decoded and re-encoded, even when a custom provider did
    not opt in to voice-message presentation. Matching MP3 chunks preserve their
    encoded frames. A failed or unavailable combine returns ``None`` so callers
    can preserve the original, individually valid files. Structured audio
    containers are never byte-joined.
    """
    if not audio_paths:
        raise ValueError("No audio chunks to combine")
    if len(audio_paths) == 1:
        source = audio_paths[0]
        if os.path.abspath(source) != os.path.abspath(output_path):
            shutil.copyfile(source, output_path)
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.concat.txt")
    temp_output = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.combining{destination.suffix}"
    )
    try:
        with concat_path.open("w", encoding="utf-8") as concat_file:
            for path in audio_paths:
                concat_file.write(f"file {shlex.quote(os.path.abspath(path))}\n")

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vn",
        ]
        suffix = destination.suffix.lower()
        if voice_compatible or suffix in {".ogg", ".opus"}:
            command.extend([
                "-c:a", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off",
            ])
        elif suffix == ".mp3" and all(
            Path(path).suffix.lower() == ".mp3" for path in audio_paths
        ):
            # Matching MP3 provider chunks already share one output codec/config.
            # Preserve those encoded frames instead of imposing a second lossy pass.
            command.extend(["-c:a", "copy"])
        command.append(str(temp_output))

        result = subprocess.run(
            command,
            capture_output=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if (
            result.returncode == 0
            and temp_output.exists()
            and temp_output.stat().st_size > 0
        ):
            os.replace(temp_output, destination)
            return str(destination)
        logger.warning(
            "ffmpeg audio combine failed: %s",
            result.stderr.decode("utf-8", errors="ignore")[:500],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg audio combine failed: %s", exc)
    finally:
        for path in (concat_path, temp_output):
            try:
                path.unlink()
            except OSError:
                pass
    return None


def _build_audio_delivery_files(
    audio_paths: List[str],
    output_path: str,
    profile: AudioDeliveryProfile,
    *,
    voice_compatible: bool = False,
) -> Tuple[List[str], bool]:
    """Pack final-encoded chunks and enforce the hard upload limit.

    Packing uses the conservative target. Every combined artifact is then
    checked at its actual post-encoding size; an over-limit group is split and
    retried. If combining fails, the valid constituent files are returned
    separately. A single final-encoded chunk above the hard limit fails closed
    rather than returning an upload that the destination will reject.
    """
    if not audio_paths:
        raise ValueError("No final-encoded TTS audio chunks")
    for path in audio_paths:
        size = Path(path).stat().st_size
        if size > profile.max_file_bytes:
            raise ValueError(
                f"Final-encoded TTS chunk exceeds {profile.platform} delivery "
                f"limit ({size} > {profile.max_file_bytes} bytes): {path}"
            )

    base = Path(output_path)
    scratch_outputs: List[str] = []
    combined_any = False
    combine_index = 0

    def emit(group: List[str]) -> List[str]:
        nonlocal combined_any, combine_index
        if len(group) == 1:
            return list(group)

        combine_index += 1
        scratch = base.with_name(
            f".{base.stem}.delivery{combine_index:03d}.{uuid.uuid4().hex}{base.suffix}"
        )
        combined = _concat_audio_files(
            group, str(scratch), voice_compatible=voice_compatible,
        )
        if not combined:
            return list(group)
        scratch_outputs.append(combined)
        combined_size = Path(combined).stat().st_size
        if combined_size <= profile.max_file_bytes:
            combined_any = True
            return [combined]

        try:
            Path(combined).unlink()
        except OSError:
            pass
        midpoint = max(1, len(group) // 2)
        return emit(group[:midpoint]) + emit(group[midpoint:])

    packed: List[str] = []
    for group in _pack_audio_files_for_delivery(audio_paths, profile):
        packed.extend(emit(group))

    final_paths: List[str] = []
    for index, source in enumerate(packed, start=1):
        if len(packed) == 1:
            destination = base
        else:
            source_suffix = Path(source).suffix or base.suffix
            destination = base.with_name(
                f"{base.stem}.part{index:02d}{source_suffix}"
            )
        if os.path.abspath(source) != os.path.abspath(destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if destination.stat().st_size > profile.max_file_bytes:
            raise ValueError(
                f"Final TTS deliverable exceeds {profile.platform} delivery limit: "
                f"{destination}"
            )
        final_paths.append(str(destination))

    try:
        return final_paths, combined_any
    finally:
        for scratch in scratch_outputs:
            if scratch not in final_paths:
                try:
                    Path(scratch).unlink()
                except OSError:
                    pass


def _tts_response_format_from_path(output_path: str) -> str:
    """Pick an OpenAI-compatible TTS response format from the output extension."""
    if output_path.endswith(".ogg"):
        return "opus"
    if output_path.endswith(".wav"):
        return "wav"
    if output_path.endswith(".flac"):
        return "flac"
    return "mp3"


# ===========================================================================
# Provider: OpenAI TTS (and any OpenAI-compatible speech endpoint).
# ===========================================================================
def _generate_openai_tts(
    text: str,
    output_path: str,
    tts_config: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
) -> str:
    """Generate audio via the OpenAI ``audio.speech.create`` SDK shape.

    Optional kwargs let OpenAI-compatible backends (DeepInfra etc.) reuse
    this function — they resolve credentials/model themselves and pass
    them through, skipping the OpenAI-only ``_resolve_openai_audio_client_config``.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict (used for ``tts.openai`` sub-block
            and the global ``speed`` default).
        api_key: Bearer token. When None, resolved from the OpenAI auth
            chain (config → env).
        base_url: API base URL. When None, falls back to
            ``tts.openai.base_url`` then the OpenAI default.
        model: Model id. When None, reads ``tts.openai.model``.
        voice: Voice id. When None, reads ``tts.openai.voice``.
        speed: Playback speed. When None, reads ``tts.openai.speed`` /
            ``tts.speed``.
        instructions: Optional voice-design guidance (tone, emotion, pacing,
            accent, whispering). Forwarded to `audio.speech.create` when
            truthy; omitted otherwise so ``tts-1``/``tts-1-hd`` and strict
            OpenAI-compatible servers that reject unknown kwargs are
            unaffected.

    Returns:
        Path to the saved audio file.
    """
    # Only resolve the OpenAI auth chain when the caller didn't pass explicit
    # credentials. OpenAI-compatible backends (DeepInfra) pass api_key /
    # base_url / model / voice through.
    fallback_base: Optional[str] = None
    if api_key is None:
        api_key, fallback_base = _resolve_openai_audio_client_config()

    # ``tts.openai: null`` in YAML yields None — coalesce so .get() is safe.
    oai_config = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    if model is None:
        model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    if voice is None:
        voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)
    config_base_url = oai_config.get("base_url")
    if base_url is None:
        # Config override wins over the auth-chain fallback (restores the
        # pre-refactor precedence, where tts.openai.base_url beat the resolved
        # default); the auth-chain value is the last-resort default. An
        # explicit base_url arg from an OpenAI-compatible caller (DeepInfra)
        # skips this block entirely and always wins.
        base_url = config_base_url or fallback_base or DEFAULT_OPENAI_BASE_URL
    if speed is None:
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        speed = float(oai_config.get("speed", speed_default))
    language = oai_config.get("language")


    response_format = _tts_response_format_from_path(output_path)

    OpenAIClient = _import_openai_client()
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    try:
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "extra_headers": {"x-idempotency-key": str(uuid.uuid4())},
        }
        if speed != 1.0:
            create_kwargs["speed"] = max(0.25, min(4.0, speed))
        if instructions:
            create_kwargs["instructions"] = instructions
        if language:
            create_kwargs["extra_body"] = {"lang_code": language}
        response = client.audio.speech.create(**create_kwargs)

        response.stream_to_file(output_path)
        return output_path
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ===========================================================================
# Provider: DeepInfra TTS
# ===========================================================================
#
# DeepInfra serves TTS over an OpenAI-compatible /v1/openai/audio/speech
# endpoint. Models are discovered live via the shared catalog helper
# (filtered by the ``tts`` surface tag) — no hardcoded model ids in this
# file, so retired models disappear from pilotage the next time the
# catalog is fetched without a patch.


# ===========================================================================
# Provider: MiniMax TTS
# ===========================================================================


# ===========================================================================
# Provider: Mistral (Voxtral TTS)
# ===========================================================================


# ===========================================================================
# Provider: Google Gemini TTS
# ===========================================================================


# ===========================================================================
# ===========================================================================


# ===========================================================================


# ===========================================================================


# ===========================================================================
# Main tool function
# ===========================================================================
def _text_to_speech_single(
    text: str,
    output_path: Optional[str] = None,
    *,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
    tts_config_override: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesize one provider-safe text chunk and return one final-encoded file.

    The public :func:`text_to_speech_tool` wrapper owns long-form splitting,
    delivery packing, and post-encoding size enforcement.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    # The wrapper already normalizes text via prepare_spoken_text; the inner
    # function should not re-normalize or truncate.
    tts_config = (
        tts_config_override
        if tts_config_override is not None
        else _load_tts_config()
    )

    # When the model supplies a speed parameter, inject it into the config
    # so all downstream provider functions pick it up uniformly.
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)  # shallow copy to avoid mutating the cache
        tts_config["speed"] = clamped

    # Allow per-call provider override; fall back to the configured default.
    if provider:
        provider = provider.lower().strip()
    else:
        provider = _get_provider(tts_config)

    # The wrapper splits text into provider-safe chunks before calling this
    # function. If text exceeds the cap here, it means the caller bypassed
    # the wrapper — log a warning but don't silently truncate.
    max_len = _resolve_max_text_length(provider, tts_config)
    if len(text) > max_len:
        logger.warning(
            "TTS text exceeds provider %s cap (%d > %d chars) — "
            "use text_to_speech_tool() for automatic chunking",
            provider, len(text), max_len,
        )

    # Detect platform from gateway env var to choose the best output format.
    # Several platforms deliver native voice bubbles only for Ogg/Opus
    # (Telegram, Matrix, Feishu/Lark, WhatsApp, Signal); OpenAI can emit
    # Opus natively, so no ffmpeg conversion is needed for them.
    from gateway.session_context import get_session_env
    platform = get_session_env("PILOTAGE_SESSION_PLATFORM", "").lower()
    want_opus = platform in OPUS_VOICE_PLATFORMS

    # Determine output path
    if output_path:
        # Reject '..' traversal components in the user-supplied path. An
        # explicit absolute path is fine (the agent legitimately writes
        # audio to user-specified locations), but a path that uses ``..``
        # to escape its declared base is almost always either a bug or
        # prompt-injection-controlled — e.g.
        # ``output_path="audio/../../etc/cron.d/x"``. The terminal tool
        # can still write anywhere with approval; this just keeps the
        # unattended TTS surface from materializing files via traversal.
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path contains '..' traversal component: "
                    f"{output_path}. Use an absolute path or one relative "
                    "to the current directory without '..'."
                ),
            }, ensure_ascii=False)
        file_path = Path(output_path).expanduser()
        from agent.file_safety import is_write_approval_required, is_write_denied

        if is_write_denied(str(file_path)) or is_write_approval_required(str(file_path)):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path targets a protected credential or system path: "
                    f"{file_path}. Choose a normal audio output location."
                ),
            }, ensure_ascii=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use .ogg where the platform needs a native Opus voice bubble.
        if want_opus:
            file_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            file_path = out_dir / f"tts_{timestamp}.mp3"

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_str = str(file_path)

    try:
        # Generate audio with the configured provider
        if provider != "openai":
            return json.dumps({
                "success": False,
                "error": f"Unsupported TTS provider '{provider}'. Only 'openai' is available."
            }, ensure_ascii=False)

        try:
            _import_openai_client()
        except ImportError:
            return json.dumps({
                "success": False,
                "error": "OpenAI provider selected but 'openai' package not installed."
            }, ensure_ascii=False)
        logger.info("Generating speech with OpenAI TTS...")
        _generate_openai_tts(text, file_str, tts_config, instructions=instructions)

        # Check the file was actually created
        if not os.path.exists(file_str) or os.path.getsize(file_str) == 0:
            return json.dumps({
                "success": False,
                "error": f"TTS generation produced no output (provider: {provider})"
            }, ensure_ascii=False)

        # Container repair: an OpenAI-compatible endpoint without opus
        # support may write MP3/WAV bytes into a .ogg output path, which
        # platforms like Telegram render as a broken 0-second voice
        # bubble. Sniff the magic bytes and transcode in place when they
        # do not match the extension.
        file_str = _repair_ogg_container(file_str)

        # OpenAI writes Opus directly into the .ogg path chosen above, so
        # a voice bubble is available whenever the platform wanted one.
        voice_compatible = want_opus and file_str.endswith(".ogg")

        file_size = os.path.getsize(file_str)
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{file_size:,}", provider)

        # Build response with MEDIA tag for platform delivery
        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider,
            "voice_compatible": voice_compatible,
        }, ensure_ascii=False)

    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        error_msg = f"TTS configuration error ({provider}): {e}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except FileNotFoundError as e:
        # Missing dependencies or files
        error_msg = f"TTS dependency missing ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    except Exception as e:
        # Unexpected errors
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)


def text_to_speech_tool(
    text: str,
    output_path: Optional[str] = None,
    speed: Optional[float] = None,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Convert text to speech audio with long-form chunking.

    Long text is normalized, split into provider-safe chunks, synthesized
    sequentially, and packed against destination platform upload limits.
    Each provider request is encoded to its final format before files are
    packed. Multi-chunk voice output is re-encoded when combined; failed
    combines preserve separate valid files, and no over-limit final artifact
    is returned.

    On messaging platforms, the returned MEDIA:<path> tag is intercepted
    by the send pipeline and delivered as a native voice message.
    In CLI mode, the file is saved to ~/voice-memos/.

    Args:
        text: The text to convert to speech. The 4096-character OpenAI
            per-request cap applies automatically; longer input is split
            into ordered chunks without silent truncation.
        output_path: Optional custom save path.
        speed: Optional playback speed multiplier (0.25-4.0).
        instructions: Optional voice-design guidance (tone, emotion, pacing).
        provider: Optional TTS provider override.

    Returns:
        str: JSON result with success, file_path, file_paths, and MEDIA tag.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    # Normalize text via the shared cleaner: markdown, emoji, think blocks,
    # verifier footer, units, newline flattening.
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)

    tts_config = _load_tts_config()

    # When the model supplies a speed parameter, inject it into the config
    # so all downstream provider functions pick it up uniformly.
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)  # shallow copy to avoid mutating the cache
        tts_config["speed"] = clamped

    # Allow per-call provider override; fall back to the configured default.
    if provider:
        provider = provider.lower().strip()
    else:
        provider = _get_provider(tts_config)

    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info(
            "TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
            provider,
            len(chunks),
            len(text),
            max_len,
        )

    from gateway.session_context import get_session_env
    platform = get_session_env("PILOTAGE_SESSION_PLATFORM", "").lower()
    want_opus = platform in OPUS_VOICE_PLATFORMS
    delivery_profile = _resolve_audio_delivery_profile(platform, tts_config)

    # Determine output path (single-chunk short-circuit uses the final path).
    if output_path:
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path contains '..' traversal component: {output_path}. "
                    "Use an absolute path or one relative to the current directory "
                    "without '..'."
                ),
            }, ensure_ascii=False)
        base_path = Path(output_path).expanduser()
        from agent.file_safety import is_write_approval_required, is_write_denied
        if is_write_denied(str(base_path)) or is_write_approval_required(str(base_path)):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path targets a protected credential or system path: "
                    f"{base_path}. Choose a normal audio output location."
                ),
            }, ensure_ascii=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        if want_opus:
            base_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            base_path = out_dir / f"tts_{timestamp}.mp3"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    generated_artifacts: set[str] = set()
    final_paths: List[str] = []
    chunk_results: List[Dict[str, Any]] = []
    try:
        encoded_paths: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            if len(chunks) == 1:
                chunk_path = base_path
            else:
                chunk_path = base_path.with_name(
                    f"{base_path.stem}.chunk{index:03d}{base_path.suffix}"
                )
            generated_artifacts.add(str(chunk_path))
            raw_result = _text_to_speech_single(
                text=chunk,
                output_path=str(chunk_path),
                speed=speed,
                instructions=instructions,
                provider=provider,
                tts_config_override=tts_config,
            )
            try:
                chunk_result = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError):
                raise RuntimeError(
                    f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}"
                )
            if not chunk_result.get("success"):
                error_msg = chunk_result.get("error", "unknown error")
                return tool_error(
                    f"TTS chunk {index} failed ({provider}): {error_msg}",
                    success=False,
                )
            actual_path = str(chunk_result.get("file_path") or chunk_path)
            if not os.path.isfile(actual_path) or os.path.getsize(actual_path) <= 0:
                raise RuntimeError(
                    f"TTS chunk {index} produced no final audio: {actual_path}"
                )
            generated_artifacts.add(actual_path)
            encoded_paths.append(actual_path)
            chunk_results.append(chunk_result)

        voice_compatible = bool(chunk_results) and all(
            bool(result.get("voice_compatible")) for result in chunk_results
        )
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = _build_audio_delivery_files(
            encoded_paths,
            str(delivery_base),
            delivery_profile,
            voice_compatible=voice_compatible,
        )

        for path in final_paths:
            logger.info(
                "TTS audio saved: %s (%s bytes, provider: %s)",
                path,
                f"{os.path.getsize(path):,}",
                provider,
            )
        media_tag = "\n".join(f"MEDIA:{path}" for path in final_paths)
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": final_paths[0],
            "file_paths": final_paths,
            "media_tag": media_tag,
            "provider": chunk_results[0].get("provider", provider),
            "voice_compatible": voice_compatible,
            "chunk_count": len(chunks),
            "delivery_file_count": len(final_paths),
            "combined_chunks": bool(combined_chunks),
            "delivery_profile": {
                "platform": delivery_profile.platform,
                "max_file_bytes": delivery_profile.max_file_bytes,
                "target_file_bytes": delivery_profile.target_file_bytes,
            },
        }, ensure_ascii=False)
    except ValueError as exc:
        error_msg = f"TTS delivery error ({provider}): {exc}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except Exception as exc:
        error_msg = f"TTS long-form generation failed ({provider}): {exc}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) in final_absolute:
                continue
            try:
                os.unlink(artifact)
            except OSError:
                pass


# ===========================================================================
# Requirements check
# ===========================================================================
def check_tts_requirements() -> bool:
    """Return whether the explicitly resolved TTS provider can run.

    Availability must mirror :func:`text_to_speech_tool` dispatch: the
    OpenAI SDK has to be importable and a key resolvable from config or env.
    """
    tts_config = _load_tts_config()
    provider = _get_provider(tts_config)
    if provider != "openai":
        return False
    if importlib.util.find_spec("openai") is None:
        return False
    return _has_openai_audio_backend()


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for the OpenAI audio client.

    Resolution order (mirrors the STT resolver):
    1. ``tts.openai.api_key`` / ``tts.openai.base_url`` from ``config.yaml``
    2. ``VOICE_TOOLS_OPENAI_KEY`` / ``OPENAI_API_KEY`` environment variables
       (still honoring ``tts.openai.base_url`` when set)
    """
    tts_config = _load_tts_config()
    openai_cfg = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    cfg_api_key = openai_cfg.get("api_key") or ""
    cfg_base_url = openai_cfg.get("base_url") or ""
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL)

    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL)

    raise ValueError(
        "Neither tts.openai.api_key in config nor "
        "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
    )


def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use config/env credentials."""
    openai_cfg = (_load_tts_config().get("openai") or {})
    if openai_cfg.get("api_key"):
        return True
    return bool(resolve_openai_audio_api_key())


# ===========================================================================
# Streaming TTS: sentence-by-sentence pipeline
# ===========================================================================
# Markdown stripping patterns (same as cli.py _voice_speak_response)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_URL = re.compile(r'https?://\S+')
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_INLINE_CODE = re.compile(r'`(.+?)`')
_MD_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
_MD_LIST_ITEM = re.compile(r'^\s*[-*]\s+', flags=re.MULTILINE)
_MD_HR = re.compile(r'---+')
_MD_EXCESS_NL = re.compile(r'\n{3,}')
# Emoji + variation selectors/ZWJ — TTS providers render these as awkward
# pauses or literal descriptions ("smiling face"), breaking the speech flow.
_EMOJI = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\U000E0020-\U000E007F]+'
)

# Strip <think>...</think> reasoning blocks before TTS — models with
# /reasoning show enabled produce think blocks that shouldn't be spoken.
_THINK_BLOCK = re.compile(r'<think[\s>].*?</think>', flags=re.DOTALL)


def _strip_markdown_for_tts(text: str) -> str:
    """Prepare text for speech via the shared cleaner in tts_text_normalize.

    One cleaner for every TTS path (tool, gateway auto-TTS, voice-mode
    streaming, web dashboard): strips <think> reasoning blocks, the
    file-mutation verifier footer, markdown, and emoji; expands units and
    symbols; and flattens newlines to sentence breaks so newline-sensitive
    providers (Kokoro) speak the whole script.  Falls back to the legacy
    regex pipeline if the normalizer ever fails.
    """
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        return prepare_spoken_text(text, max_chars=None)
    except Exception:
        pass
    text = _THINK_BLOCK.sub(' ', text)
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _EMOJI.sub(' ', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()


# ===========================================================================
# Main -- quick diagnostics
# ===========================================================================
if __name__ == "__main__":
    print("🔊 Text-to-Speech Tool Module")
    print("=" * 50)

    def _check(importer, label):
        try:
            importer()
            return True
        except ImportError:
            return False

    print("\nProvider availability:")
    print(f"  OpenAI:     {'installed' if _check(_import_openai_client, 'oai') else 'not installed'}")
    print(
        "    API Key:  "
        f"{'set' if resolve_openai_audio_api_key() else 'not set (VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY)'}"
    )
    config = _load_tts_config()
    print(f"  ffmpeg:     {'✅ found' if _has_ffmpeg() else '❌ not found (needed for Telegram Opus)'}")
    print(f"\n  Output dir: {DEFAULT_OUTPUT_DIR}")

    provider = _get_provider(config)
    print(f"  Configured provider: {provider}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and model are user-configured under tts.openai, not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. The 4096-character OpenAI per-request cap applies automatically; longer input is split into ordered chunks without silent truncation."
            },
            "output_path": {
                "type": "string",
                "description": f"Optional custom file path to save the audio. Defaults to {display_pilotage_home()}/audio_cache/<timestamp>.mp3"
            },
            "speed": {
                "type": "number",
                "description": "Playback speed multiplier. 1.0 = normal, 0.5 = very slow (language learning), 2.0 = fast. Range: 0.25-4.0. Overrides the speed configured in config.yaml."
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional voice-design guidance: tone, emotion, pacing, accent, "
                    "whispering, impressions (e.g. 'Speak in a cheerful, excited whisper'). "
                    "Forwarded to the OpenAI backend (gpt-4o-mini-tts and OpenAI-compatible "
                    "voice-design servers). Silently ignored by backends that don't support it."
                )
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional TTS provider override. Only 'openai' is "
                    "available; when omitted, tts.provider from config.yaml is used."
                )
            }
        },
        "required": ["text"]
    }
}

registry.register(
    name="text_to_speech",
    toolset="tts",
    schema=TTS_SCHEMA,
    handler=lambda args, **kw: text_to_speech_tool(
        text=args.get("text", ""),
        output_path=args.get("output_path"),
        speed=args.get("speed"),
        instructions=args.get("instructions"),
        provider=args.get("provider")),
    check_fn=check_tts_requirements,
    emoji="🔊",
)
