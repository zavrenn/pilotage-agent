"""OpenAI speech-to-text for inbound voice messages.

This is the production-used slice of Hermes' transcription_tools module:
OpenAI transcription, the 25 MiB upload boundary, optional silence trimming,
and the Ogg/Opus-to-m4a retry used by messaging voice notes. Provider
discovery, local models, lazy installation, and plugin backends are outside the
Genesis boundary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .media import Attachment
from .settings import ConfigError, Settings
from .tools.file_safety import get_read_block_error

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_STT_MODEL = "whisper-1"
OPENAI_MODELS = frozenset(
    {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"}
)
SUPPORTED_FORMATS = frozenset(
    {
        ".aac", ".flac", ".m4a", ".mp3", ".mp4", ".mpeg",
        ".mpga", ".oga", ".ogg", ".opus", ".wav", ".webm",
    }
)
MAX_FILE_SIZE = 25 * 1024 * 1024

_WHISPER_PROMPT_MAX_CHARS = 224 * 4
_CLOUD_TRIM_THRESHOLD_DB = -40
_CLOUD_TRIM_KEEP_MS = 300
_CLOUD_TRIM_MIN_SAVING = 0.10
_CLOUD_TRIM_MIN_RESULT_SECONDS = 0.3
_CLOUD_TRIM_MIN_INPUT_SECONDS = 12.0
_STT_M4A_ENCODE_ARGS = (
    "-vn", "-ac", "1", "-ar", "16000",
    "-c:a", "aac", "-b:a", "32k", "-movflags", "+faststart",
)


def validate_settings(settings: Settings) -> None:
    """Validate the fixed OpenAI STT configuration at startup."""
    if settings.get("stt.openai.api_key") is not None:
        raise ConfigError(
            "stt.openai.api_key is secret; set VOICE_TOOLS_OPENAI_KEY in "
            "~/.pilotage-agent/.env instead"
        )
    settings.flag("stt.enabled", True)
    settings.flag("stt.echo_transcripts", True)
    provider = settings.text("stt.provider", "openai").lower()
    if provider != "openai":
        raise ConfigError(
            f"stt.provider must be 'openai' in this runtime, not {provider!r}"
        )
    model = settings.text("stt.openai.model", DEFAULT_STT_MODEL)
    if model not in OPENAI_MODELS:
        supported = ", ".join(sorted(OPENAI_MODELS))
        raise ConfigError(
            f"stt.openai.model must be one of {supported}, not {model!r}"
        )
    settings.text("stt.language", "")
    settings.text("stt.openai.language", "")
    settings.text("stt.prompt", "")
    settings.flag("stt.cloud_trim_silence", True)
    threshold = settings.count(
        "stt.cloud_trim_threshold_db", _CLOUD_TRIM_THRESHOLD_DB
    )
    if not -100 <= threshold <= 0:
        raise ConfigError("stt.cloud_trim_threshold_db must be between -100 and 0")
    keep_ms = settings.count("stt.cloud_trim_keep_ms", _CLOUD_TRIM_KEEP_MS)
    if not 0 <= keep_ms <= 5000:
        raise ConfigError("stt.cloud_trim_keep_ms must be between 0 and 5000")


def transcript_echo_enabled(settings: Settings) -> bool:
    return settings.flag("stt.echo_transcripts", True)


def _resolve_openai_audio_api_key() -> str:
    """Hermes' direct OpenAI voice-key precedence."""
    return (
        os.environ.get("VOICE_TOOLS_OPENAI_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )


def _language(settings: Settings) -> Optional[str]:
    return (
        settings.text("stt.openai.language", "")
        or settings.text("stt.language", "")
        or None
    )


def _prompt(settings: Settings) -> Optional[str]:
    prompt = settings.text("stt.prompt", "") or None
    if prompt and len(prompt) > _WHISPER_PROMPT_MAX_CHARS:
        logger.warning(
            "Transcription prompt exceeds Whisper's context window; keeping its tail"
        )
        return prompt[-_WHISPER_PROMPT_MAX_CHARS:]
    return prompt


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _find_binary(name: str) -> Optional[str]:
    return shutil.which(name)


def _run_ffmpeg_stt_encode(
    ffmpeg: str,
    input_path: str,
    output_path: str,
    *,
    audio_filter: Optional[str] = None,
) -> None:
    command = [ffmpeg, "-y", "-i", input_path]
    if audio_filter:
        command += ["-af", audio_filter]
    command += [*_STT_M4A_ENCODE_ARGS, output_path]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        stdin=subprocess.DEVNULL,
        creationflags=_creation_flags(),
    )


def _transcode_audio_for_stt(
    file_path: str, work_dir: str
) -> tuple[Optional[str], Optional[str]]:
    """Normalize a provider-rejected voice note to compact AAC/m4a. (Hermes)"""
    ffmpeg = _find_binary("ffmpeg")
    if not ffmpeg:
        return None, "audio needs transcoding for the STT API, but ffmpeg was not found"
    converted_path = os.path.join(
        work_dir, f"{Path(file_path).stem or 'audio'}-stt.m4a"
    )
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, converted_path)
        return converted_path, None
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        logger.error("ffmpeg STT transcode failed for %s: %s", file_path, details)
        return None, f"failed to transcode audio for the STT API: {details}"
    except Exception as exc:  # noqa: BLE001 - best-effort compatibility retry
        logger.error("unexpected STT transcode failure for %s: %s", file_path, exc)
        return None, f"failed to transcode audio for the STT API: {exc}"


def _probe_audio_duration(file_path: str) -> Optional[float]:
    ffprobe = _find_binary("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            stdin=subprocess.DEVNULL,
            creationflags=_creation_flags(),
        )
        return float(result.stdout.strip())
    except Exception:  # noqa: BLE001 - probing is optional
        return None


def _trim_silence_for_cloud_stt(
    file_path: str, settings: Settings
) -> Optional[str]:
    """Return a useful silence-trimmed copy, or leave the original untouched."""
    if not settings.flag("stt.cloud_trim_silence", True):
        return None
    ffmpeg = _find_binary("ffmpeg")
    if not ffmpeg:
        return None
    original_duration = _probe_audio_duration(file_path)
    if not original_duration or original_duration < _CLOUD_TRIM_MIN_INPUT_SECONDS:
        return None

    threshold_db = settings.count(
        "stt.cloud_trim_threshold_db", _CLOUD_TRIM_THRESHOLD_DB
    )
    keep_ms = settings.count("stt.cloud_trim_keep_ms", _CLOUD_TRIM_KEEP_MS)
    keep_seconds = keep_ms / 1000.0
    audio_filter = (
        "silenceremove="
        f"start_periods=1:start_threshold={threshold_db}dB:"
        f"start_silence={keep_seconds}:stop_periods=-1:"
        f"stop_threshold={threshold_db}dB:stop_silence={keep_seconds}"
    )
    work_dir = tempfile.mkdtemp(prefix="pilotage-stt-trim-")
    trimmed_path = os.path.join(
        work_dir, f"{Path(file_path).stem or 'audio'}-trimmed.m4a"
    )
    keep_result = False
    try:
        _run_ffmpeg_stt_encode(
            ffmpeg, file_path, trimmed_path, audio_filter=audio_filter
        )
        trimmed_duration = _probe_audio_duration(trimmed_path)
        minimum = max(_CLOUD_TRIM_MIN_RESULT_SECONDS, 2 * keep_seconds)
        if not trimmed_duration or trimmed_duration < minimum:
            return None
        if trimmed_duration > original_duration * (1 - _CLOUD_TRIM_MIN_SAVING):
            return None
        logger.info(
            "Trimmed silence before STT upload (%.1fs to %.1fs)",
            original_duration,
            trimmed_duration,
        )
        keep_result = True
        return trimmed_path
    except Exception as exc:  # noqa: BLE001 - upload the original on trim failure
        logger.debug("Cloud STT silence trim failed for %s: %s", file_path, exc)
        return None
    finally:
        if not keep_result:
            shutil.rmtree(work_dir, ignore_errors=True)


def _validate_audio_file(file_path: str) -> Optional[Dict[str, Any]]:
    blocked = get_read_block_error(file_path)
    if blocked:
        return {"success": False, "transcript": "", "error": blocked}
    path = Path(file_path)
    if path.is_symlink():
        return {
            "success": False,
            "transcript": "",
            "error": f"Path is a symbolic link: {file_path}",
        }
    if not path.is_file():
        return {
            "success": False,
            "transcript": "",
            "error": f"Audio file not found: {file_path}",
        }
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported audio format: {path.suffix}",
        }
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": f"Failed to access audio file: {exc}",
        }
    if size > MAX_FILE_SIZE:
        return {
            "success": False,
            "transcript": "",
            "error": (
                f"Audio file is too large: {size / (1024 * 1024):.1f}MB "
                "(max 25MB)"
            ),
        }
    return None


def _extract_transcript_text(transcription: Any) -> str:
    text: Optional[str] = None
    if isinstance(transcription, str):
        text = transcription.strip()
    if text is None and hasattr(transcription, "text"):
        value = getattr(transcription, "text")
        if isinstance(value, str):
            text = value.strip()
    if text is None and isinstance(transcription, dict):
        value = transcription.get("text")
        if isinstance(value, str):
            text = value.strip()
    if text is None:
        text = str(transcription).strip()
    match = re.match(
        r"\s*language\s+[\w.-]+(?:\s*<audio_language>[^<]*</audio_language>)?"
        r"\s*<asr_text>\s*(?P<text>.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group("text").strip() if match else text


def _create_transcription(
    client: Any,
    file_path: str,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
) -> Any:
    with open(file_path, "rb") as audio_file:
        kwargs: Dict[str, Any] = {
            "model": model,
            "file": audio_file,
            "response_format": "text" if model == "whisper-1" else "json",
        }
        if language:
            if model == "gpt-transcribe":
                kwargs["extra_body"] = {"languages": [language]}
            else:
                kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        return client.audio.transcriptions.create(**kwargs)


def transcribe_audio(file_path: str, settings: Settings) -> Dict[str, Any]:
    """Transcribe one validated local audio file through OpenAI. (Hermes)"""
    error = _validate_audio_file(file_path)
    if error:
        return error
    api_key = _resolve_openai_audio_api_key()
    if not api_key:
        return {
            "success": False,
            "transcript": "",
            "error": "VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY is not set",
        }

    model = settings.text("stt.openai.model", DEFAULT_STT_MODEL)
    language = _language(settings)
    prompt = _prompt(settings)
    trimmed_path = _trim_silence_for_cloud_stt(file_path, settings)
    upload_path = trimmed_path or file_path
    trim_dir = str(Path(trimmed_path).parent) if trimmed_path else ""

    try:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            BadRequestError,
            OpenAI,
        )

        client = OpenAI(
            api_key=api_key,
            base_url=OPENAI_BASE_URL,
            timeout=30,
            max_retries=0,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="pilotage-stt-") as work_dir:
                try:
                    transcription = _create_transcription(
                        client, upload_path, model, language, prompt
                    )
                except BadRequestError as exc:
                    message = str(exc).lower()
                    if not any(
                        marker in message
                        for marker in ("unsupported", "corrupted", "invalid file")
                    ):
                        raise
                    converted, transcode_error = _transcode_audio_for_stt(
                        upload_path, work_dir
                    )
                    if transcode_error:
                        return {
                            "success": False,
                            "transcript": "",
                            "error": transcode_error,
                        }
                    logger.info(
                        "Retrying OpenAI STT after transcoding %s to m4a",
                        Path(file_path).name,
                    )
                    transcription = _create_transcription(
                        client, str(converted), model, language, prompt
                    )
            text = _extract_transcript_text(transcription)
            logger.info(
                "Transcribed %s via OpenAI (%s, %d chars)",
                Path(file_path).name,
                model,
                len(text),
            )
            return {"success": True, "transcript": text, "provider": "openai"}
        finally:
            client.close()
    except PermissionError:
        return {
            "success": False,
            "transcript": "",
            "error": f"Permission denied: {file_path}",
        }
    except APIConnectionError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": f"Connection error: {exc}",
        }
    except APITimeoutError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": f"Request timeout: {exc}",
        }
    except APIError as exc:
        return {
            "success": False,
            "transcript": "",
            "error": f"API error: {exc}",
        }
    except Exception as exc:  # noqa: BLE001 - one voice note cannot stop the channel
        logger.error("OpenAI transcription failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "transcript": "",
            "error": f"Transcription failed: {exc}",
        }
    finally:
        if trim_dir:
            shutil.rmtree(trim_dir, ignore_errors=True)


async def enrich_message(
    user_text: str,
    attachments: Sequence[Attachment],
    settings: Settings,
) -> tuple[str, list[str]]:
    """Prepend Hermes-formatted transcripts for inbound WhatsApp voice notes."""
    seen: set[Path] = set()
    voice_paths: list[Path] = []
    for attachment in attachments:
        if not attachment.is_voice_message:
            continue
        path = attachment.path.resolve()
        if path not in seen:
            seen.add(path)
            voice_paths.append(path)
    if not voice_paths:
        return user_text, []

    parts: list[str] = []
    transcripts: list[str] = []
    if not settings.flag("stt.enabled", True):
        parts.extend(
            f"[The user sent a voice message: {path}]" for path in voice_paths
        )
    else:
        for path in voice_paths:
            result = await asyncio.to_thread(transcribe_audio, str(path), settings)
            if result.get("success"):
                transcript = str(result.get("transcript") or "").strip()
                if transcript:
                    transcripts.append(transcript)
                    parts.append(f'"{transcript}"')
                else:
                    parts.append(
                        "[The user sent a voice message but it came through empty or "
                        "inaudible — speech-to-text returned no words. Do not guess at "
                        "the content; ask the user to resend or type it out.]"
                    )
            else:
                logger.info(
                    "Voice transcription failed for %s: %s",
                    path,
                    result.get("error", "unknown error"),
                )
                parts.append(
                    "[voice message could not be transcribed automatically; "
                    f"the audio is available at: {path}]"
                )

    prefix = "\n\n".join(parts)
    if prefix and user_text:
        return f"{prefix}\n\n{user_text}", transcripts
    return prefix or user_text, transcripts


__all__ = [
    "enrich_message",
    "transcribe_audio",
    "transcript_echo_enabled",
    "validate_settings",
]
