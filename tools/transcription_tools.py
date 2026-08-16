#!/usr/bin/env python3
"""
Transcription Tools Module

Provides speech-to-text transcription through the OpenAI transcription API
(requires ``VOICE_TOOLS_OPENAI_KEY`` or ``OPENAI_API_KEY``). Additional
backends can be added by plugins via the STT provider registry.

Used by the messaging gateway to automatically transcribe voice messages
sent by users on Telegram, Discord, WhatsApp, Slack, and Signal.

Supported input formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, aac

Usage::

    from tools.transcription_tools import transcribe_audio

    result = transcribe_audio("/path/to/audio.ogg")
    if result["success"]:
        print(result["transcript"])
"""

import logging
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from pilotage_cli._subprocess_compat import windows_hide_flags
from utils import is_truthy_value
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_openai_audio_api_key,
)

logger = logging.getLogger(__name__)

def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``pilotage_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so STT does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from pilotage_cli.config import get_env_value as _get_env_value
    except ImportError:
        return os.getenv(name, default)
    value = _get_env_value(name)
    return default if value is None else value


def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve an STT provider API key via the shared voice-key resolver.

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

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _safe_find_spec(module_name: str) -> bool:
    try:
        return _ilu.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in globals() or module_name in os.sys.modules


_HAS_OPENAI = _safe_find_spec("openai")
_HAS_PILK = _safe_find_spec("pilk")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER = "openai"
DEFAULT_STT_MODEL = os.getenv("STT_OPENAI_MODEL", "whisper-1")
COMMON_LOCAL_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

OPENAI_BASE_URL = os.getenv("STT_OPENAI_BASE_URL", "https://api.openai.com/v1")

SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".opus", ".aac", ".flac", ".caf"}
LOCAL_NATIVE_AUDIO_FORMATS = {".wav", ".aiff", ".aif"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

# Known model set for auto-correction
OPENAI_MODELS = {"whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------



def _load_stt_config() -> dict:
    """Load the ``stt`` section from user config, falling back to defaults."""
    try:
        from pilotage_cli.config import load_config
        return load_config().get("stt") or {}
    except Exception:
        return {}


def is_stt_enabled(stt_config: Optional[dict] = None) -> bool:
    """Return whether STT is enabled in config."""
    if stt_config is None:
        stt_config = _load_stt_config()
    enabled = stt_config.get("enabled", True)
    return is_truthy_value(enabled, default=True)


def _resolve_stt_language(
    provider_key: str,
    stt_config: Optional[Dict[str, Any]] = None,
    *,
    extra_keys: tuple = (),
) -> Optional[str]:
    """Resolve the language hint for an STT provider (class-level, all providers).

    Resolution order (first non-empty wins):
      1. ``stt.<provider>.language`` (plus any *extra_keys* aliases, e.g.
         ElevenLabs' historical ``language_code``)
      2. ``stt.language``           — global default for every provider
      3. ``None``                   — let the provider auto-detect

    Returns a stripped ISO-639-1-ish code or None. Never returns "".
    """
    if stt_config is None:
        stt_config = _load_stt_config()
    provider_cfg = _get_stt_section(stt_config, provider_key)
    candidates = [provider_cfg.get("language")]
    for key in extra_keys:
        candidates.append(provider_cfg.get(key))
    if isinstance(stt_config, dict):
        candidates.append(stt_config.get("language"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use config credentials, env credentials, or the managed gateway."""
    try:
        _resolve_openai_audio_client_config()
        return True
    except ValueError:
        return False


def _find_binary(binary_name: str) -> Optional[str]:
    """Find a local binary, checking common Homebrew/local prefixes as well as PATH."""
    for directory in COMMON_LOCAL_BIN_DIRS:
        candidate = Path(directory) / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary_name)


def _find_ffmpeg_binary() -> Optional[str]:
    return _find_binary("ffmpeg")


# Shared encode profile for every STT-bound m4a we produce (transcode and
# silence-trim): 16 kHz mono 32 kbps AAC, faststart. One owner — codec or
# bitrate changes must not drift between the two paths.
_STT_M4A_ENCODE_ARGS = (
    "-vn", "-ac", "1", "-ar", "16000",
    "-c:a", "aac", "-b:a", "32k", "-movflags", "+faststart",
)


def _run_ffmpeg_stt_encode(
    ffmpeg: str, input_path: str, output_path: str, *, audio_filter: Optional[str] = None
) -> None:
    """Run the shared STT m4a encode, optionally with an ``-af`` filter.

    Raises on failure (CalledProcessError / TimeoutExpired) — callers own
    the error semantics (transcode reports, trim swallows).
    """
    command = [ffmpeg, "-y", "-i", input_path]
    if audio_filter:
        command += ["-af", audio_filter]
    command += [*_STT_M4A_ENCODE_ARGS, output_path]
    subprocess.run(
        command, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
        stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
    )


def _transcode_audio_for_stt(file_path: str, work_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Transcode ``file_path`` to a compact, broadly-accepted .m4a for STT upload.

    Newer OpenAI transcription models (``gpt-4o-transcribe``,
    ``gpt-4o-mini-transcribe``) reject some containers the legacy ``whisper-1``
    endpoint accepted -- notably the Ogg/Opus voice notes messaging apps send --
    and gateway downloads occasionally arrive with a misleading extension.
    Normalizing to 16 kHz mono AAC/m4a produces a small file the endpoints
    accept. Returns ``(converted_path, None)`` on success or ``(None, error)``.
    """
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        return None, "audio needs transcoding for the STT API, but ffmpeg was not found"
    converted_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-stt.m4a")
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, converted_path)
        return converted_path, None
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        logger.error("ffmpeg STT transcode failed for %s: %s", file_path, details)
        return None, f"failed to transcode audio for the STT API: {details}"
    except Exception as exc:  # noqa: BLE001 - transcode is best-effort
        logger.error("unexpected STT transcode failure for %s: %s", file_path, exc, exc_info=True)
        return None, f"failed to transcode audio for the STT API: {exc}"


# Names of the STT providers with native handlers in this module.
# Kept in sync with ``agent.transcription_registry._BUILTIN_NAMES`` —
# a regression test fails if they drift. The plugin hook from
#-style follow-up rejects plugins registering under any
# of these names; the dispatcher in ``transcribe_audio`` short-circuits
# them defensively as well.
BUILTIN_STT_PROVIDERS = frozenset({"openai"})


# ---------------------------------------------------------------------------
# Command-provider registry (``stt.providers.<name>: type: command``)
# ---------------------------------------------------------------------------
#
# Mirrors the TTS command-provider registry shipped in — same
# placeholder grammar, same shell-quote-aware rendering, same process-tree
# termination on timeout. Lets any whisper CLI / ASR CLI / curl pipeline
# become an STT backend with zero Python.
#
# Resolution order:
#   1. Built-in (``local``, ``local_command``, ``groq``, ``openai``,
#      ``mistral``, ``xai``)              → native handler. **Always wins.**
#   2. ``stt.providers.<name>: type: command``  → command-provider runner.
#   3. Plugin-registered TranscriptionProvider  → plugin dispatch.
#   4. No match                                 → "No STT provider available".
#
# The single-env-var ``PILOTAGE_LOCAL_STT_COMMAND`` escape hatch is preserved
# untouched via the built-in ``local_command`` path. Use the command-provider
# registry when you want MULTIPLE shell-driven STT engines, or you want a
# named provider you can pick via ``stt.provider`` in config.yaml.
DEFAULT_COMMAND_STT_TIMEOUT_SECONDS = 300
DEFAULT_COMMAND_STT_LANGUAGE = "en"
DEFAULT_COMMAND_STT_OUTPUT_FORMAT = "txt"
COMMAND_STT_OUTPUT_FORMATS = frozenset({"txt", "json", "srt", "vtt"})


def _get_stt_section(stt_config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return an stt sub-section if it's a dict, else an empty dict."""
    if not isinstance(stt_config, dict):
        return {}
    section = stt_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_provider(stt_config: dict) -> str:
    """Determine which STT provider to use.

    OpenAI is the only built-in backend. An explicitly configured name that
    is not ``openai`` falls through so a plugin-registered provider can
    claim it downstream.
    """
    if not is_stt_enabled(stt_config):
        return "none"

    explicit = "provider" in stt_config
    provider = stt_config.get("provider", DEFAULT_PROVIDER)

    if explicit:
        if provider == "openai":
            if _HAS_OPENAI and _has_openai_audio_backend():
                return "openai"
            logger.warning(
                "STT provider 'openai' configured but no API key available"
            )
            return "none"
        return provider  # Unknown — plugin dispatch or failure downstream

    if _HAS_OPENAI and _has_openai_audio_backend():
        return "openai"
    return "none"


def _unregistered_stt_provider_error(provider: str) -> Dict[str, Any]:
    key = str(provider or "").strip()
    return {
        "success": False,
        "transcript": "",
        "provider": key,
        "error_type": "provider_not_registered",
        "error": (
            f"stt.provider='{key}' is set but no built-in, command, or plugin "
            "provider registered that name. Run `pilotage plugins list` to see "
            "installed STT plugins, or configure a command provider under "
            f"`stt.providers.{key}.command`."
        ),
    }


# ---------------------------------------------------------------------------
# Plugin provider dispatch (issue follow-up to — STT pluggability)
# ---------------------------------------------------------------------------


def _dispatch_to_plugin_provider(
    file_path: str,
    provider: str,
    stt_config: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Route the call to a plugin-registered transcription provider, or
    return None.

    Returns the transcribe-response dict on dispatch, or ``None`` when no
    plugin claimed the provider name.

    Resolution invariants enforced here:

    1. Built-in provider names short-circuit — never reach the plugin
       registry. The caller (``transcribe_audio``) handles ``local``,
       ``groq``, ``openai``, etc. via its existing elif chain; this
       function defensively rejects those names so a plugin can't be
       silently dispatched under a built-in name even if it somehow
       slipped past the registry's built-in shadow guard.
    2. Same-name command-type provider declared under
       ``stt.providers.<name>: type: command`` wins over a plugin. The
       caller short-circuits to the command runner before reaching us,
       but we re-verify here so a refactor of the caller can't silently
       break the invariant (matches TTS precedence rule).
    3. Plugin dispatch fires only when ``provider`` matches a
       registered :class:`TranscriptionProvider` whose ``name`` equals
       the configured value. Unknown names with no plugin registered
       return None (caller surfaces the configured-provider error when
       the name came from ``stt.provider``).
    4. Availability gating: when the matched plugin reports
       ``is_available() == False`` (missing API key, missing optional
       SDK, etc.) this returns an error envelope identifying the
       plugin as unavailable — **not** ``None`` — because the user
       explicitly opted into this plugin via ``stt.provider`` and the
       generic fallthrough message would be misleading.

    Provider exceptions are caught and converted into the standard
    error envelope (matches the legacy built-in error shapes — the
    gateway/CLI caller already expects ``{success: False, error:
    "...", transcript: ""}`` on failure).
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_STT_PROVIDERS or key == "none":
        return None
    try:
        from agent.transcription_registry import get_provider
        from pilotage_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before a
            # bundled backend was patched in or before config changed.
            # Retry once with a forced refresh before surfacing fall-
            # through. Mirrors the image_gen / browser dispatcher
            # recovery pattern.
            _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("STT plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # Availability gate: when a plugin reports it's not configured
    # (missing API key, missing optional SDK, etc.) surface a clean
    # error envelope **instead of** falling through to the generic
    # "No STT provider" message. The user explicitly set
    # ``stt.provider: <plugin>`` in config — surfacing the plugin's
    # own availability failure is more actionable than the generic
    # auto-detect-failure error, and avoids routing the call into a
    # plugin that's about to crash messily.
    #
    # ``is_available()`` MUST NOT raise per the ABC contract; defend
    # anyway so a buggy plugin can't break dispatch for everyone.
    try:
        available = plugin_provider.is_available()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' is_available() raised: %s — "
            "treating as unavailable", key, exc, exc_info=True,
        )
        available = False
    if not available:
        logger.info(
            "STT plugin provider '%s' reports not available; returning "
            "unavailability envelope.", key,
        )
        return {
            "success": False,
            "transcript": "",
            "error": (
                f"STT plugin '{key}' is not available — check that its "
                "required credentials / dependencies are configured."
            ),
            "provider": key,
        }

    logger.info("Transcribing with plugin STT provider '%s'...", key)
    # Plugin providers receive the transcription prompt via the ABC's
    # existing ``**extra`` kwargs — no signature change needed. The key is
    # only sent when a prompt is actually set so providers that predate it
    # see byte-identical calls on the no-prompt path.
    extra_kwargs: Dict[str, Any] = {}
    if prompt is not None:
        extra_kwargs["prompt"] = prompt
    try:
        result = plugin_provider.transcribe(
            file_path,
            model=model,
            language=language,
            **extra_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STT plugin provider '%s' raised: %s", key, exc, exc_info=True,
        )
        return {
            "success": False,
            "transcript": "",
            "error": f"STT plugin '{key}' raised: {exc}",
            "provider": key,
        }

    # Defensive: plugins should return a dict matching the contract. If
    # they don't, surface a clear error envelope rather than leaking a
    # weird object back to the gateway.
    if not isinstance(result, dict):
        return {
            "success": False,
            "transcript": "",
            "error": f"STT plugin '{key}' returned a non-dict result",
            "provider": key,
        }
    # Stamp provider if the plugin forgot to.
    result.setdefault("provider", key)
    return result


# ---------------------------------------------------------------------------
# pre_transcription plugin hook ( — STT prompt/vocab threading)
# ---------------------------------------------------------------------------


# Fields a pre_transcription hook may mutate. ``file_path`` is deliberately
# absent — it is read-only; attempts to change it are logged and dropped.
_PRE_TRANSCRIPTION_MUTABLE_FIELDS = ("prompt", "language", "model")

# Whisper-family models silently use only the final ~224 tokens of the
# prompt/initial_prompt; longer values waste upload bytes and can trip
# stricter OpenAI-compatible servers. Enforce the cap client-side for the
# whisper-family backends: truncate with a warning, never error.
# Approximation: ~4 characters per token (no tokenizer dependency).
_WHISPER_PROMPT_TOKEN_CAP = 224
_PROMPT_CHARS_PER_TOKEN = 4
# Providers whose prompt parameter feeds a whisper-family model.
_WHISPER_PROMPT_CAPPED_PROVIDERS = frozenset({"openai"})


def _enforce_prompt_length_limit(
    prompt: Optional[str], provider: str
) -> Optional[str]:
    """Truncate *prompt* to the provider's known token cap (fail-open).

    Only whisper-family backends have a documented ~224-token prompt window;
    other providers (mistral, plugin providers) own their own validation.
    Truncation keeps the TAIL of the prompt because whisper conditions on
    the final context window — the most recently appended hints survive.
    """
    if not prompt or provider not in _WHISPER_PROMPT_CAPPED_PROVIDERS:
        return prompt
    max_chars = _WHISPER_PROMPT_TOKEN_CAP * _PROMPT_CHARS_PER_TOKEN
    if len(prompt) <= max_chars:
        return prompt
    logger.warning(
        "Transcription prompt is ~%d tokens; whisper-family provider '%s' "
        "only uses the final ~%d — truncating to the last %d characters.",
        len(prompt) // _PROMPT_CHARS_PER_TOKEN,
        provider,
        _WHISPER_PROMPT_TOKEN_CAP,
        max_chars,
    )
    return prompt[-max_chars:]


def _apply_pre_transcription_hook(
    *,
    file_path: str,
    provider: str,
    model: Optional[str],
    language: Optional[str],
    prompt: Optional[str],
    source: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fire the ``pre_transcription`` plugin hook and merge its results.

    Mirrors the ``transform_*`` hook mechanics (``transform_tool_result``):
    gated on ``has_hook`` so the no-hook dispatch path never builds hook
    kwargs, and fail-open — any hook-plumbing error leaves the dispatch
    untouched. ``invoke_hook`` returns results in registration order, and
    plugin discovery scans plugin directories in sorted order, so multiple
    plugins' hints compose deterministically (sorted by plugin id, then
    each plugin's own registration order). Each dict result is applied
    field-by-field on top of the previous ones, so the last hook to write
    a field wins (last-writer-wins per field).

    Model values are accepted as-is: the dispatcher has no catalog-level
    validation today, so a hook-set model flows through the exact same
    per-backend normalization a caller-supplied model would, and otherwise
    errors at the backend as it would today.

    Returns ``(model, language_override, prompt)``. ``language_override``
    is ``None`` unless a hook explicitly set ``language`` — backends keep
    their existing config/env language resolution when no hook overrides
    it.
    """
    try:
        from pilotage_cli.plugins import has_hook, invoke_hook

        # No-hook short-circuit: keep the no-plugin dispatch path
        # byte-identical (no kwargs built, no invoke_hook call).
        if not has_hook("pre_transcription"):
            return model, None, prompt

        hook_results = invoke_hook(
            "pre_transcription",
            file_path=file_path,
            provider=provider,
            model=model,
            language=language,
            prompt=prompt,
            source=source,
        )
        overrides: Dict[str, Any] = {}
        for hook_result in hook_results:
            if not isinstance(hook_result, dict):
                continue
            for key, value in hook_result.items():
                if key == "file_path":
                    # file_path is read-only for hooks — log and drop.
                    logger.warning(
                        "pre_transcription hook attempted to change "
                        "file_path (read-only) — ignoring the attempt."
                    )
                    continue
                if key not in _PRE_TRANSCRIPTION_MUTABLE_FIELDS:
                    logger.debug(
                        "pre_transcription hook returned unsupported field "
                        "%r — ignoring.", key,
                    )
                    continue
                if not isinstance(value, str):
                    logger.debug(
                        "pre_transcription hook returned non-string value "
                        "%r for field %r — ignoring.", value, key,
                    )
                    continue
                overrides[key] = value

        if "model" in overrides:
            model = overrides["model"]
        if "prompt" in overrides:
            # Hook results win over the static ``stt.prompt`` config value —
            # config is the base, hooks mutate on top. An empty string
            # clears the config prompt.
            prompt = overrides["prompt"] or None
        return model, overrides.get("language") or None, prompt
    except Exception as _hook_err:  # noqa: BLE001 — hook plumbing is fail-open
        logger.debug("pre_transcription hook error: %s", _hook_err)
        return model, None, prompt


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_audio_file_size(audio_path: Path) -> Optional[Dict[str, Any]]:
    """Return an error when *audio_path* exceeds the remote upload cap."""
    try:
        file_size = audio_path.stat().st_size
    except OSError as e:
        return {"success": False, "transcript": "", "error": f"Failed to access file: {e}"}
    if file_size > MAX_FILE_SIZE:
        return {
            "success": False,
            "transcript": "",
            "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE / (1024*1024):.0f}MB)",
        }
    return None


def _validate_audio_source_file(
    file_path: str,
    *,
    enforce_size_limit: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate source path safety (and optionally size) before any decoder runs."""
    audio_path = Path(file_path)

    if os.path.islink(audio_path):
        return {"success": False, "transcript": "", "error": f"Path is a symbolic link: {file_path}"}
    if not audio_path.exists():
        return {"success": False, "transcript": "", "error": f"Audio file not found: {file_path}"}
    if not audio_path.is_file():
        return {"success": False, "transcript": "", "error": f"Path is not a file: {file_path}"}
    if enforce_size_limit:
        return _validate_audio_file_size(audio_path)
    try:
        audio_path.stat()
    except OSError as e:
        return {"success": False, "transcript": "", "error": f"Failed to access file: {e}"}
    return None


def _validate_audio_file(
    file_path: str,
    *,
    enforce_size_limit: bool = True,
) -> Optional[Dict[str, Any]]:
    """Validate a supported, decoder-safe audio file."""
    source_error = _validate_audio_source_file(
        file_path, enforce_size_limit=enforce_size_limit
    )
    if source_error:
        return source_error

    audio_path = Path(file_path)
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        return {
            "success": False,
            "transcript": "",
            "error": f"Unsupported format: {audio_path.suffix}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}",
        }
    return None


def _prepare_audio_for_transcription(
    file_path: str,
) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Convert a decoder-safe .silk source to a temporary supported WAV file."""
    audio_path = Path(file_path)
    if audio_path.suffix.lower() != ".silk":
        return file_path, None, None
    if not _HAS_PILK:
        # pilk is a tiny silk-v3 codec binding — lazy-install it on first
        # .silk voice note instead of bloating the base install.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("stt.silk", prompt=False)
        except Exception:
            pass
        if not _safe_find_spec("pilk"):
            return None, None, {
                "success": False,
                "transcript": "",
                "error": "Unsupported format: .silk. Install the optional 'pilk' dependency to enable WeChat voice transcription.",
            }

    temp_dir = tempfile.mkdtemp(prefix="pilotage-silk-")
    converted_path = os.path.join(temp_dir, f"{audio_path.stem}.wav")
    try:
        import pilk

        pilk.silk_to_wav(file_path, converted_path)
        if not Path(converted_path).is_file() or Path(converted_path).stat().st_size == 0:
            raise RuntimeError("pilk did not produce a readable WAV file")
        return converted_path, temp_dir, None
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Failed to convert .silk audio %s: %s", file_path, exc, exc_info=True)
        return None, None, {
            "success": False,
            "transcript": "",
            "error": f"Failed to convert .silk audio for transcription: {exc}",
        }

# ---------------------------------------------------------------------------
# Provider: openai (Whisper API)
# ---------------------------------------------------------------------------


def _transcribe_openai(
    file_path: str,
    model_name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_label: str = "openai",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe via the OpenAI ``audio.transcriptions.create`` SDK shape.

    Also serves as the shared backend for every OpenAI-compatible STT
    endpoint (DeepInfra etc.) — callers pass an explicit ``api_key`` /
    ``base_url`` to skip the OpenAI-only auth chain, and a
    ``provider_label`` so the response carries the right ``provider``
    name.
    """
    if api_key is None:
        try:
            api_key, fallback_base = _resolve_openai_audio_client_config()
        except ValueError as exc:
            return {"success": False, "transcript": "", "error": str(exc)}
        base_url = base_url or fallback_base

    # Language: hook override > stt.<provider>.language > stt.language >
    # env > auto-detect. Explicit language hint improves accuracy for
    # non-English languages.
    language = language or _resolve_stt_language(provider_label)

    if not _HAS_OPENAI:
        return {"success": False, "transcript": "", "error": "openai package not installed"}

    try:
        from openai import (
            OpenAI,
            APIError,
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
        )
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)

        def _create_transcription(path: str):
            with open(path, "rb") as audio_file:
                create_kwargs = {
                    "model": model_name,
                    "file": audio_file,
                    "response_format": "text" if model_name == "whisper-1" else "json",
                }
                if language:
                    if model_name == "gpt-transcribe":
                        # gpt-transcribe replaces the singular ``language``
                        # field with a ``languages`` list; the API rejects
                        # requests that send the legacy field.
                        create_kwargs["extra_body"] = {"languages": [language]}
                    else:
                        create_kwargs["language"] = language
                    logger.debug("Using language hint '%s' for OpenAI STT", language)
                if prompt:
                    # Only send the prompt when set so the no-hook, no-config
                    # request stays byte-identical to today's.
                    create_kwargs["prompt"] = prompt
                return client.audio.transcriptions.create(**create_kwargs)

        try:
            with tempfile.TemporaryDirectory(prefix="pilotage-stt-") as work_dir:
                try:
                    transcription = _create_transcription(file_path)
                except BadRequestError as exc:
                    message = str(exc).lower()
                    if not any(k in message for k in ("unsupported", "corrupted", "invalid file")):
                        raise
                    # Newer models (e.g. gpt-4o-transcribe) reject some containers
                    # whisper-1 accepted (notably Ogg/Opus voice notes). Transcode
                    # to a compact .m4a and retry once.
                    converted_path, transcode_error = _transcode_audio_for_stt(file_path, work_dir)
                    if transcode_error:
                        return {"success": False, "transcript": "", "error": transcode_error}
                    logger.info(
                        "Retrying %s STT after transcoding %s to m4a (API rejected the original container)",
                        provider_label, Path(file_path).name,
                    )
                    transcription = _create_transcription(converted_path)

            transcript_text = _extract_transcript_text(transcription)
            logger.info(
                "Transcribed %s via %s (%s, %d chars)",
                Path(file_path).name, provider_label, model_name, len(transcript_text),
            )

            return {"success": True, "transcript": transcript_text, "provider": provider_label}
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except APIConnectionError as e:
        return {"success": False, "transcript": "", "error": f"Connection error: {e}"}
    except APITimeoutError as e:
        return {"success": False, "transcript": "", "error": f"Request timeout: {e}"}
    except APIError as e:
        return {"success": False, "transcript": "", "error": f"API error: {e}"}
    except Exception as e:
        logger.error("%s transcription failed: %s", provider_label, e, exc_info=True)
        return {"success": False, "transcript": "", "error": f"Transcription failed: {e}"}

# ---------------------------------------------------------------------------
# Cloud pre-upload silence trim
# ---------------------------------------------------------------------------
#
# Cloud endpoints transcribe the raw upload, so every second of silence is
# paid for twice — once in upload time and once in per-audio-minute
# billing — and cloud Whisper hallucinates junk tokens on silent
# stretches.
#
# Before uploading to a built-in cloud provider we collapse long pauses with
# ffmpeg's silenceremove filter, keeping ``stt.cloud_trim_keep_ms`` of every
# pause so word boundaries and natural pacing survive. The trim is purely
# best-effort — ANY of these falls back to uploading the original untouched:
#   - ``stt.cloud_trim_silence: false``
#   - ffmpeg or ffprobe not installed
#   - the trim command fails or times out
#   - the trimmed result is suspiciously empty (mostly-silence clip — the
#     provider, not a client-side heuristic, decides whether it has speech)
#   - the trim saves less than ~10% (re-encoding for nothing)
#
# Command-type and plugin providers are deliberately NOT trimmed: they may
# wrap local CLIs that want the original bytes (and may run their own VAD).

_CLOUD_TRIM_THRESHOLD_DB_DEFAULT = -40  # audio below this level counts as silence
_CLOUD_TRIM_KEEP_MS_DEFAULT = 300  # how much of each pause survives the trim
_CLOUD_TRIM_MIN_SAVING = 0.10  # use the trimmed file only when >=10% shorter
_CLOUD_TRIM_MIN_RESULT_SECONDS = 0.3  # all-silence guard floor: never upload ~empty audio
# Below this duration the trim can't pay for itself: a >=10% saving on a short
# clip is ~a second of audio, several providers bill a per-request minimum
# anyway (Groq: 10s), and the encode would sit on the synchronous voice-note
# response path. Skip the whole pipeline.
_CLOUD_TRIM_MIN_INPUT_SECONDS = 12.0

# Built-in providers that upload audio to a remote API.
CLOUD_STT_PROVIDERS = frozenset(BUILTIN_STT_PROVIDERS)


def _convert_caf_to_wav(file_path: str) -> Optional[str]:
    """Convert CAF to WAV using ffmpeg or afconvert (macOS)."""
    audio_path = Path(file_path)
    wav_path = os.path.join(audio_path.parent, f"{audio_path.stem}.wav")
    ffmpeg = _find_ffmpeg_binary()
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-i", file_path, wav_path],
                check=True, capture_output=True, text=True,
                timeout=300, stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags())
            return wav_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("ffmpeg CAF to WAV failed for %s: %s", file_path, e)
    afconvert = shutil.which("afconvert")
    if afconvert:
        try:
            subprocess.run([afconvert, file_path, wav_path, "-d", "LEI16", "-f", "WAVE"],
                check=True, capture_output=True, text=True,
                timeout=300, stdin=subprocess.DEVNULL)
            return wav_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("afconvert CAF to WAV failed for %s: %s", file_path, e)
    return None


def _find_ffprobe_binary() -> Optional[str]:
    return _find_binary("ffprobe")


def _probe_audio_duration(file_path: str) -> Optional[float]:
    """Return the audio duration in seconds via ffprobe, or None.

    Canonical sync seconds-probe. ``gateway/run.py._probe_audio_duration``
    (async, returns a display string) and the Telegram adapter's
    ``_probe_voice_duration_seconds`` carry local variants of the same
    ffprobe invocation — keep the command shape in sync.
    """
    ffprobe = _find_ffprobe_binary()
    if not ffprobe:
        return None
    command = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
            stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
        )
        return float(result.stdout.strip())
    except Exception:  # noqa: BLE001 - probe is best-effort
        return None


def _cloud_trim_settings(stt_config: Dict[str, Any]) -> tuple[bool, int, int]:
    """Resolve (enabled, threshold_db, keep_ms) for the cloud silence trim."""
    cfg = stt_config if isinstance(stt_config, dict) else {}
    # is_truthy_value: the module's established config-boolean normalizer —
    # a YAML string "false" must disable, exactly like is_stt_enabled.
    enabled = is_truthy_value(cfg.get("cloud_trim_silence", True), default=True)
    try:
        threshold_db = int(cfg.get("cloud_trim_threshold_db", _CLOUD_TRIM_THRESHOLD_DB_DEFAULT))
    except (TypeError, ValueError):
        threshold_db = _CLOUD_TRIM_THRESHOLD_DB_DEFAULT
    try:
        keep_ms = int(cfg.get("cloud_trim_keep_ms", _CLOUD_TRIM_KEEP_MS_DEFAULT))
    except (TypeError, ValueError):
        keep_ms = _CLOUD_TRIM_KEEP_MS_DEFAULT
    return enabled, threshold_db, max(keep_ms, 0)


def _trim_silence_for_cloud_stt(
    file_path: str, stt_config: Dict[str, Any]
) -> Optional[str]:
    """Return a silence-trimmed copy of *file_path* for cloud upload, or None.

    ``None`` always means "upload the original file": the trim is disabled,
    the tools are missing, the clip is too short for a trim to pay for
    itself, the trim failed, the clip is mostly silence, or trimming would
    not save enough to justify the re-encode. On success the caller owns
    deleting the returned file's parent directory.
    """
    enabled, threshold_db, keep_ms = _cloud_trim_settings(stt_config)
    if not enabled:
        return None
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        logger.debug("Cloud STT silence trim skipped: ffmpeg not found")
        return None
    original_duration = _probe_audio_duration(file_path)
    if not original_duration or original_duration <= 0:
        logger.debug("Cloud STT silence trim skipped: could not probe %s", file_path)
        return None
    if original_duration < _CLOUD_TRIM_MIN_INPUT_SECONDS:
        # Short clip: savings can't matter (some providers bill a 10s
        # minimum per request anyway) — skip the encode entirely.
        logger.debug(
            "Cloud STT silence trim skipped for %s: %.1fs is below the %.0fs gate",
            Path(file_path).name, original_duration, _CLOUD_TRIM_MIN_INPUT_SECONDS,
        )
        return None

    keep_seconds = keep_ms / 1000.0
    # start_periods=1 strips leading silence; stop_periods=-1 collapses every
    # interior/trailing silence, keeping ``keep_seconds`` of each pause.
    filter_expr = (
        f"silenceremove="
        f"start_periods=1:start_threshold={threshold_db}dB:start_silence={keep_seconds}:"
        f"stop_periods=-1:stop_threshold={threshold_db}dB:stop_silence={keep_seconds}"
    )
    work_dir = tempfile.mkdtemp(prefix="pilotage-stt-trim-")
    trimmed_path = os.path.join(work_dir, f"{Path(file_path).stem or 'audio'}-trimmed.m4a")
    # Scale the all-silence guard with keep_ms: an output consisting solely
    # of kept pause must never be uploaded as "speech".
    min_result_seconds = max(_CLOUD_TRIM_MIN_RESULT_SECONDS, 2 * keep_seconds)
    keep_result = False
    try:
        _run_ffmpeg_stt_encode(ffmpeg, file_path, trimmed_path, audio_filter=filter_expr)
        trimmed_duration = _probe_audio_duration(trimmed_path)
        if not trimmed_duration or trimmed_duration < min_result_seconds:
            # Mostly/all silence. Deciding "no speech" belongs to the
            # provider, not a client-side dB heuristic — upload the original.
            logger.debug(
                "Cloud STT silence trim discarded for %s: trimmed result ~empty (%.2fs)",
                Path(file_path).name, trimmed_duration or 0.0,
            )
            return None
        if trimmed_duration > original_duration * (1 - _CLOUD_TRIM_MIN_SAVING):
            logger.debug(
                "Cloud STT silence trim discarded for %s: saves <%.0f%% (%.1fs -> %.1fs)",
                Path(file_path).name, _CLOUD_TRIM_MIN_SAVING * 100,
                original_duration, trimmed_duration,
            )
            return None
        logger.info(
            "Trimmed silence from %s before cloud STT upload (%.1fs -> %.1fs, -%d%%)",
            Path(file_path).name, original_duration, trimmed_duration,
            round((1 - trimmed_duration / original_duration) * 100),
        )
        keep_result = True
        return trimmed_path
    except Exception as exc:  # noqa: BLE001 - trim is best-effort
        logger.debug("Cloud STT silence trim failed for %s: %s", file_path, exc)
        return None
    finally:
        if not keep_result:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _transcribe_prepared_audio(
    file_path: str,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio file using the configured STT provider.

    Provider priority:
      1. User config (``stt.provider`` in config.yaml)
      2. Auto-detect: local > Groq > OpenAI > Mistral > xAI > ElevenLabs

    Args:
        file_path: Absolute path to the audio file to transcribe.
        model:     Override the model. If None, uses config or provider default.
        source:    Optional caller-surface label (e.g. ``"gateway"``,
                   ``"voice_mode"``) forwarded to the ``pre_transcription``
                   plugin hook for observability. Not used for dispatch.

    Returns:
        dict with keys:
          - "success" (bool): Whether transcription succeeded
          - "transcript" (str): The transcribed text (empty on failure)
          - "error" (str, optional): Error message if success is False
          - "provider" (str, optional): Which provider was used
    """
    # Refuse to feed a credential / secret store (auth.json, .env, OAuth
    # tokens, mcp-tokens/, ...) to an STT provider: an external provider would
    # ship its plaintext contents to a third-party API. Mirrors the local-input
    # read guard added to image-gen (587be5b5b) and xAI video-gen (104232979).
    from agent.file_safety import get_read_block_error
    blocked = get_read_block_error(file_path)
    if blocked:
        return {"success": False, "transcript": "", "error": blocked}

    # Apply common path validation before provider resolution so invalid files
    # cannot trigger provider setup or lazy installation. The remote-upload
    # size cap is enforced separately below, only for non-local providers.
    error = _validate_audio_file(file_path, enforce_size_limit=False)
    if error:
        return error

    # Load config and determine provider
    stt_config = _load_stt_config()
    if not is_stt_enabled(stt_config):
        return {
            "success": False,
            "transcript": "",
            "error": "STT is disabled in config.yaml (stt.enabled: false).",
        }

    provider = _get_provider(stt_config)
    error = _validate_audio_file_size(Path(file_path))
    if error:
        return error

    # Convert CAF (iMessage voice notes) to WAV for cloud STT providers.
    if Path(file_path).suffix.lower() == ".caf":
        converted = _convert_caf_to_wav(file_path)
        if converted:
            file_path = converted
        else:
            return {"success": False, "transcript": "",
                    "error": "CAF audio could not be converted to WAV."}

    # Pre-upload silence trim for built-in cloud providers: local whisper gets
    # Silero VAD, cloud endpoints get the raw file — collapse long pauses
    # client-side so silence isn't uploaded, billed, or hallucinated on.
    # Best-effort: any failure uploads the original untouched.
    trim_cleanup_dir: Optional[str] = None
    if provider in CLOUD_STT_PROVIDERS:
        trimmed = _trim_silence_for_cloud_stt(file_path, stt_config)
        if trimmed:
            file_path = trimmed
            trim_cleanup_dir = os.path.dirname(trimmed)

    try:
        return _dispatch_stt_provider(file_path, provider, stt_config, model, source)
    finally:
        if trim_cleanup_dir:
            shutil.rmtree(trim_cleanup_dir, ignore_errors=True)


def _dispatch_stt_provider(
    file_path: str,
    provider: str,
    stt_config: Dict[str, Any],
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Route *file_path* to the handler for *provider* (built-in > command > plugin)."""
    # Optional static transcription prompt (``stt.prompt`` in config.yaml):
    # vocabulary/context hints threaded to prompt-capable backends.
    # Ordering: config is the base; pre_transcription hook results mutate on
    # top, in registration order, so the last hook to set a field wins.
    prompt = stt_config.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = None

    # pre_transcription plugin hook — fires after provider resolution and
    # BEFORE any backend (built-in, command-type, or plugin-registered) is
    # invoked. Hooks may mutate prompt/language/model; file_path is
    # read-only. The helper short-circuits on has_hook() so the no-hook
    # dispatch path stays byte-identical. ``language`` stays None unless a
    # hook overrides it — backends keep their own config/env resolution.
    model, language, prompt = _apply_pre_transcription_hook(
        file_path=file_path,
        provider=provider,
        model=model,
        language=_get_stt_section(stt_config, provider).get("language"),
        prompt=prompt,
        source=source,
    )

    # Whisper-family prompt windows top out around 224 tokens — truncate
    # (keeping the tail) with a warning rather than erroring or letting a
    # strict server reject the request.
    prompt = _enforce_prompt_length_limit(prompt, provider)

    if provider == "openai":
        openai_cfg = stt_config.get("openai") or {}
        model_name = model or openai_cfg.get("model", DEFAULT_STT_MODEL)
        return _transcribe_openai(
            file_path, model_name, language=language, prompt=prompt,
        )

    # Plugin-registered STT backend. Fires only when ``provider`` is
    # neither a built-in nor ``"none"``. Returns None when no plugin is
    # registered for the configured name; explicit configured names get a
    # provider-specific error before the generic fallback below.
    #
    # Plugin-scoped config namespace mirrors the built-in pattern
    # (``stt.openai.model``): plugins read their per-provider config under
    # ``stt.<provider>`` and the dispatcher forwards ``language`` from there.
    # Top-level ``model`` argument overrides any config-set model.
    plugin_cfg = stt_config.get(provider, {}) if isinstance(stt_config.get(provider), dict) else {}
    plugin_language = language or _resolve_stt_language(provider, stt_config)
    plugin_model = model or plugin_cfg.get("model")
    plugin_result = _dispatch_to_plugin_provider(
        file_path,
        provider,
        stt_config,
        model=plugin_model,
        language=plugin_language,
        prompt=prompt,
    )
    if plugin_result is not None:
        return plugin_result

    provider_key = str(provider or "").strip().lower()
    if (
        "provider" in stt_config
        and provider_key
        and provider_key not in BUILTIN_STT_PROVIDERS
        and provider_key != "none"
    ):
        return _unregistered_stt_provider_error(provider_key)

    # No provider available
    return {
        "success": False,
        "transcript": "",
        "error": (
            "No STT provider available. Set VOICE_TOOLS_OPENAI_KEY or "
            "OPENAI_API_KEY for the OpenAI transcription API."
        ),
    }


def transcribe_audio(
    file_path: str,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Safely validate, preprocess supported inputs, and dispatch transcription.

    ``source`` is an optional caller-surface label (e.g. ``"gateway"``,
    ``"voice_mode"``) forwarded to the ``pre_transcription`` plugin hook for
    observability. Not used for dispatch.
    """
    # Refuse to feed a credential / secret store (auth.json, .env, OAuth
    # tokens, mcp-tokens/, ...) to an STT provider — before ANY validation or
    # preprocessing, so the refusal names the real reason rather than a
    # format error. Mirrors the image-gen / video-gen read guards.
    from agent.file_safety import get_read_block_error
    blocked = get_read_block_error(file_path)
    if blocked:
        return {"success": False, "transcript": "", "error": blocked}

    # Cap .silk sources before the decoder runs (decoder safety). For all
    # other inputs the remote-upload size cap is enforced in
    # _transcribe_prepared_audio.
    is_silk = Path(file_path).suffix.lower() == ".silk"
    source_error = _validate_audio_source_file(file_path, enforce_size_limit=is_silk)
    if source_error:
        return source_error

    prepared_path, cleanup_dir, prep_error = _prepare_audio_for_transcription(file_path)
    if prep_error:
        return prep_error
    if prepared_path is None:
        return {
            "success": False,
            "transcript": "",
            "error": "Audio preprocessing did not produce a file for transcription.",
        }

    try:
        prepared_error = _validate_audio_file(prepared_path, enforce_size_limit=False)
        if prepared_error:
            return prepared_error
        return _transcribe_prepared_audio(prepared_path, model, source)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def _is_local_or_private_url(url: str) -> bool:
    """True when *url* points at a loopback/RFC-1918/LAN-internal host.

    Used to decide whether an empty ``stt.openai.api_key`` is acceptable:
    local OpenAI-compatible STT servers (faster-whisper-server, speaches,
    vLLM whisper variants...) ignore the auth header, so users shouldn't
    have to write a sham ``api_key: not-needed`` in config.yaml.
    """
    try:
        from urllib.parse import urlparse
        import ipaddress

        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        if host == "localhost" or host.endswith((".local", ".lan", ".internal")):
            return True
        try:
            return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False
    except Exception:
        return False


def _resolve_openai_audio_client_config() -> tuple[str, str]:
    """Return direct OpenAI audio config or a managed gateway fallback."""
    stt_config = _load_stt_config()
    openai_cfg = stt_config.get("openai") or {}
    cfg_api_key = openai_cfg.get("api_key", "")
    cfg_base_url = openai_cfg.get("base_url", "")
    if cfg_api_key:
        return cfg_api_key, (cfg_base_url or OPENAI_BASE_URL)

    # A local OpenAI-compatible server needs no key — send a placeholder so
    # the SDK doesn't refuse to construct a client (, credit @nnnet).
    if cfg_base_url and _is_local_or_private_url(cfg_base_url):
        return "not-needed", cfg_base_url

    direct_api_key = resolve_openai_audio_api_key()
    if direct_api_key:
        return direct_api_key, OPENAI_BASE_URL

    managed_gateway = resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = "Neither stt.openai.api_key in config nor VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        if managed_nous_tools_enabled():
            message += (
                ". "
                + nous_tool_gateway_unavailable_message(
                    "managed OpenAI audio for transcription",
                )
            )
        raise ValueError(message)

    return managed_gateway.nous_user_token, urljoin(
        f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"
    )


def _extract_transcript_text(transcription: Any) -> str:
    """Normalize text and JSON transcription responses to a plain string."""
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
        r"\s*language\s+[\w.-]+(?:\s*<audio_language>[^<]*</audio_language>)?\s*<asr_text>\s*(?P<text>.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group("text").strip()

    return text
