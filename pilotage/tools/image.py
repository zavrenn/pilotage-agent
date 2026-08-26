"""Image generation through ChatGPT/Codex OAuth, adapted from Hermes.

Genesis needs the one provider and tier its production profiles use:
``openai-codex`` with ``gpt-image-2-high``.  Hermes' working request shape,
SSE parser, image-input validation, bounded error reporting, and result
contract are kept.  Its provider registry and unrelated image backends are
not.

The blocking image request runs off the asyncio loop, and at most four run at
once — Hermes' conservative default.  Generated files are saved inside a
declared outbound-media root so Genesis' existing ``MEDIA:`` confinement can
deliver them without widening that boundary.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import json
import logging
import struct
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from ..codex import auth
from ..codex.client import cloudflare_headers
from ..settings import ConfigError, Settings
from .file_safety import get_read_block_error
from .path_security import validate_within_dir
from .registry import Tool, ToolContext, tool_error

logger = logging.getLogger(__name__)

PROVIDER = "openai-codex"
API_MODEL = "gpt-image-2"
DEFAULT_MODEL = "gpt-image-2-high"
DEFAULT_ASPECT_RATIO = "landscape"
VALID_ASPECT_RATIOS = ("landscape", "square", "portrait")

_MODELS: Dict[str, Dict[str, str]] = {
    "gpt-image-2-high": {"quality": "high"},
}
_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Hermes' Codex-hosted image request.  The chat model hosts the tool call;
# gpt-image-2 performs the actual image work.
_CODEX_CHAT_MODEL = "gpt-5.5"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation and image editing "
    "requests by using the image_generation tool when provided."
)
_REQUEST_TIMEOUT_SECONDS = 300.0
_MAX_REFERENCE_IMAGES = 16
_MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_ERROR_BODY_CHARS = 500
_MAX_PARALLEL_REQUESTS = 4
_IMAGE_SLOTS = threading.BoundedSemaphore(_MAX_PARALLEL_REQUESTS)
_ACCEPTED_INPUT_MIME = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)

# Designated Hermes behavior: progressive frames are previews, never final
# deliverables.  Ask for none, track them separately if the backend still
# sends one, and retry once for a real final result.
_PARTIAL_IMAGES_REQUESTED = 0
_NONFINAL_RETRIES = 1


class ImageAPIError(RuntimeError):
    """A Codex image request reached the server and was refused."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def validate_image_settings(settings: Settings) -> str:
    """Validate the fixed provider and selected Hermes tier at startup."""
    provider = settings.text("image_gen.provider", PROVIDER)
    if provider != PROVIDER:
        raise ConfigError(
            f"image_gen.provider must be {PROVIDER!r}, not {provider!r}"
        )
    model = settings.text("image_gen.model", DEFAULT_MODEL)
    if model not in _MODELS:
        choices = ", ".join(_MODELS)
        raise ConfigError(
            f"image_gen.model must be one of {choices}, not {model!r}"
        )
    return model


def _summarize_error_body(body: str) -> str:
    """Keep the actionable API error while bounding what enters history."""
    text = body or ""
    try:
        payload = json.loads(text)
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.strip():
            return message.strip()[:_MAX_ERROR_BODY_CHARS]
    except (TypeError, ValueError):
        pass
    return text[:_MAX_ERROR_BODY_CHARS]


def _resolve_aspect_ratio(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_ASPECT_RATIO
    candidate = value.strip().lower()
    return candidate if candidate in VALID_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO


def _normalize_reference_images(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    """Hermes' magic-byte checks, restricted to Codex-supported raster types."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _data_url_to_input_image_url(value: str) -> str:
    if "," not in value:
        raise ValueError("Image data URL is missing a comma separator")
    header, data = value.split(",", 1)
    header_lower = header.lower()
    if not header_lower.startswith("data:image/") or ";base64" not in header_lower:
        raise ValueError(
            "Only base64 data:image URLs are supported as Codex image inputs"
        )
    raw = base64.b64decode(data, validate=True)
    if len(raw) > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError("Image data URL exceeds 25MB cap")
    mime = _sniff_image_mime(raw)
    if mime not in _ACCEPTED_INPUT_MIME:
        raise ValueError("Image data URL does not contain supported image bytes")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _local_image_to_data_url(value: str) -> str:
    blocked = get_read_block_error(value)
    if blocked:
        raise ValueError(blocked)

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Image input path does not exist or is not a file: {value}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Image input path is empty: {value}")
    if size > _MAX_INPUT_IMAGE_BYTES:
        raise ValueError(f"Image input path exceeds 25MB cap: {value}")
    raw = path.read_bytes()
    mime = _sniff_image_mime(raw)
    if mime not in _ACCEPTED_INPUT_MIME:
        raise ValueError(f"Image input path is not a supported image: {value}")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _to_input_image_part(value: str) -> Dict[str, str]:
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("Blank image input")
    lowered = candidate.lower()
    if lowered.startswith(("http://", "https://")):
        image_url = candidate
    elif lowered.startswith("data:"):
        image_url = _data_url_to_input_image_url(candidate)
    else:
        image_url = _local_image_to_data_url(candidate)
    return {"type": "input_image", "image_url": image_url}


def _normalize_input_images(
    image_url: Any,
    reference_image_urls: Any,
) -> List[Dict[str, str]]:
    values: List[str] = []
    if isinstance(image_url, str) and image_url.strip():
        values.append(image_url.strip())
    values.extend(_normalize_reference_images(reference_image_urls))
    return [
        _to_input_image_part(value)
        for value in values[:_MAX_REFERENCE_IMAGES]
    ]


def _build_responses_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build Hermes' working Codex hosted-tool request shape."""
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if input_images:
        content.extend(input_images)
    return {
        "model": _CODEX_CHAT_MODEL,
        "store": False,
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": content,
            }
        ],
        "tools": [
            {
                "type": "image_generation",
                "model": API_MODEL,
                "size": size,
                "quality": quality,
                "output_format": "png",
                "background": "opaque",
                "partial_images": _PARTIAL_IMAGES_REQUESTED,
            }
        ],
        # The Codex backend rejects tool_choice for hosted image tools.  Hermes
        # steers with instructions and deliberately omits it.
        "stream": True,
    }


def _extract_image_candidates(value: Any) -> tuple[Optional[str], Optional[str]]:
    """Return ``(final, latest_partial)`` without letting a preview win."""
    final_b64: Optional[str] = None
    partial_b64: Optional[str] = None

    def walk(node: Any) -> None:
        nonlocal final_b64, partial_b64
        if isinstance(node, dict):
            if node.get("type") == "image_generation_call":
                result = node.get("result")
                if isinstance(result, str) and result:
                    final_b64 = result
            partial = node.get("partial_image_b64")
            if isinstance(partial, str) and partial:
                partial_b64 = partial
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return final_b64, partial_b64


def _extract_image_b64(value: Any) -> Optional[str]:
    """Return a final image when present, otherwise the partial candidate."""
    final_b64, partial_b64 = _extract_image_candidates(value)
    return final_b64 or partial_b64


def _png_pixel_size(raw: bytes) -> Optional[str]:
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return f"{width}x{height}"


def _iter_sse_json(response: Any) -> Iterable[Dict[str, Any]]:
    """Parse raw SSE because hosted image events may be newer than the SDK."""
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def flush() -> Optional[Dict[str, Any]]:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        raw = "\n".join(data_lines).strip()
        event = event_name
        event_name = None
        data_lines = []
        if not raw or raw == "[DONE]":
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        if event and "type" not in payload:
            payload["type"] = event
        return payload

    for line in response.iter_lines():
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = str(line)
        if line == "":
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    payload = flush()
    if payload is not None:
        yield payload


def _collect_image_b64(
    credentials: auth.Credentials,
    *,
    prompt: str,
    size: str,
    quality: str,
    input_images: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, str]]:
    headers = cloudflare_headers(credentials.access_token)
    headers.update(
        {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
        }
    )
    payload = _build_responses_payload(
        prompt=prompt,
        size=size,
        quality=quality,
        input_images=input_images,
    )
    timeout = httpx.Timeout(
        _REQUEST_TIMEOUT_SECONDS,
        connect=30.0,
        read=_REQUEST_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )
    endpoint = f"{credentials.base_url.rstrip('/')}/responses"

    final_b64: Optional[str] = None
    partial_b64: Optional[str] = None
    with httpx.Client(timeout=timeout, headers=headers) as client:
        with client.stream("POST", endpoint, json=payload) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                exc.response.read()
                detail = _summarize_error_body(exc.response.text)
                raise ImageAPIError(
                    exc.response.status_code,
                    f"Codex Responses API returned HTTP "
                    f"{exc.response.status_code}: {detail}",
                ) from exc
            for event in _iter_sse_json(response):
                event_final, event_partial = _extract_image_candidates(event)
                if event_final:
                    final_b64 = event_final
                if event_partial:
                    partial_b64 = event_partial
    if final_b64:
        return {"b64": final_b64, "source": "final"}
    if partial_b64:
        return {"b64": partial_b64, "source": "partial"}
    return None


def _resolve_credentials(config: Any, *, force_refresh: bool = False) -> auth.Credentials:
    return auth.resolve_credentials(
        Path(config.credentials_path),
        fallback_path=Path(config.main_credentials_path),
        force_refresh=force_refresh,
    )


def _save_b64_image(b64_data: str, workspace: Path, model: str) -> Path:
    """Save Hermes' PNG result inside Genesis' existing delivery boundary."""
    raw = base64.b64decode(b64_data)
    workspace = workspace.resolve()
    output_dir = workspace / "generated-images"
    output_dir.mkdir(parents=True, exist_ok=True)
    containment_error = validate_within_dir(output_dir, workspace)
    if containment_error:
        raise ValueError(containment_error)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    path = output_dir / f"openai_codex_{model}_{timestamp}_{short}.png"
    path.write_bytes(raw)
    return path.resolve()


def _output_root(context: ToolContext) -> Path:
    """Choose a workspace root that outbound-media confinement will accept.

    ``terminal.cwd`` may be broader than the directory the operator declared
    deliverable (for example ``/workspace`` with only ``/workspace/exports``
    allowed).  Saving directly below that cwd produces a valid image that is
    then silently rejected at delivery.  Prefer a declared root nested inside
    the active cwd, and fall back to the profile workspace, which is always a
    delivery root.
    """

    config = context.config
    working = Path(
        context.working_directory or config.workspace_dir
    ).expanduser().resolve(strict=False)
    if getattr(config, "session_isolated_workspaces", False):
        return (working / "exports").resolve(strict=False)

    profile_workspace = Path(config.workspace_dir).expanduser().resolve(
        strict=False
    )
    raw_roots = getattr(config, "outbound_media_roots", ()) or ()
    declared_roots = tuple(
        Path(root).expanduser().resolve(strict=False) for root in raw_roots
    ) or (profile_workspace,)

    # The active cwd is already deliverable in full.
    for root in declared_roots:
        try:
            working.relative_to(root)
            return working
        except ValueError:
            continue

    # Only a child such as <cwd>/exports is deliverable.
    for root in declared_roots:
        try:
            root.relative_to(working)
            return root
        except ValueError:
            continue

    return profile_workspace


def _error_response(
    *,
    error: str,
    error_type: str,
    prompt: str,
    aspect_ratio: str,
    model: str = "",
) -> Dict[str, Any]:
    return {
        "success": False,
        "image": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": PROVIDER,
    }


def _generate(args: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
    prompt_value = args.get("prompt")
    prompt = prompt_value.strip() if isinstance(prompt_value, str) else ""
    aspect = _resolve_aspect_ratio(args.get("aspect_ratio"))
    if not prompt:
        return _error_response(
            error="Prompt is required and must be a non-empty string",
            error_type="invalid_argument",
            prompt="",
            aspect_ratio=aspect,
        )

    try:
        model = validate_image_settings(context.config.settings)
    except ConfigError as exc:
        return _error_response(
            error=str(exc),
            error_type="configuration_error",
            prompt=prompt,
            aspect_ratio=aspect,
        )
    quality = _MODELS[model]["quality"]
    size = _SIZES[aspect]

    try:
        credentials = _resolve_credentials(context.config)
    except auth.AuthError as exc:
        return _error_response(
            error=str(exc),
            error_type="auth_required" if exc.relogin_required else "auth_error",
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    try:
        input_images = _normalize_input_images(
            args.get("image_url"),
            args.get("reference_image_urls"),
        )
    except Exception as exc:  # noqa: BLE001 - validation errors are model-visible
        return _error_response(
            error=f"Invalid image input for Codex image editing: {exc}",
            error_type="invalid_image_input",
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    collected: Optional[Dict[str, str]] = None
    credentials_refreshed = False
    for attempt in range(_NONFINAL_RETRIES + 1):
        while True:
            try:
                collected = _collect_image_b64(
                    credentials,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    input_images=input_images or None,
                )
                break
            except ImageAPIError as exc:
                if (
                    exc.status_code in {401, 403}
                    and not credentials_refreshed
                ):
                    try:
                        credentials = _resolve_credentials(
                            context.config,
                            force_refresh=True,
                        )
                    except auth.AuthError as refresh_exc:
                        return _error_response(
                            error=str(refresh_exc),
                            error_type=(
                                "auth_required"
                                if refresh_exc.relogin_required
                                else "auth_error"
                            ),
                            model=model,
                            prompt=prompt,
                            aspect_ratio=aspect,
                        )
                    credentials_refreshed = True
                    continue
                logger.debug("Codex image generation failed", exc_info=True)
                return _error_response(
                    error=(
                        "OpenAI image generation via Codex auth failed: "
                        f"{exc}"
                    ),
                    error_type="api_error",
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                logger.debug("Codex image generation failed", exc_info=True)
                return _error_response(
                    error=(
                        "OpenAI image generation via Codex auth failed: "
                        f"{exc}"
                    ),
                    error_type="api_error",
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        if (
            collected
            and collected.get("source") == "final"
            and collected.get("b64")
        ):
            break
        if attempt < _NONFINAL_RETRIES:
            kind = (
                "progressive-only partial frame"
                if collected and collected.get("source") == "partial"
                else "no image_generation_call result"
            )
            logger.warning(
                "Codex image stream ended with %s (attempt %s/%s); "
                "retrying once before failing closed.",
                kind,
                attempt + 1,
                _NONFINAL_RETRIES + 1,
            )

    if not collected or not collected.get("b64"):
        return _error_response(
            error=(
                "Codex response contained no image_generation_call result "
                f"after {_NONFINAL_RETRIES + 1} attempt(s)"
            ),
            error_type="empty_response",
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    image_source = collected.get("source") or "unknown"
    image_b64 = collected["b64"]
    if image_source != "final":
        try:
            partial_pixel_size = _png_pixel_size(
                base64.b64decode(image_b64, validate=False)
            )
        except Exception:
            partial_pixel_size = None
        detail = (
            "Codex returned only a progressive partial image frame after "
            f"{_NONFINAL_RETRIES + 1} attempt(s); refusing to save it as a "
            "final deliverable."
        )
        if partial_pixel_size:
            detail = f"{detail} partial_pixel_size={partial_pixel_size}."
        result = _error_response(
            error=detail,
            error_type="incomplete_image",
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
        )
        result.update(
            {
                "image_source": image_source,
                "requested_size": size,
                "partial_pixel_size": partial_pixel_size,
                "nonfinal_retries": _NONFINAL_RETRIES,
            }
        )
        return result

    try:
        pixel_size = _png_pixel_size(base64.b64decode(image_b64))
        output_root = _output_root(context)
        saved_path = _save_b64_image(
            image_b64,
            output_root,
            model,
        )
    except Exception as exc:  # noqa: BLE001 - filesystem errors vary
        return _error_response(
            error=f"Could not save image to the workspace: {exc}",
            error_type="io_error",
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    return {
        "success": True,
        "image": str(saved_path),
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect,
        "modality": "image" if input_images else "text",
        "provider": PROVIDER,
        "size": size,
        "quality": quality,
        "input_image_count": len(input_images),
        "image_source": image_source,
        "requested_size": size,
        "pixel_size": pixel_size,
    }


def _generate_with_slot(args: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
    with _IMAGE_SLOTS:
        return _generate(args, context)


async def handle_image_generate(args: Dict[str, Any], context: ToolContext) -> str:
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return tool_error(
            "prompt is required for image generation",
            success=False,
        )
    result = await asyncio.to_thread(_generate_with_slot, args, context)
    return json.dumps(result, ensure_ascii=False)


IMAGE_GENERATE_SCHEMA = {
    "name": "image_generate",
    "description": (
        "Generate a high-quality image from a detailed text prompt, or edit an "
        "existing image by passing `image_url` and optional "
        "`reference_image_urls`. The configured backend is fixed by the "
        "operator. A successful result contains an absolute local path in "
        "`image`; the runtime delivers that file to the user automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "A detailed description of the desired image, or the edit "
                    "to apply to the supplied image."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": list(VALID_ASPECT_RATIOS),
                "description": (
                    "Output shape: landscape, square, or portrait. "
                    "Defaults to landscape."
                ),
                "default": DEFAULT_ASPECT_RATIO,
            },
            "image_url": {
                "type": "string",
                "description": (
                    "Optional primary image to edit, as an http(s) URL, "
                    "base64 data:image URL, or absolute local image path."
                ),
            },
            "reference_image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_REFERENCE_IMAGES,
                "description": (
                    "Optional reference images for style, character, or "
                    "composition guidance."
                ),
            },
        },
        "required": ["prompt"],
    },
}

IMAGE_GENERATE_TOOL = Tool(
    name="image_generate",
    group="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=handle_image_generate,
    emoji="🎨",
    max_result_chars=100_000,
)


__all__ = [
    "DEFAULT_MODEL",
    "IMAGE_GENERATE_TOOL",
    "PROVIDER",
    "validate_image_settings",
]
