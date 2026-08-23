"""Native image analysis through the active ChatGPT/Codex model.

This is Hermes' production vision_analyze fast path, narrowed to Genesis'
single provider. Local files, HTTP(S) URLs, file URIs, and base64 data URLs are
resolved to verified image bytes, normalized, optionally cropped, proactively
resized for history reuse, and returned in Hermes' multimodal tool envelope.
The existing agent loop then gives those pixels back to the same main model.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import httpx

from .file_safety import get_read_block_error
from .registry import Tool, ToolContext, tool_error
from .terminal import get_terminal_session, shell_cwd
from .url_safety import (
    async_is_safe_url,
    create_ssrf_safe_async_client,
    redirect_target_from_response,
)

logger = logging.getLogger(__name__)

_MAX_INGEST_BYTES = 50 * 1024 * 1024
_MAX_BASE64_BYTES = 20 * 1024 * 1024
_EMBED_TARGET_BYTES = 256 * 1024
_EMBED_MAX_DIMENSION = 1568
_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_SUPPORTED_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


class ImageResolutionError(ValueError):
    """The named source could not safely become a supported image."""


@dataclass(frozen=True)
class ResolvedImage:
    data: bytes
    mime: str
    origin: str


def _detect_host_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


_VISION_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=_detect_host_cpus(),
    thread_name_prefix="vision-encode",
)


async def _run_cpu(function: Any, *args: Any, **kwargs: Any) -> Any:
    import functools

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _VISION_CPU_EXECUTOR,
        functools.partial(function, *args, **kwargs),
    )


def _detect_image_mime(data: bytes) -> Optional[str]:
    """Hermes' authoritative magic-byte image checks."""
    header = data[:64]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _finalize_image(data: bytes, origin: str) -> ResolvedImage:
    if len(data) > _MAX_INGEST_BYTES:
        raise ImageResolutionError("Image exceeds the 50MB ingest limit")
    mime = _detect_image_mime(data)
    if mime is not None:
        return ResolvedImage(data=data, mime=mime, origin=origin)
    if b"<svg" in data[:4096].lower():
        return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)
    raise ImageResolutionError("Source is not a recognized image")


def _resolve_data_url(source: str) -> ResolvedImage:
    header, separator, payload = source.partition(",")
    if not separator or ";base64" not in header.lower():
        raise ImageResolutionError("data: URL must be base64-encoded")
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise ImageResolutionError("data: URL exceeds the 50MB ingest limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ImageResolutionError(
            f"Invalid base64 in data: URL: {exc}"
        ) from exc
    return _finalize_image(data, "data")


def _file_uri_path(source: str) -> str:
    parsed = urlparse(source)
    if parsed.netloc not in {"", "localhost"}:
        raise ImageResolutionError("Remote file:// image sources are not allowed")
    path = unquote(parsed.path)
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


async def _resolve_local_image(
    source: str,
    context: ToolContext,
) -> ResolvedImage:
    raw = _file_uri_path(source) if source.lower().startswith("file://") else source
    session = get_terminal_session(context)
    async with session.lock:
        base = Path(
            session.shell.cwd
            if session.shell is not None
            else shell_cwd(context)
        )
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute():
            path = base / path
        path = path.resolve()

        blocked = get_read_block_error(str(path))
        if blocked:
            raise ImageResolutionError(blocked)
        if not path.is_file():
            raise ImageResolutionError(f"Image file not found: '{path}'")
        size = path.stat().st_size
        if size > _MAX_INGEST_BYTES:
            raise ImageResolutionError("Image exceeds the 50MB ingest limit")
    data = await asyncio.to_thread(path.read_bytes)
    return _finalize_image(data, "file")


def _is_retryable_download_error(error: Exception) -> bool:
    if isinstance(error, (ImageResolutionError, PermissionError, ValueError)):
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or status >= 500
    return True


async def _stream_download_to_file(
    client: httpx.AsyncClient,
    url: str,
    destination: Path,
) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    bytes_written = 0
    try:
        async with client.stream(
            "GET",
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                ),
                "Accept": "image/*,*/*;q=0.8",
            },
        ) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared:
                try:
                    declared_size = int(declared)
                except ValueError:
                    declared_size = None
                if (
                    declared_size is not None
                    and declared_size > _MAX_INGEST_BYTES
                ):
                    raise ImageResolutionError(
                        f"Image too large ({declared_size} bytes, "
                        f"max {_MAX_INGEST_BYTES})"
                    )

            with temporary.open("wb") as stream:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > _MAX_INGEST_BYTES:
                        raise ImageResolutionError(
                            f"Image too large ({bytes_written} bytes, "
                            f"max {_MAX_INGEST_BYTES})"
                        )
                    stream.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


async def _download_image(
    url: str,
    destination: Path,
    max_retries: int = 3,
) -> None:
    if not await async_is_safe_url(url):
        raise ImageResolutionError("Blocked unsafe or private image URL")

    async def _redirect_guard(response: httpx.Response) -> None:
        target = redirect_target_from_response(response)
        if target and not await async_is_safe_url(target):
            raise ImageResolutionError(
                f"Blocked redirect to private/internal address: {target}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            async with create_ssrf_safe_async_client(
                timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT_SECONDS),
                follow_redirects=True,
                event_hooks={"response": [_redirect_guard]},
            ) as client:
                await _stream_download_to_file(client, url, destination)
            return
        except Exception as exc:
            last_error = exc
            if (
                not _is_retryable_download_error(exc)
                or attempt >= max_retries - 1
            ):
                raise
            await asyncio.sleep(2 ** (attempt + 1))
    if last_error is not None:
        raise last_error
    raise ImageResolutionError("Image download was not attempted")


def _cache_dir(context: ToolContext) -> Path:
    state_dir = getattr(context.config, "state_dir", None)
    if state_dir is None:
        raise ImageResolutionError("Vision requires a configured profile state directory")
    path = Path(state_dir) / "cache" / "vision"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _resolve_image_source(
    source: str,
    context: ToolContext,
) -> ResolvedImage:
    if not isinstance(source, str) or not source.strip():
        raise ImageResolutionError("image_url is required")
    value = source.strip()
    lowered = value.lower()
    if lowered.startswith("data:"):
        return await asyncio.to_thread(_resolve_data_url, value)
    if lowered.startswith(("http://", "https://")):
        path = _cache_dir(context) / f"download_{uuid.uuid4().hex}.img"
        try:
            await _download_image(value, path)
            data = await asyncio.to_thread(path.read_bytes)
        finally:
            path.unlink(missing_ok=True)
        return _finalize_image(data, "http")
    if "://" in value and not lowered.startswith("file://"):
        raise ImageResolutionError(
            "Unrecognized image source scheme. Use http(s), file://, "
            "a local path, or a data: URL."
        )
    return await _resolve_local_image(value, context)


def _rasterize_svg_to_png(source: Path, destination: Path) -> bool:
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=str(source), write_to=str(destination))
        return destination.exists() and destination.stat().st_size > 0
    except Exception:
        pass

    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPM  # type: ignore

        drawing = svg2rlg(str(source))
        if drawing is not None:
            renderPM.drawToFile(drawing, str(destination), fmt="PNG")
            return destination.exists() and destination.stat().st_size > 0
    except Exception:
        pass

    for command in (
        ["rsvg-convert", "-o", str(destination), str(source)],
        [
            "inkscape",
            str(source),
            "--export-type=png",
            f"--export-filename={destination}",
        ],
    ):
        if not shutil.which(command[0]):
            continue
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if destination.exists() and destination.stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


def _normalize_supported_image(
    image_path: Path,
    detected_mime: str,
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    if detected_mime in _SUPPORTED_MEDIA_TYPES:
        return image_path, detected_mime, None

    output = image_path.with_name(f"converted_{uuid.uuid4().hex}.png")
    if detected_mime == "image/svg+xml":
        if _rasterize_svg_to_png(image_path, output):
            return output, "image/png", None
        output.unlink(missing_ok=True)
        return (
            None,
            None,
            "SVG images require a rasterizer before vision can inspect them. "
            "Convert this image to PNG and retry.",
        )

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGBA")
            image.save(output, format="PNG")
        if output.exists() and output.stat().st_size > 0:
            return output, "image/png", None
    except Exception as exc:
        logger.warning(
            "Failed to normalize %s image to PNG: %s",
            detected_mime,
            exc,
        )
    output.unlink(missing_ok=True)
    return (
        None,
        None,
        f"Image format {detected_mime!r} could not be converted to PNG.",
    )


def _crop_image_region(
    image_path: Path,
    region: Any,
    offset_out: Optional[dict] = None,
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    try:
        from PIL import Image
    except ImportError:
        return None, None, "Region cropping requires Pillow."

    if (
        not isinstance(region, (list, tuple))
        or len(region) != 4
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in region
        )
    ):
        return (
            None,
            None,
            "Invalid region: expected [x1, y1, x2, y2] as four numbers.",
        )

    output: Optional[Path] = None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            x1, y1, x2, y2 = (int(value) for value in region)
            cx1 = max(0, min(x1, width))
            cy1 = max(0, min(y1, height))
            cx2 = max(0, min(x2, width))
            cy2 = max(0, min(y2, height))
            if cx2 <= cx1 or cy2 <= cy1:
                return (
                    None,
                    None,
                    f"Invalid region [{x1}, {y1}, {x2}, {y2}]: zero area "
                    f"after clamping to the {width}x{height} image.",
                )
            cropped = image.crop((cx1, cy1, cx2, cy2))
            if offset_out is not None:
                offset_out.update(
                    x=cx1,
                    y=cy1,
                    width=cx2 - cx1,
                    height=cy2 - cy1,
                )
            output = image_path.with_name(
                f"{image_path.stem}_region_{uuid.uuid4().hex[:8]}.png"
            )
            if cropped.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                cropped = cropped.convert("RGB")
            cropped.save(output, format="PNG")
        return output, "image/png", None
    except Exception as exc:
        if output is not None:
            output.unlink(missing_ok=True)
        return None, None, f"Failed to crop region: {exc}"


def _build_scale_note(
    scale_info: Optional[dict],
    crop_offset: Optional[dict],
) -> Optional[str]:
    parts: list[str] = []
    if scale_info:
        original_width = scale_info["orig_width"]
        original_height = scale_info["orig_height"]
        new_width = scale_info["new_width"]
        new_height = scale_info["new_height"]
        factor_x = original_width / new_width if new_width else 1.0
        factor_y = original_height / new_height if new_height else 1.0
        if f"{factor_x:.2f}" == f"{factor_y:.2f}":
            mapping = (
                f"multiply reported coordinates by {factor_x:.2f} "
                "to map them to the original image."
            )
        else:
            mapping = (
                f"multiply x coordinates by {factor_x:.2f} and y "
                f"coordinates by {factor_y:.2f} to map them to the original."
            )
        parts.append(
            f"Image downscaled from {original_width}x{original_height} to "
            f"{new_width}x{new_height}; {mapping}"
        )
    if crop_offset:
        parts.append(
            "Analysis used a crop starting at "
            f"({crop_offset['x']}, {crop_offset['y']}); add that offset "
            "to map crop-relative coordinates to the full image."
        )
    return " ".join(parts) if parts else None


def _image_to_data_url(image_path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_exceeds_dimension(image_path: Path, maximum: int) -> bool:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return max(image.size) > maximum
    except Exception:
        return False


def _resize_image_for_vision(
    image_path: Path,
    mime_type: str,
    max_base64_bytes: int,
    max_dimension: int,
    scale_out: Optional[dict] = None,
) -> str:
    file_size = image_path.stat().st_size
    estimated_base64 = (file_size * 4) // 3 + 100
    needs_resize_for_bytes = estimated_base64 > max_base64_bytes
    needs_resize_for_dimensions = _image_exceeds_dimension(
        image_path,
        max_dimension,
    )

    data_url: Optional[str]
    if not needs_resize_for_bytes and not needs_resize_for_dimensions:
        data_url = _image_to_data_url(image_path, mime_type)
        if len(data_url) <= max_base64_bytes:
            return data_url
    else:
        data_url = None

    try:
        import io
        from PIL import Image
    except ImportError:
        return data_url or _image_to_data_url(image_path, mime_type)

    try:
        with Image.open(image_path) as opened:
            image = opened.copy()
    except Exception:
        return data_url or _image_to_data_url(image_path, mime_type)

    output_format = "PNG" if mime_type == "image/png" else "JPEG"
    output_mime = "image/png" if output_format == "PNG" else "image/jpeg"
    if output_format == "JPEG" and image.mode in {"RGBA", "P"}:
        converted = image.convert("RGB")
        image.close()
        image = converted

    quality_steps = (85, 70, 50) if output_format == "JPEG" else (None,)
    original_dimensions = (image.width, image.height)
    previous_dimensions = original_dimensions
    candidate: Optional[str] = None

    def _record_scale(width: int, height: int) -> None:
        if scale_out is not None and (width, height) != original_dimensions:
            scale_out.update(
                orig_width=original_dimensions[0],
                orig_height=original_dimensions[1],
                new_width=width,
                new_height=height,
            )

    for attempt in range(5):
        if attempt > 0:
            new_width = max(int(image.width * 0.5), 64)
            new_height = max(int(image.height * 0.5), 64)
            if new_width == 64 and image.width > 0:
                effective_scale = 64 / image.width
                new_height = max(int(image.height * effective_scale), 64)
            elif new_height == 64 and image.height > 0:
                effective_scale = 64 / image.height
                new_width = max(int(image.width * effective_scale), 64)
            if (new_width, new_height) == previous_dimensions:
                break
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            image.close()
            image = resized
            previous_dimensions = (new_width, new_height)

        for quality in quality_steps:
            buffer = io.BytesIO()
            options: Dict[str, Any] = {"format": output_format}
            if quality is not None:
                options["quality"] = quality
            image.save(buffer, **options)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            candidate = f"data:{output_mime};base64,{encoded}"
            if (
                len(candidate) <= max_base64_bytes
                and max(image.width, image.height) <= max_dimension
            ):
                _record_scale(image.width, image.height)
                image.close()
                return candidate

    if candidate is not None:
        _record_scale(image.width, image.height)
        image.close()
        return candidate
    image.close()
    return data_url or _image_to_data_url(image_path, mime_type)


def _build_native_result(
    image_url: str,
    question: str,
    image_data_url: str,
    image_size_bytes: int,
    scale_note: Optional[str],
) -> Dict[str, Any]:
    text = (
        "Image loaded into your context — you can see it natively now. "
        "Use your built-in vision to answer the user."
    )
    if question.strip():
        text += f"\n\nQuestion: {question.strip()}"
    if scale_note:
        text += f"\n\nNote: {scale_note}"
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
        ],
        "text_summary": (
            "Image attached natively for the main model "
            f"({image_size_bytes / 1024:.1f} KB). "
            "Answer using built-in vision."
        ),
        "meta": {
            "image_url": image_url[:200],
            "size_bytes": image_size_bytes,
            "native_vision": True,
        },
    }


async def handle_vision_analyze(
    args: Dict[str, Any],
    context: ToolContext,
) -> Any:
    image_url = args.get("image_url")
    question = args.get("question")
    if not isinstance(image_url, str) or not image_url.strip():
        return tool_error("image_url is required", success=False)
    if not isinstance(question, str):
        return tool_error("question is required", success=False)

    temporary_paths: set[Path] = set()
    try:
        resolved = await _resolve_image_source(image_url, context)
        image_path = _cache_dir(context) / f"image_{uuid.uuid4().hex}.img"
        await asyncio.to_thread(image_path.write_bytes, resolved.data)
        temporary_paths.add(image_path)
        image_size_bytes = len(resolved.data)
        mime_type = resolved.mime

        normalized, normalized_mime, normalization_error = await asyncio.to_thread(
            _normalize_supported_image,
            image_path,
            mime_type,
        )
        if normalization_error or normalized is None or normalized_mime is None:
            return tool_error(
                normalization_error or "Image normalization failed",
                success=False,
            )
        if normalized != image_path:
            temporary_paths.add(normalized)
        image_path = normalized
        mime_type = normalized_mime
        image_size_bytes = image_path.stat().st_size

        crop_offset: dict = {}
        scale_info: dict = {}
        region = args.get("region")
        if region is not None:
            cropped, cropped_mime, crop_error = await asyncio.to_thread(
                _crop_image_region,
                image_path,
                region,
                crop_offset,
            )
            if crop_error or cropped is None or cropped_mime is None:
                return tool_error(
                    crop_error or "Region crop failed",
                    success=False,
                )
            temporary_paths.add(cropped)
            image_path = cropped
            mime_type = cropped_mime
            image_size_bytes = image_path.stat().st_size

        image_data_url = await _run_cpu(
            _image_to_data_url,
            image_path,
            mime_type,
        )
        exceeds_dimensions = await _run_cpu(
            _image_exceeds_dimension,
            image_path,
            _EMBED_MAX_DIMENSION,
        )
        if (
            len(image_data_url) > _EMBED_TARGET_BYTES
            or exceeds_dimensions
        ):
            image_data_url = await _run_cpu(
                _resize_image_for_vision,
                image_path,
                mime_type,
                _EMBED_TARGET_BYTES,
                _EMBED_MAX_DIMENSION,
                scale_info,
            )

        if len(image_data_url) > _MAX_BASE64_BYTES:
            return tool_error(
                "Image remains too large for vision after resizing "
                f"({len(image_data_url) / (1024 * 1024):.1f} MB; "
                f"limit {_MAX_BASE64_BYTES / (1024 * 1024):.0f} MB).",
                success=False,
            )

        return _build_native_result(
            image_url=image_url,
            question=question,
            image_data_url=image_data_url,
            image_size_bytes=image_size_bytes,
            scale_note=_build_scale_note(
                scale_info or None,
                crop_offset or None,
            ),
        )
    except ImageResolutionError as exc:
        return tool_error(str(exc), success=False)
    except Exception as exc:
        logger.warning("Native vision analysis failed: %s", exc, exc_info=True)
        return tool_error(f"Native vision failed: {exc}", success=False)
    finally:
        for path in temporary_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Could not remove temporary vision image %s",
                    path,
                    exc_info=True,
                )


VISION_ANALYZE_SCHEMA = {
    "name": "vision_analyze",
    "description": (
        "Load an image into the conversation so you can inspect it with your "
        "native vision. Accepts an HTTP(S) URL, local file path, file:// URI, "
        "or base64 data URL. Call this whenever the user references an image "
        "path or an image appears in tool output."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": (
                    "HTTP(S) URL, local path, file:// URI, or data: image URL."
                ),
            },
            "question": {
                "type": "string",
                "description": "The specific question to answer about the image.",
            },
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Optional [x1, y1, x2, y2] crop in original-image pixels. "
                    "Use after loading the full image to zoom into a detail."
                ),
            },
        },
        "required": ["image_url", "question"],
    },
}


VISION_ANALYZE_TOOL = Tool(
    name="vision_analyze",
    group="vision",
    schema=VISION_ANALYZE_SCHEMA,
    handler=handle_vision_analyze,
    emoji="👁️",
    max_result_chars=100_000,
)


__all__ = [
    "VISION_ANALYZE_SCHEMA",
    "VISION_ANALYZE_TOOL",
    "handle_vision_analyze",
]
