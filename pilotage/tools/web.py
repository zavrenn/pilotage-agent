"""Web search and extraction, adapted from Hermes.

Pilotage needs the one backend its production profiles already use, not
Hermes' provider registry or runtime installer. The load-bearing behavior is
kept: normalized results, a hard 30-second wall-clock bound, and a disposable
worker process because ``ddgs``/``primp`` can block in native code while
holding the Python GIL. The synchronous worker wait runs off Genesis' asyncio
loop so one slow search does not freeze every chat.

Full-page extraction keeps Hermes' Firecrawl path: per-URL timeouts, URL and
redirect safety checks, response-shape normalization, base64 image removal,
and deterministic truncate-and-store output. Genesis drops only provider
routing and gateway support; Firecrawl is the single extraction backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ..settings import ConfigError, Settings
from .registry import Tool, ToolContext, tool_error
from .spill_safety import ensure_spill_dir, write_text_exclusive
from .url_safety import (
    async_is_safe_url,
    contains_known_secret,
    has_url_credentials,
    normalize_url_for_request,
    sensitive_query_param_name,
)

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECONDS = 30.0
_WORKER_REAP_SECONDS = 1.0
_MAX_RESULTS = 100
_WORKER_MODULE = "pilotage.tools._ddgs_worker"
_TEST_HOOK_ENV = "PILOTAGE_DDGS_ALLOW_TEST_HOOKS"

DEFAULT_EXTRACT_CHAR_LIMIT = 15_000
MAX_STORED_TEXT_CHARS = 2_000_000
_MIN_EXTRACT_CHAR_LIMIT = 2_000
_MAX_EXTRACT_CHAR_LIMIT = 500_000
_MAX_EXTRACT_URLS = 5
_EXTRACT_TIMEOUT_SECONDS = 60.0
_DEFAULT_WEB_RESULT_CHAR_LIMIT = 100_000

# Test-only hook forwarded to the isolated worker. Production never sets it.
_test_hook: Optional[str] = None
_last_worker_proc: Optional[subprocess.Popen] = None
_firecrawl_client: Any = None
_firecrawl_client_config: Optional[tuple[str, Optional[str], Optional[str]]] = None


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run one blocking DDGS query and normalize its result shape."""
    from ddgs import DDGS  # type: ignore

    results: list[dict[str, Any]] = []
    with DDGS(timeout=10) as client:
        for index, hit in enumerate(client.text(query, max_results=safe_limit)):
            if index >= safe_limit:
                break
            results.append(
                {
                    "title": str(hit.get("title", "")),
                    "url": str(hit.get("href") or hit.get("url") or ""),
                    "description": str(hit.get("body", "")),
                    "position": index + 1,
                }
            )
    return results


def _worker_environment() -> Dict[str, str]:
    """Give the search child what networking needs, without agent secrets."""
    allowed = {
        "ALL_PROXY",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONUTF8"] = "1"
    if _test_hook:
        env[_TEST_HOOK_ENV] = "1"
    return env


def _terminate_and_reap(proc: subprocess.Popen) -> None:
    """Kill a worker and wait for it so a timeout cannot leave an orphan."""
    try:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=_WORKER_REAP_SECONDS)
    except Exception as exc:  # noqa: BLE001 - cleanup is best effort
        logger.warning("Could not reap DDGS worker pid=%s: %s", proc.pid, exc)


def _run_ddgs_search_bounded(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run DDGS in a disposable process with a hard wall-clock deadline."""
    global _last_worker_proc

    request: Dict[str, Any] = {"query": query, "safe_limit": safe_limit}
    if _test_hook:
        request["test_hook"] = _test_hook

    process_options: Dict[str, Any] = {}
    if sys.platform == "win32":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        process_options["start_new_session"] = True

    proc = subprocess.Popen(
        [sys.executable, "-m", _WORKER_MODULE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_worker_environment(),
        text=True,
        encoding="utf-8",
        errors="replace",
        **process_options,
    )
    _last_worker_proc = proc
    try:
        raw, _ = proc.communicate(
            json.dumps(request, ensure_ascii=False),
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        _terminate_and_reap(proc)
        raise TimeoutError(
            f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECONDS:g}s"
        ) from exc
    except BaseException:
        _terminate_and_reap(proc)
        raise

    raw = (raw or "").strip()
    if not raw:
        raise RuntimeError(f"DDGS worker exited without a result (code={proc.returncode})")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DDGS worker returned invalid JSON: {raw[:200]!r}") from exc
    if not isinstance(envelope, dict):
        raise RuntimeError(f"DDGS worker returned an invalid envelope: {envelope!r}")
    if not envelope.get("ok"):
        raise RuntimeError(str(envelope.get("error") or "DDGS worker failed"))
    results = envelope.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError("DDGS worker returned non-list results")
    return results


def _search(query: str, limit: Any = 5) -> Dict[str, Any]:
    """Hermes' DDGS result contract, reduced to Pilotage's fixed backend."""
    try:
        import ddgs  # noqa: F401
    except ImportError:
        return {
            "success": False,
            "error": (
                "The ddgs dependency is not installed. "
                "Run scripts/install.sh again before using web search."
            ),
        }

    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = 5
    safe_limit = min(max(safe_limit, 1), _MAX_RESULTS)

    try:
        results = _run_ddgs_search_bounded(query, safe_limit)
    except TimeoutError:
        logger.warning(
            "DDGS search timed out after %gs for query: %r",
            _SEARCH_TIMEOUT_SECONDS,
            query,
        )
        return {
            "success": False,
            "error": (
                f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECONDS:g}s. "
                "DuckDuckGo may be slow or rate-limiting; try again later."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - DDGS owns its exception types
        logger.warning("DDGS search failed for %r: %s", query, exc)
        return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

    logger.info("DDGS search %r returned %d results", query, len(results))
    return {"success": True, "data": {"web": results}}


async def handle_web_search(args: Dict[str, Any], context: ToolContext) -> str:
    del context
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query must not be empty", success=False)
    result = await asyncio.to_thread(_search, query, args.get("limit", 5))
    return json.dumps(result, ensure_ascii=False)


def validate_web_settings(settings: Settings) -> int:
    """Validate Hermes' per-page extraction budget at startup."""
    char_limit = settings.count(
        "web.extract_char_limit",
        DEFAULT_EXTRACT_CHAR_LIMIT,
    )
    if not _MIN_EXTRACT_CHAR_LIMIT <= char_limit <= _MAX_EXTRACT_CHAR_LIMIT:
        raise ConfigError(
            "web.extract_char_limit must be between "
            f"{_MIN_EXTRACT_CHAR_LIMIT} and {_MAX_EXTRACT_CHAR_LIMIT}, "
            f"not {char_limit!r}"
        )
    return char_limit


def _web_extract_url(value: Any) -> Optional[str]:
    """Return a URL string from a URL or a forwarded search-result object."""
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _get_direct_firecrawl_config(
) -> tuple[Dict[str, str], tuple[str, Optional[str], Optional[str]]]:
    """Resolve direct cloud or self-hosted Firecrawl configuration."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    api_url = os.environ.get("FIRECRAWL_API_URL", "").strip().rstrip("/")
    if not api_key and not api_url:
        raise ValueError(
            "Web extraction is not configured. Set FIRECRAWL_API_KEY in the "
            "profile .env for cloud Firecrawl, or FIRECRAWL_API_URL for a "
            "self-hosted Firecrawl instance."
        )

    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_url:
        kwargs["api_url"] = api_url
    return kwargs, ("direct", api_url or None, api_key or None)


def _get_firecrawl_client() -> Any:
    """Lazily create Hermes' Firecrawl SDK client and cache its config."""
    global _firecrawl_client, _firecrawl_client_config

    kwargs, client_config = _get_direct_firecrawl_config()
    if _firecrawl_client is not None and _firecrawl_client_config == client_config:
        return _firecrawl_client
    try:
        from firecrawl import Firecrawl  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "The firecrawl-py dependency is not installed. Run "
            "scripts/install.sh again before using web extraction."
        ) from exc
    _firecrawl_client = Firecrawl(**kwargs)
    _firecrawl_client_config = client_config
    return _firecrawl_client


def _to_plain_object(value: Any) -> Any:
    """Normalize Firecrawl SDK objects to plain Python values."""
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001 - SDK compatibility fallback
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                key: item
                for key, item in value.__dict__.items()
                if not key.startswith("_")
            }
        except Exception:  # noqa: BLE001 - SDK compatibility fallback
            pass
    return value


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    """Normalize Firecrawl SDK, direct, and gateway-era response shapes."""
    result = _to_plain_object(scrape_result)
    if not isinstance(result, dict):
        return {}
    nested = result.get("data")
    return nested if isinstance(nested, dict) else result


async def _extract_firecrawl(
    urls: list[str],
    *,
    format: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Hermes' sequential, bounded Firecrawl per-URL extraction loop."""
    if format == "markdown":
        formats = ["markdown"]
    elif format == "html":
        formats = ["html"]
    else:
        formats = ["markdown", "html"]

    client = _get_firecrawl_client()
    results: list[Dict[str, Any]] = []
    for url in urls:
        try:
            logger.info("Firecrawl scraping: %s", url)
            try:
                scrape_result = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.scrape,
                        url=url,
                        formats=formats,
                    ),
                    timeout=_EXTRACT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning("Firecrawl scrape timed out for %s", url)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": (
                            "Scrape timed out after "
                            f"{_EXTRACT_TIMEOUT_SECONDS:g}s; the page may be "
                            "too large or unresponsive."
                        ),
                    }
                )
                continue

            payload = _extract_scrape_payload(scrape_result)
            metadata = _to_plain_object(payload.get("metadata", {}))
            if not isinstance(metadata, dict):
                metadata = {}
            markdown = payload.get("markdown")
            html = payload.get("html")
            title = str(metadata.get("title") or "")
            # Firecrawl's v2 SDK maps ``sourceURL`` to ``source_url``;
            # direct/gateway-era payloads retain the camelCase key.
            reported_url = (
                metadata.get("source_url")
                or metadata.get("sourceURL")
                or metadata.get("url")
                or url
            )
            final_url = reported_url if isinstance(reported_url, str) else url

            # Firecrawl follows redirects on its server; re-check the reported
            # destination before returning it as a trusted fetch result.
            if has_url_credentials(final_url):
                logger.info(
                    "Blocked redirected web_extract with credential-bearing "
                    "final URL"
                )
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": "",
                        "raw_content": "",
                        "error": (
                            "Blocked: redirected URL embeds username/password "
                            "credentials"
                        ),
                    }
                )
                continue
            if not await async_is_safe_url(final_url):
                logger.info(
                    "Blocked redirected web_extract for unsafe final URL: %s",
                    final_url,
                )
                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": "",
                        "raw_content": "",
                        "error": (
                            "Blocked: URL targets a private or internal "
                            "network address"
                        ),
                    }
                )
                continue

            if format == "markdown" or (format is None and markdown):
                content = markdown
            else:
                content = html or markdown or ""
            if not isinstance(content, str):
                content = str(content or "")
            results.append(
                {
                    "url": final_url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one URL must not lose the rest
            logger.debug("Firecrawl scrape failed for %s: %s", url, exc)
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": str(exc),
                }
            )
    return results


def convert_base64_images_to_links(text: str) -> str:
    """Replace inline base64 image token bombs with labeled placeholders."""

    def markdown_replacement(match: re.Match[str]) -> str:
        alt = (match.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    markdown_image = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
    )
    cleaned = markdown_image.sub(markdown_replacement, text)
    cleaned = re.sub(
        r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)",
        "[IMAGE]",
        cleaned,
    )
    return re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "[IMAGE]",
        cleaned,
    )


def _store_full_text(context: ToolContext, url: str, content: str) -> Optional[str]:
    """Store one bounded full-text copy in this profile's web cache."""
    try:
        state_dir = Path(context.config.state_dir)
        cache_dir = ensure_spill_dir(state_dir / "cache" / "web", private=True)
        host = (urlparse(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"{slug}-{digest}.md"
        if len(content) > MAX_STORED_TEXT_CHARS:
            original_length = len(content)
            content = (
                content[:MAX_STORED_TEXT_CHARS]
                + "\n\n[... stored copy truncated at "
                f"{MAX_STORED_TEXT_CHARS:,} chars of {original_length:,}; "
                "re-extract a more specific URL for the rest ...]"
            )
        write_text_exclusive(path, content, private=True, overwrite=True)
        return str(path.resolve())
    except Exception as exc:  # noqa: BLE001 - storage is best effort
        logger.debug("Failed to store full web_extract text for %s: %s", url, exc)
        return None


def _truncate_with_footer(
    context: ToolContext,
    content: str,
    url: str,
    char_limit: int,
) -> tuple[str, bool]:
    """Return Hermes' head/tail page window and a full-text read cursor."""
    if len(content) <= char_limit:
        return content, False

    stored_path = _store_full_text(context, url, content)
    return _format_truncated_content(content, char_limit, stored_path), True


def _format_truncated_content(
    content: str,
    char_limit: int,
    stored_path: Optional[str],
) -> str:
    """Format one cached page preview without repeating the cache write."""
    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget
    head = content[:head_budget]
    tail = content[-tail_budget:]
    newline = head.rfind("\n")
    if newline > head_budget * 0.5:
        head = head[:newline]
    newline = tail.find("\n")
    if 0 <= newline < tail_budget * 0.5:
        tail = tail[newline + 1 :]

    footer = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {len(content):,} total clean characters.",
    ]
    if stored_path:
        middle_start_line = head.count("\n") + 2
        footer.append(f"Full text saved to: {stored_path}")
        footer.append(
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete "
            "page; raise/lower offset to page through it)."
        )
    else:
        footer.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL for the complete page."
        )
    footer.append("─" * 29)
    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    return model_text + "\n" + "\n".join(footer)


def _effective_extract_char_limit(args: Dict[str, Any], context: ToolContext) -> int:
    configured = DEFAULT_EXTRACT_CHAR_LIMIT
    settings = getattr(context.config, "settings", None)
    if settings is not None:
        configured = validate_web_settings(settings)
    requested = args.get("char_limit")
    if requested is None:
        return configured
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return configured
    return max(_MIN_EXTRACT_CHAR_LIMIT, min(value, _MAX_EXTRACT_CHAR_LIMIT))


def _extract_result_char_limit(context: ToolContext) -> int:
    """Return the same per-result ceiling the Agent passes to the registry."""
    value = getattr(context.config, "max_tool_result_chars", None)
    if value is None:
        return _DEFAULT_WEB_RESULT_CHAR_LIMIT
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_WEB_RESULT_CHAR_LIMIT
    return limit if limit > 0 else _DEFAULT_WEB_RESULT_CHAR_LIMIT


def _render_extract_payload(
    context: ToolContext,
    results: list[Dict[str, Any]],
    clean_contents: list[str],
    char_limit: int,
    stored_paths: Dict[int, Optional[str]],
) -> str:
    """Render valid JSON, caching each page at most once when truncated."""
    trimmed: list[Dict[str, Any]] = []
    for index, result in enumerate(results):
        error = result.get("error")
        content = ""
        if not error:
            clean_content = clean_contents[index]
            if len(clean_content) <= char_limit:
                content = clean_content
            else:
                if index not in stored_paths:
                    stored_paths[index] = _store_full_text(
                        context,
                        str(result.get("url") or ""),
                        clean_content,
                    )
                content = _format_truncated_content(
                    clean_content,
                    char_limit,
                    stored_paths[index],
                )
        trimmed.append(
            {
                "url": str(result.get("url") or ""),
                "title": str(result.get("title") or ""),
                "content": content,
                "error": error,
            }
        )
    return json.dumps({"results": trimmed}, ensure_ascii=False)


def _fit_extract_payload(
    context: ToolContext,
    results: list[Dict[str, Any]],
    clean_contents: list[str],
    char_limit: int,
) -> str:
    """Keep recovery footers intact under Genesis' actual result ceiling."""
    result_limit = _extract_result_char_limit(context)
    stored_paths: Dict[int, Optional[str]] = {}
    payload = _render_extract_payload(
        context,
        results,
        clean_contents,
        char_limit,
        stored_paths,
    )
    if len(payload) <= result_limit:
        return payload

    best: Optional[str] = None
    low, high = 1, char_limit - 1
    while low <= high:
        candidate_limit = (low + high) // 2
        candidate = _render_extract_payload(
            context,
            results,
            clean_contents,
            candidate_limit,
            stored_paths,
        )
        if len(candidate) <= result_limit:
            best = candidate
            low = candidate_limit + 1
        else:
            high = candidate_limit - 1
    if best is not None:
        return best

    error = tool_error(
        "tools.max_result_chars is too small for web_extract recovery metadata; "
        "increase it and retry",
        success=False,
    )
    if len(error) <= result_limit:
        return error
    return "{}" if result_limit >= 2 else "0"


async def handle_web_extract(args: Dict[str, Any], context: ToolContext) -> str:
    """Extract up to five URLs through the fixed Hermes Firecrawl backend."""
    supplied = args.get("urls")
    if not isinstance(supplied, list):
        return tool_error("urls must be an array", success=False)
    urls = supplied[:_MAX_EXTRACT_URLS]
    if not urls:
        return tool_error("urls must contain at least one URL", success=False)

    normalized_urls: list[str] = []
    normalized_indices: list[int] = []
    invalid: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(urls):
        raw_url = _web_extract_url(item)
        if raw_url is None:
            invalid[index] = {
                "url": "",
                "title": "",
                "content": "",
                "error": (
                    f"Invalid URL item at index {index}: expected a URL string "
                    "or an object with a string 'url' or 'href' field"
                ),
            }
            continue
        normalized = normalize_url_for_request(raw_url)
        if has_url_credentials(raw_url) or has_url_credentials(normalized):
            return tool_error(
                "Blocked: URL embeds username/password credentials. Firecrawl "
                "is a third-party reader; remove credentials from the URL.",
                success=False,
            )
        if contains_known_secret(raw_url) or contains_known_secret(normalized):
            return tool_error(
                "Blocked: URL contains what appears to be an API key or token. "
                "Secrets must not be sent in URLs.",
                success=False,
            )
        sensitive_key = sensitive_query_param_name(normalized)
        if sensitive_key:
            return tool_error(
                "Blocked: URL contains a credential-like query parameter "
                f"({sensitive_key}). Firecrawl is a third-party reader; remove "
                "the sensitive query parameter.",
                success=False,
            )
        normalized_urls.append(normalized)
        normalized_indices.append(index)

    safe_urls: list[str] = []
    safe_indices: list[int] = []
    blocked: Dict[int, Dict[str, Any]] = {}
    for index, url in zip(normalized_indices, normalized_urls):
        if await async_is_safe_url(url):
            safe_urls.append(url)
            safe_indices.append(index)
        else:
            blocked[index] = {
                "url": url,
                "title": "",
                "content": "",
                "error": (
                    "Blocked: URL targets a private or internal network address"
                ),
            }

    try:
        provider_results = (
            await _extract_firecrawl(safe_urls, format="markdown")
            if safe_urls
            else []
        )
        by_index: Dict[int, Dict[str, Any]] = {**invalid, **blocked}
        for position, index in enumerate(safe_indices):
            if position < len(provider_results):
                by_index[index] = provider_results[position]
            else:
                by_index[index] = {
                    "url": safe_urls[position],
                    "title": "",
                    "content": "",
                    "error": "Extract backend returned no result for this URL",
                }
        results = [by_index[index] for index in range(len(urls))]

        char_limit = _effective_extract_char_limit(args, context)
        clean_contents: list[str] = []
        for result in results:
            error = result.get("error")
            if not error:
                raw_content = result.get("raw_content") or result.get("content") or ""
            else:
                raw_content = ""
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
            clean_contents.append(convert_base64_images_to_links(raw_content))
        return _fit_extract_payload(context, results, clean_contents, char_limit)
    except Exception as exc:  # noqa: BLE001 - the model needs a recoverable error
        logger.debug("Error extracting content: %s", exc)
        return tool_error(f"Error extracting content: {exc}", success=False)


WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web for information. Returns up to 5 results by default "
        "with titles, URLs, and descriptions. DuckDuckGo may support operators "
        'such as site:domain, filetype:pdf, intitle:word, -term, and "exact phrase".'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. It may include DuckDuckGo-supported "
                    "operators such as site:example.com or filetype:pdf."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": _MAX_RESULTS,
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": (
        "Extract clean markdown/text from web page URLs through Firecrawl, "
        "including PDF URLs. Pages within the character budget return whole; "
        "larger pages return a head/tail window and save the full text for "
        "read_file. Inline base64 images become [IMAGE: alt] placeholders."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs to extract, with at most 5 per call.",
                "maxItems": _MAX_EXTRACT_URLS,
            },
            "char_limit": {
                "type": "integer",
                "description": (
                    "Optional per-page character budget. Defaults to the "
                    "profile's web.extract_char_limit setting (15000). The "
                    "returned previews shrink automatically when needed to fit "
                    "tools.max_result_chars; full text remains cached."
                ),
                "minimum": _MIN_EXTRACT_CHAR_LIMIT,
                "maximum": _MAX_EXTRACT_CHAR_LIMIT,
            },
        },
        "required": ["urls"],
    },
}

WEB_SEARCH_TOOL = Tool(
    name="web_search",
    group="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=handle_web_search,
    emoji="🔍",
    max_result_chars=_DEFAULT_WEB_RESULT_CHAR_LIMIT,
)

WEB_EXTRACT_TOOL = Tool(
    name="web_extract",
    group="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=handle_web_extract,
    emoji="📄",
    max_result_chars=_DEFAULT_WEB_RESULT_CHAR_LIMIT,
)
