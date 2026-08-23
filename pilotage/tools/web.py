"""DuckDuckGo web search, adapted from Hermes' DDGS provider.

Pilotage needs the one backend its production profiles already use, not
Hermes' provider registry or runtime installer. The load-bearing behavior is
kept: normalized results, a hard 30-second wall-clock bound, and a disposable
worker process because ``ddgs``/``primp`` can block in native code while
holding the Python GIL. The synchronous worker wait runs off Genesis' asyncio
loop so one slow search does not freeze every chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Optional

from .registry import Tool, ToolContext, tool_error

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECONDS = 30.0
_WORKER_REAP_SECONDS = 1.0
_MAX_RESULTS = 100
_WORKER_MODULE = "pilotage.tools._ddgs_worker"
_TEST_HOOK_ENV = "PILOTAGE_DDGS_ALLOW_TEST_HOOKS"

# Test-only hook forwarded to the isolated worker. Production never sets it.
_test_hook: Optional[str] = None
_last_worker_proc: Optional[subprocess.Popen] = None


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

WEB_SEARCH_TOOL = Tool(
    name="web_search",
    group="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=handle_web_search,
    emoji="🔍",
    max_result_chars=100_000,
)
