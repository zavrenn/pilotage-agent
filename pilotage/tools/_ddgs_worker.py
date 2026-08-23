"""Isolated DDGS search worker, adapted from Hermes.

The parent starts this module with ``python -m`` and communicates only through
one JSON request on stdin and one JSON envelope on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time

from .web import _MAX_RESULTS, _TEST_HOOK_ENV, _run_ddgs_search


def _run_test_hook(hook: str) -> dict:
    if hook == "sleep":
        time.sleep(30)
        return {"ok": False, "error": "sleep hook returned unexpectedly"}
    if hook == "success":
        return {
            "ok": True,
            "results": [
                {
                    "title": "Hit",
                    "url": "https://example.com",
                    "description": "body",
                    "position": 1,
                }
            ],
        }
    if hook == "empty":
        return {"ok": True, "results": []}
    if hook == "error":
        return {"ok": False, "error": "RuntimeError: boom"}
    return {"ok": False, "error": f"unknown test_hook: {hook!r}"}


def _write(envelope: dict) -> None:
    json.dump(envelope, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        _write({"ok": False, "error": f"invalid request: {exc}"})
        return 2

    hook = request.get("test_hook")
    if hook:
        if os.environ.get(_TEST_HOOK_ENV) != "1":
            _write({"ok": False, "error": "test_hook refused"})
            return 3
        envelope = _run_test_hook(str(hook))
        _write(envelope)
        return 0 if envelope.get("ok") else 1

    query = str(request.get("query") or "")
    safe_limit = min(
        max(1, int(request.get("safe_limit") or 1)),
        _MAX_RESULTS,
    )
    try:
        results = _run_ddgs_search(query, safe_limit)
        _write({"ok": True, "results": results})
        return 0
    except Exception as exc:  # noqa: BLE001 - returned to the parent as data
        _write({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
