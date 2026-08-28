"""The cross-process lock on the shared ChatGPT credentials.

Two agent profiles can share one sign-in. The refresh token is single-use and
rotates, so two processes refreshing at once log each other out. These tests
use a real second process, because a lock that only orders coroutines inside
one process would pass anything weaker.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

from pilotage.codex import auth

REPO_ROOT = Path(__file__).resolve().parent.parent

# Takes the lock, says so, and holds it.
HOLDER = """
import pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from pilotage.codex import auth
with auth.credentials_lock(pathlib.Path(sys.argv[2]), timeout_seconds=30.0):
    pathlib.Path(sys.argv[3]).write_text("held")
    time.sleep(float(sys.argv[4]))
"""


def _token(expires_in: float) -> str:
    """A JWT we only ever read the expiry from."""
    claims = json.dumps({"exp": time.time() + expires_in}).encode("utf-8")
    payload = base64.urlsafe_b64encode(claims).decode("ascii").rstrip("=")
    return f"header.{payload}.signature"


def _write(path: Path, access_token: str, refresh_token: str = "refresh-1") -> None:
    auth.write_credentials(
        path,
        auth.Credentials(
            access_token=access_token,
            refresh_token=refresh_token,
            base_url=auth.DEFAULT_CODEX_BASE_URL,
            last_refresh="",
        ),
    )


class LockBudgetTests(unittest.TestCase):
    def test_the_wait_outlasts_the_refresh_it_waits_for(self):
        """A slow but healthy holder must not look like a stuck one.

        httpx applies its timeout per phase — connect, write, read — so one
        refresh can legitimately take three times the configured timeout. A
        waiter that gives up sooner fails a turn it could have served.
        """
        worst_case = 3 * auth.REFRESH_TIMEOUT_SECONDS
        self.assertGreater(auth.CREDENTIALS_LOCK_TIMEOUT_SECONDS, worst_case)


class EndpointValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "codex-auth.json"

    def test_legacy_stored_endpoint_is_rejected_before_use(self):
        self.path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": "access",
                        "refresh_token": "refresh",
                    },
                    "base_url": "https://legacy.example/codex",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(auth.AuthError) as caught:
            auth.read_credentials(self.path)

        self.assertEqual(caught.exception.code, "codex_auth_base_url_mismatch")
        self.assertTrue(caught.exception.relogin_required)

    def test_credentials_can_only_be_written_for_the_chatgpt_codex_route(self):
        credentials = auth.Credentials(
            access_token="access",
            refresh_token="refresh",
            base_url="https://legacy.example/codex",
            last_refresh="",
        )

        with self.assertRaises(auth.AuthError):
            auth.write_credentials(self.path, credentials)

        self.assertFalse(self.path.exists())

    def test_trailing_slash_normalizes_to_the_exact_route(self):
        self.assertEqual(
            auth.validated_codex_base_url(auth.DEFAULT_CODEX_BASE_URL + "/"),
            auth.DEFAULT_CODEX_BASE_URL,
        )


class CrossProcessLockTests(unittest.TestCase):
    def setUp(self):
        # Windows releases a terminated process's file handles a moment late.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "codex-auth.json"
        self.ready = Path(self._tmp.name) / "held"

    @contextmanager
    def _held_elsewhere(self, seconds: float):
        child = subprocess.Popen(
            [sys.executable, "-c", HOLDER, str(REPO_ROOT), str(self.path), str(self.ready), str(seconds)]
        )
        try:
            deadline = time.monotonic() + 20.0
            while not self.ready.exists():
                if child.poll() is not None:
                    self.fail("the holder process died before taking the lock")
                if time.monotonic() > deadline:
                    self.fail("the holder process never took the lock")
                time.sleep(0.02)
            yield
        finally:
            child.terminate()
            child.wait(timeout=10)

    def test_another_process_is_kept_out(self):
        with self._held_elsewhere(30.0):
            started = time.monotonic()
            with self.assertRaises(auth.AuthError) as caught:
                with auth.credentials_lock(self.path, timeout_seconds=1.0):
                    self.fail("two processes held the credentials lock at once")
        self.assertEqual(caught.exception.code, "codex_auth_lock_timeout")
        self.assertGreaterEqual(time.monotonic() - started, 1.0)

    def test_the_lock_is_released_when_the_holder_finishes(self):
        with self._held_elsewhere(0.5):
            with auth.credentials_lock(self.path, timeout_seconds=20.0):
                pass  # Waited for the holder rather than failing.

    def test_the_credentials_file_is_never_the_locked_one(self):
        _write(self.path, _token(3600))
        with auth.credentials_lock(self.path):
            self.assertTrue(self.path.exists())
        self.assertTrue(self.path.with_suffix(".json.lock").exists())
        self.assertEqual(auth.read_credentials(self.path).refresh_token, "refresh-1")


class ResolveCredentialsTests(unittest.TestCase):
    def setUp(self):
        # Windows releases a terminated process's file handles a moment late.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "codex-auth.json"
        self.refreshes = 0

        def _refresh(credentials: auth.Credentials) -> auth.Credentials:
            self.refreshes += 1
            return auth.Credentials(
                access_token=_token(3600),
                refresh_token="refresh-mine",
                base_url=credentials.base_url,
                last_refresh="now",
            )

        self._real_refresh = auth.refresh_credentials
        auth.refresh_credentials = _refresh
        self.addCleanup(setattr, auth, "refresh_credentials", self._real_refresh)

    def test_a_live_token_is_used_as_is(self):
        _write(self.path, _token(3600))
        auth.resolve_credentials(self.path)
        self.assertEqual(self.refreshes, 0)

    def test_an_expiring_token_is_refreshed(self):
        _write(self.path, _token(10))
        resolved = auth.resolve_credentials(self.path)
        self.assertEqual(self.refreshes, 1)
        self.assertEqual(resolved.refresh_token, "refresh-mine")

    def test_a_refresh_by_another_process_is_adopted(self):
        """Whoever waited uses the token the other one just got.

        Refreshing again would spend a refresh token that has already been
        rotated away, and log both agents out.
        """
        _write(self.path, _token(10))
        real_lock = auth.credentials_lock

        @contextmanager
        def _lock_and_meanwhile(path, timeout_seconds=auth.CREDENTIALS_LOCK_TIMEOUT_SECONDS):
            with real_lock(path, timeout_seconds):
                # The other process refreshed while we were waiting for it.
                _write(path, _token(3600), refresh_token="refresh-theirs")
                yield

        auth.credentials_lock = _lock_and_meanwhile
        self.addCleanup(setattr, auth, "credentials_lock", real_lock)

        resolved = auth.resolve_credentials(self.path)
        self.assertEqual(self.refreshes, 0)
        self.assertEqual(resolved.refresh_token, "refresh-theirs")


if __name__ == "__main__":
    unittest.main()
