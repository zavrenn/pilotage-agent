"""The resident HTTP transport used by the fixed Codex backend."""

from __future__ import annotations

import unittest

from pilotage.codex import auth
from pilotage.codex.client import build_client, build_http_client


class CodexHttpClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_pool_is_bounded_and_sse_reads_have_no_competing_timeout(self):
        client = build_http_client(timeout_seconds=300)
        try:
            pool = client._transport._pool
            self.assertEqual(pool._max_connections, 100)
            self.assertEqual(pool._max_keepalive_connections, 20)
            self.assertEqual(pool._keepalive_expiry, 20.0)
            self.assertIsNone(client.timeout.read)
            self.assertEqual(client.timeout.connect, 15.0)
        finally:
            await client.aclose()

    async def test_client_rejects_an_in_memory_legacy_endpoint(self):
        credentials = auth.Credentials(
            access_token="access",
            refresh_token="refresh",
            base_url="https://legacy.example/codex",
            last_refresh="",
        )

        with self.assertRaises(auth.AuthError):
            build_client(credentials, timeout_seconds=300)


if __name__ == "__main__":
    unittest.main()
