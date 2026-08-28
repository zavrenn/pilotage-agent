"""The OpenAI client, pointed at the Codex backend.

``https://chatgpt.com/backend-api/codex`` is behind Cloudflare, which returns
403 to anything that does not look like the Codex CLI. The originator header and
a codex-shaped User-Agent are what get us through; the account id comes out of
the access token's own claims.
"""

from __future__ import annotations

from typing import Dict

import httpx
from openai import AsyncOpenAI

from .auth import Credentials, chatgpt_account_id, validated_codex_base_url

CODEX_USER_AGENT = "codex_cli_rs/0.0.0 (Pilotage Agent)"


def build_http_client(*, timeout_seconds: float) -> httpx.AsyncClient:
    """Hermes' resident SSE pool, narrowed to Pilotage's fixed backend."""

    timeout = max(1.0, float(timeout_seconds))
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=20.0,
        ),
        timeout=httpx.Timeout(
            connect=min(15.0, timeout),
            read=None,
            write=min(15.0, timeout),
            pool=min(10.0, timeout),
        ),
    )


def cloudflare_headers(access_token: str) -> Dict[str, str]:
    headers = {
        "User-Agent": CODEX_USER_AGENT,
        "originator": "codex_cli_rs",
    }
    account_id = chatgpt_account_id(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def build_client(credentials: Credentials, *, timeout_seconds: float) -> AsyncOpenAI:
    base_url = validated_codex_base_url(credentials.base_url)
    return AsyncOpenAI(
        api_key=credentials.access_token,
        base_url=base_url,
        default_headers=cloudflare_headers(credentials.access_token),
        timeout=timeout_seconds,
        max_retries=2,
        http_client=build_http_client(timeout_seconds=timeout_seconds),
    )


__all__ = ["build_client", "build_http_client", "cloudflare_headers"]
