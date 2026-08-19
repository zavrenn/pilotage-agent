"""The OpenAI client, pointed at the Codex backend.

``https://chatgpt.com/backend-api/codex`` is behind Cloudflare, which returns
403 to anything that does not look like the Codex CLI. The originator header and
a codex-shaped User-Agent are what get us through; the account id comes out of
the access token's own claims.
"""

from __future__ import annotations

from typing import Dict

from openai import AsyncOpenAI

from .auth import Credentials, chatgpt_account_id

CODEX_USER_AGENT = "codex_cli_rs/0.0.0 (Pilotage Agent)"


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
    return AsyncOpenAI(
        api_key=credentials.access_token,
        base_url=credentials.base_url,
        default_headers=cloudflare_headers(credentials.access_token),
        timeout=timeout_seconds,
        max_retries=2,
    )
