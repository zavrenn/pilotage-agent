"""Codex OAuth — device-code login, token storage, refresh.

Authentication is a ChatGPT subscription, not an API key. The access token is a
short-lived JWT; the refresh token is single-use and rotates on every refresh.
That is why we keep our own store and never share ``~/.codex/auth.json``: two
processes refreshing the same refresh token invalidate each other.

The login flow is device-code, not a loopback redirect, because agents run
headless in LXC containers with no browser and no reachable localhost.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

CODEX_ISSUER = "https://auth.openai.com"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = f"{CODEX_ISSUER}/oauth/token"
CODEX_DEVICE_URL = f"{CODEX_ISSUER}/codex/device"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Refresh this many seconds before the JWT actually expires.
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

_HTTP_TIMEOUT = httpx.Timeout(20.0)
_USER_AGENT = "pilotage-agent/0.1"


class AuthError(Exception):
    """Something is wrong with the Codex credentials."""

    def __init__(self, message: str, *, code: str = "codex_auth_error", relogin_required: bool = False):
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


@dataclass(frozen=True)
class Credentials:
    access_token: str
    refresh_token: str
    base_url: str
    last_refresh: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------


def read_credentials(path: Path) -> Credentials:
    if not path.exists():
        raise AuthError(
            f"No Codex credentials at {path}. Run `pilotage login` first.",
            code="codex_auth_missing",
            relogin_required=True,
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuthError(
            f"Codex credentials at {path} are unreadable: {exc}",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        ) from exc

    tokens = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(tokens, dict):
        raise AuthError(
            f"Codex credentials at {path} have an unexpected shape.",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    access_token = str(tokens.get("access_token") or "")
    refresh_token = str(tokens.get("refresh_token") or "")
    if not access_token:
        raise AuthError(
            "Codex credentials have no access token.",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    if not refresh_token:
        raise AuthError(
            "Codex credentials have no refresh token.",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    return Credentials(
        access_token=access_token,
        refresh_token=refresh_token,
        base_url=str(data.get("base_url") or DEFAULT_CODEX_BASE_URL),
        last_refresh=str(data.get("last_refresh") or ""),
    )


def write_credentials(path: Path, credentials: Credentials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
        },
        "base_url": credentials.base_url,
        "last_refresh": credentials.last_refresh,
        "auth_mode": "chatgpt",
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Best effort; Windows has no equivalent mode.


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def decode_jwt_claims(token: str) -> Dict[str, Any]:
    """Read a JWT payload without verifying it. Never used for authorization."""
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def access_token_is_expiring(token: str, skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    claims = decode_jwt_claims(token)
    exp = claims.get("exp")
    if exp is None:
        # No expiry claim we can read — refresh rather than risk a 401 mid-turn.
        return True
    try:
        return float(exp) <= time.time() + max(0, int(skew_seconds))
    except (TypeError, ValueError):
        return True


def chatgpt_account_id(token: str) -> str:
    claims = decode_jwt_claims(token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        return str(auth_claims.get("chatgpt_account_id") or "")
    return ""


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def refresh_credentials(credentials: Credentials) -> Credentials:
    """Exchange the refresh token for a new access token.

    The refresh token rotates: whatever comes back replaces the stored one, and
    the old one is dead the moment this succeeds.
    """
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            response = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": credentials.refresh_token,
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": _USER_AGENT,
                },
            )
    except httpx.HTTPError as exc:
        raise AuthError(f"Codex token refresh failed: {exc}", code="codex_refresh_network") from exc

    if response.status_code == 429:
        # Quota, not credentials. The stored tokens are still good.
        raise AuthError(
            "OpenAI is rate-limiting the token refresh (HTTP 429). The credentials are still valid.",
            code="codex_rate_limited",
            relogin_required=False,
        )

    if response.status_code != 200:
        raise _refresh_error(response)

    body = response.json()
    access_token = str(body.get("access_token") or "")
    if not access_token:
        raise AuthError(
            "Codex token refresh returned no access token.",
            code="codex_refresh_no_access_token",
            relogin_required=True,
        )

    return Credentials(
        access_token=access_token,
        refresh_token=str(body.get("refresh_token") or credentials.refresh_token),
        base_url=credentials.base_url,
        last_refresh=_now_iso(),
    )


def _refresh_error(response: httpx.Response) -> AuthError:
    """Classify a failed refresh. Two error shapes come back from this endpoint."""
    code = ""
    description = ""
    try:
        body = response.json()
    except ValueError:
        body = {}

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):  # OpenAI shape
            code = str(error.get("code") or error.get("type") or "")
            description = str(error.get("message") or "")
        elif isinstance(error, str):  # OAuth shape
            code = error
            description = str(body.get("error_description") or "")

    relogin = code in {"invalid_grant", "invalid_token", "invalid_request"} or response.status_code in {401, 403}

    if code == "refresh_token_reused":
        return AuthError(
            "The Codex refresh token was consumed by another client. "
            "Run `pilotage login` again, and make sure only one process holds these credentials.",
            code="codex_refresh_token_reused",
            relogin_required=True,
        )

    detail = f" ({code}: {description})" if code or description else ""
    return AuthError(
        f"Codex token refresh returned HTTP {response.status_code}{detail}.",
        code="codex_refresh_failed",
        relogin_required=relogin,
    )


def resolve_credentials(path: Path, *, force_refresh: bool = False) -> Credentials:
    """Read the stored credentials, refreshing them if the access token is stale."""
    credentials = read_credentials(path)
    if not force_refresh and not access_token_is_expiring(credentials.access_token):
        return credentials
    refreshed = refresh_credentials(credentials)
    write_credentials(path, refreshed)
    return refreshed


# ---------------------------------------------------------------------------
# Device-code login
# ---------------------------------------------------------------------------


def _retry_after_seconds(response: httpx.Response) -> Optional[int]:
    raw = response.headers.get("Retry-After", "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def device_code_login(*, print_fn=print) -> Credentials:
    """Run the OpenAI device-code flow and return fresh credentials."""
    with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
        device = _request_device_code(client, print_fn=print_fn)
        user_code = device["user_code"]
        device_auth_id = device["device_auth_id"]
        poll_interval = max(3, int(device.get("interval") or 5))

        print_fn("")
        print_fn(f"  1. Open  {CODEX_DEVICE_URL}")
        print_fn(f"  2. Enter {user_code}")
        print_fn("")
        print_fn("Waiting for sign-in... (Ctrl+C to cancel)")

        authorization = _poll_for_authorization(
            client, device_auth_id=device_auth_id, user_code=user_code, interval=poll_interval
        )
        return _exchange_authorization_code(client, authorization)


def _request_device_code(client: httpx.Client, *, print_fn) -> Dict[str, Any]:
    response = None
    for attempt in range(1, 5):
        try:
            response = client.post(
                f"{CODEX_ISSUER}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"Failed to request a device code: {exc}", code="device_code_request_failed") from exc

        if response.status_code != 429:
            break
        if attempt < 4:
            delay = _retry_after_seconds(response) or 2**attempt
            delay = max(1, min(delay, 60))
            print_fn(f"OpenAI is rate-limiting login (429); retrying in {delay}s...")
            time.sleep(delay)

    if response is None or response.status_code == 429:
        raise AuthError(
            "OpenAI is rate-limiting Codex login (HTTP 429). Wait a minute and try again.",
            code="codex_rate_limited",
        )
    if response.status_code != 200:
        raise AuthError(
            f"Device code request returned HTTP {response.status_code}.",
            code="device_code_request_error",
        )

    device = response.json()
    if not device.get("user_code") or not device.get("device_auth_id"):
        raise AuthError("Device code response is missing required fields.", code="device_code_incomplete")
    return device


def _poll_for_authorization(
    client: httpx.Client, *, device_auth_id: str, user_code: str, interval: int
) -> Dict[str, Any]:
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        time.sleep(interval)
        response = client.post(
            f"{CODEX_ISSUER}/api/accounts/deviceauth/token",
            json={"device_auth_id": device_auth_id, "user_code": user_code},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            return response.json()
        if response.status_code in {403, 404}:
            continue  # Sign-in not finished yet.
        raise AuthError(
            f"Device auth polling returned HTTP {response.status_code}.",
            code="device_code_poll_error",
        )
    raise AuthError("Login timed out after 15 minutes.", code="device_code_timeout")


def _exchange_authorization_code(client: httpx.Client, authorization: Dict[str, Any]) -> Credentials:
    code = str(authorization.get("authorization_code") or "")
    verifier = str(authorization.get("code_verifier") or "")
    if not code or not verifier:
        raise AuthError(
            "Device auth response is missing the authorization code.",
            code="device_code_incomplete_exchange",
        )

    try:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": f"{CODEX_ISSUER}/deviceauth/callback",
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except httpx.HTTPError as exc:
        raise AuthError(f"Token exchange failed: {exc}", code="token_exchange_failed") from exc

    if response.status_code == 429:
        raise AuthError(
            "OpenAI is rate-limiting the token exchange (HTTP 429). Wait a minute and try again.",
            code="codex_rate_limited",
        )
    if response.status_code != 200:
        raise AuthError(
            f"Token exchange returned HTTP {response.status_code}.",
            code="token_exchange_error",
        )

    tokens = response.json()
    access_token = str(tokens.get("access_token") or "")
    if not access_token:
        raise AuthError("Token exchange returned no access token.", code="token_exchange_no_access_token")

    base_url = os.environ.get("PILOTAGE_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL
    return Credentials(
        access_token=access_token,
        refresh_token=str(tokens.get("refresh_token") or ""),
        base_url=base_url,
        last_refresh=_now_iso(),
    )
