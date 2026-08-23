"""SSRF-safe HTTP connections for model-supplied URLs.

Ported from Hermes tools/url_safety.py and narrowed to Genesis' current
contract. Pilotage has no private-URL override or browser policy layer, so
private, loopback, link-local, reserved, multicast, CGNAT, and cloud metadata
targets always fail closed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from typing import Any, Optional
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urljoin,
    urlparse,
    urlsplit,
    urlunsplit,
)

logger = logging.getLogger(__name__)

_PROXY_ENV_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)
_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
    }
)
_ALWAYS_BLOCKED_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("169.254.170.2"),
        ipaddress.ip_address("169.254.169.253"),
        ipaddress.ip_address("fd00:ec2::254"),
        ipaddress.ip_address("100.100.100.200"),
        ipaddress.ip_address("::ffff:169.254.169.254"),
        ipaddress.ip_address("::ffff:169.254.170.2"),
        ipaddress.ip_address("::ffff:169.254.169.253"),
        ipaddress.ip_address("::ffff:100.100.100.200"),
    }
)
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::ffff:169.254.0.0/112"),
)
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_MAX_SSRF_CONNECT_IPS = 8

# Hermes' known API-token prefixes. Web extraction hands its URL to a
# third-party reader, so a recognizable secret in any URL component fails
# closed before that handoff.
_TOKEN_PREFIX_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{10,}",
    r"ghp_[A-Za-z0-9]{10,}",
    r"github_pat_[A-Za-z0-9_]{10,}",
    r"gho_[A-Za-z0-9]{10,}",
    r"ghu_[A-Za-z0-9]{10,}",
    r"ghs_[A-Za-z0-9]{10,}",
    r"ghr_[A-Za-z0-9]{10,}",
    r"xapp-\d+-[A-Za-z0-9-]{10,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"AIza[A-Za-z0-9_-]{30,}",
    r"pplx-[A-Za-z0-9]{10,}",
    r"fal_[A-Za-z0-9_-]{10,}",
    r"fc-[A-Za-z0-9]{10,}",
    r"bb_live_[A-Za-z0-9_-]{10,}",
    r"gAAAA[A-Za-z0-9_=-]{20,}",
    r"AKIA[A-Z0-9]{16}",
    r"sk_live_[A-Za-z0-9]{10,}",
    r"sk_test_[A-Za-z0-9]{10,}",
    r"rk_live_[A-Za-z0-9]{10,}",
    r"SG\.[A-Za-z0-9_-]{10,}",
    r"hf_[A-Za-z0-9]{10,}",
    r"r8_[A-Za-z0-9]{10,}",
    r"npm_[A-Za-z0-9]{10,}",
    r"pypi-[A-Za-z0-9_-]{10,}",
    r"dop_v1_[A-Za-z0-9]{10,}",
    r"doo_v1_[A-Za-z0-9]{10,}",
    r"am_[A-Za-z0-9_-]{10,}",
    r"sk_[A-Za-z0-9_]{10,}",
    r"tvly-[A-Za-z0-9]{10,}",
    r"exa_[A-Za-z0-9]{10,}",
    r"gsk_[A-Za-z0-9]{10,}",
    r"syt_[A-Za-z0-9]{10,}",
    r"retaindb_[A-Za-z0-9]{10,}",
    r"hsk-[A-Za-z0-9]{10,}",
    r"mem0_[A-Za-z0-9]{10,}",
    r"brv_[A-Za-z0-9]{10,}",
    r"xai-[A-Za-z0-9]{30,}",
    r"ntn_[A-Za-z0-9]{10,}",
    r"fw-[A-Za-z0-9]{30,}",
    r"fw_[A-Za-z0-9]{30,}",
    r"fpk_[A-Za-z0-9]{30,}",
    r"glpat-[A-Za-z0-9_\-]{10,}",
    r"gloas-[A-Za-z0-9_\-]{10,}",
    r"gldt-[A-Za-z0-9_\-]{10,}",
    r"glrt-[A-Za-z0-9_.\-]{10,}",
    r"glrtr-[A-Za-z0-9_.\-]{10,}",
    r"glcbt-[A-Za-z0-9_\-]{10,}",
    r"glptt-[A-Za-z0-9_\-]{10,}",
    r"glft-[A-Za-z0-9_\-]{10,}",
    r"glimt-[A-Za-z0-9_\-]{10,}",
    r"glagent-[A-Za-z0-9_\-]{10,}",
    r"glsoat-[A-Za-z0-9_\-]{10,}",
    r"glffct-[A-Za-z0-9_\-]{10,}",
    r"glwt-[A-Za-z0-9_\-]{10,}",
    r"GR1348941[A-Za-z0-9_\-]{10,}",
    r"pk-lf-[A-Za-z0-9\-]{8,}",
)
_TOKEN_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_TOKEN_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

_SENSITIVE_QUERY_PARAM_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "awsaccesskeyid",
        "client_secret",
        "credential",
        "credentials",
        "jwt",
        "password",
        "passwd",
        "secret",
        "session_id",
        "signature",
        "token",
        "x_amz_security_token",
        "x_amz_signature",
        "x-amz-security-token",
        "x-amz-signature",
    }
)


def _proxy_is_configured() -> bool:
    return any(os.environ.get(name) for name in _PROXY_ENV_VARS)


def normalize_url_for_request(url: str) -> str:
    """Return an ASCII-safe HTTP URL for model-supplied URL tools."""
    if not isinstance(url, str):
        return url
    raw = url.strip()
    if not raw:
        return raw
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme.lower() not in {"http", "https"}:
        return raw

    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)

    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    fragment = quote(parsed.fragment, safe="/%:@!$&'()*+,;=?")
    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


def has_url_credentials(url: str) -> bool:
    """Return whether an HTTP URL embeds credentials in its authority."""
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url.strip())
        return parsed.scheme.lower() in {"http", "https"} and (
            parsed.username is not None or parsed.password is not None
        )
    except ValueError:
        return False


def sensitive_query_param_name(url: str) -> Optional[str]:
    """Return the first credential-bearing query parameter in a URL."""
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


def contains_known_secret(url: str) -> bool:
    """Return whether a raw or percent-decoded URL contains a known token."""
    if not isinstance(url, str):
        return False
    return bool(_TOKEN_PREFIX_RE.search(url) or _TOKEN_PREFIX_RE.search(unquote(url)))


def _is_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        embedded = ip.ipv4_mapped
        return (
            embedded.is_private
            or embedded.is_loopback
            or embedded.is_link_local
            or embedded.is_reserved
            or embedded.is_multicast
            or embedded.is_unspecified
            or embedded in _CGNAT_NETWORK
        )
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or ip in _CGNAT_NETWORK
    )


def _resolved_addresses(hostname: str, port: Optional[int]) -> list[str]:
    info = socket.getaddrinfo(
        hostname,
        port,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
    )
    addresses: list[str] = []
    for _family, _, _, _, sockaddr in info:
        raw = str(sockaddr[0]).split("%", 1)[0]
        if raw not in addresses:
            addresses.append(raw)
    return addresses


def _address_is_allowed(hostname: str, raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        logger.warning(
            "Blocked request: unparseable IP %r for %s", raw, hostname
        )
        return False
    if ip in _ALWAYS_BLOCKED_IPS or any(
        ip in network for network in _ALWAYS_BLOCKED_NETWORKS
    ):
        logger.warning(
            "Blocked request to cloud metadata address: %s -> %s",
            hostname,
            raw,
        )
        return False
    if _is_blocked_ip(ip):
        logger.warning(
            "Blocked request to private/internal address: %s -> %s",
            hostname,
            raw,
        )
        return False
    return True


def is_safe_url(url: str) -> bool:
    """Resolve and reject non-public HTTP(S) targets. Fail closed."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in {"http", "https"} or not hostname:
            return False
        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        try:
            addresses = _resolved_addresses(hostname, parsed.port)
        except socket.gaierror:
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                literal_ip = False
            else:
                literal_ip = True
            if not literal_ip and _proxy_is_configured():
                return True
            logger.warning(
                "Blocked request: DNS resolution failed for %s", hostname
            )
            return False

        return bool(addresses) and all(
            _address_is_allowed(hostname, address)
            for address in addresses
        )
    except Exception as exc:
        logger.warning("Blocked request: URL safety check failed: %s", exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    return await asyncio.to_thread(is_safe_url, url)


class SSRFConnectionBlocked(ValueError):
    """A connect-time DNS answer violated the URL safety policy."""


def _resolved_http_connect_ips(
    host: str,
    port: int,
) -> list[str]:
    hostname = (host or "").strip().lower().rstrip(".")
    if not hostname:
        raise SSRFConnectionBlocked("Blocked request with empty hostname")
    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFConnectionBlocked(
            f"Blocked request to internal hostname: {hostname}"
        )
    try:
        addresses = _resolved_addresses(hostname, port)
    except socket.gaierror as exc:
        raise SSRFConnectionBlocked(
            f"Blocked request: DNS resolution failed for {hostname}"
        ) from exc

    safe: list[str] = []
    for address in addresses:
        if not _address_is_allowed(hostname, address):
            raise SSRFConnectionBlocked(
                f"Blocked request to private/internal address: "
                f"{hostname} -> {address}"
            )
        if address not in safe and len(safe) < _MAX_SSRF_CONNECT_IPS:
            safe.append(address)
    if not safe:
        raise SSRFConnectionBlocked(
            f"Blocked request: DNS returned no results for {hostname}"
        )
    return safe


class _SSRFGuardedAsyncNetworkBackend:
    def __init__(self):
        from httpcore._backends.auto import AutoBackend

        self._backend = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> Any:
        import httpcore

        ips = await asyncio.to_thread(
            _resolved_http_connect_ips,
            host,
            port,
        )
        last_error: Optional[Exception] = None
        for ip in ips:
            try:
                return await self._backend.connect_tcp(
                    ip,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise SSRFConnectionBlocked(
            f"Blocked request: DNS returned no usable IPs for {host}"
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> Any:
        raise SSRFConnectionBlocked(
            "Blocked Unix socket connection in SSRF-safe transport"
        )

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def _install_ssrf_guard_on_async_client(client: Any) -> None:
    transport = getattr(client, "__dict__", {}).get("_transport")
    state = getattr(transport, "__dict__", {})
    pool = state.get("_pool")
    if transport is None or pool is None or not hasattr(
        pool, "_network_backend"
    ):
        raise SSRFConnectionBlocked(
            "Unsupported async HTTP transport cannot be made SSRF-safe"
        )
    pool._network_backend = _SSRFGuardedAsyncNetworkBackend()


def create_ssrf_safe_async_client(**kwargs: Any) -> Any:
    """Create Hermes' connect-time guarded async httpx client."""
    import httpx

    client = httpx.AsyncClient(**kwargs)
    _install_ssrf_guard_on_async_client(client)
    return client


def redirect_target_from_response(response: Any) -> Optional[str]:
    """Resolve a redirect target from its Location header."""
    if not getattr(response, "is_redirect", False):
        return None
    headers = getattr(response, "headers", {}) or {}
    location = headers.get("location")
    if location:
        return urljoin(str(getattr(response, "url", "")), str(location))
    next_request = getattr(response, "next_request", None)
    return str(next_request.url) if next_request else None
