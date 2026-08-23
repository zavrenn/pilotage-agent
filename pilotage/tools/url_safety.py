"""SSRF-safe HTTP connections for model-supplied image URLs.

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
import socket
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

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


def _proxy_is_configured() -> bool:
    return any(os.environ.get(name) for name in _PROXY_ENV_VARS)


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
