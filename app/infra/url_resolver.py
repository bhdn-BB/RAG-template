"""SSRF guard for URL ingestion. The page itself is downloaded and parsed by LangChain's
`WebBaseLoader` (see `app/services/parsing_service.py`); this module only decides whether a
URL may be fetched and resolves redirects safely before LangChain touches it.

Protections:
- only http/https, no credentials in the URL;
- every hostname (including each redirect hop) is resolved and rejected if it points to a
  private, loopback, link-local, multicast, reserved or unspecified address
  (blocks e.g. http://169.254.169.254/ cloud metadata, http://localhost:6333/ Qdrant);
- redirects are followed manually (max 5) so each hop is validated; LangChain then loads the
  final URL with redirects disabled.

Note: DNS can change between validation and connection (rebinding). For a hardened production
setup also restrict egress at the network level.
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx

from app.core.exceptions import UpstreamError, ValidationError

USER_AGENT = "Mozilla/5.0 (compatible; UkrainianRAG/0.1)"


async def _assert_public_host(host: str) -> None:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValidationError(f"Cannot resolve host '{host}'.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValidationError(f"URL host '{host}' resolves to a non-public address; refusing to fetch.")


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValidationError("Only absolute http(s) URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValidationError("URLs with credentials are not allowed.")
    return parsed.hostname


class SafeUrlResolver:
    def __init__(self, timeout: float = 20.0, max_redirects: int = 5, allow_private: bool = False) -> None:
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.allow_private = allow_private

    async def resolve(self, url: str) -> str:
        """Validate the URL and every redirect hop; return the final, safe-to-fetch URL."""
        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=False, headers={"User-Agent": USER_AGENT}
        ) as client:
            for _ in range(self.max_redirects + 1):
                host = _validate_url(url)
                if not self.allow_private:
                    await _assert_public_host(host)
                try:
                    # Stream so only headers are read; the body is downloaded later by LangChain.
                    async with client.stream("GET", url) as response:
                        if response.is_redirect:
                            url = urljoin(url, response.headers["location"])
                            continue
                        if response.status_code >= 400:
                            raise UpstreamError(f"Fetching {url} failed with HTTP {response.status_code}.")
                        return url
                except httpx.HTTPError as exc:
                    raise UpstreamError(f"Fetching {url} failed: {exc}") from exc
        raise ValidationError("Too many redirects.")
