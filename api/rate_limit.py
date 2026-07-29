"""Shared rate limiter for the API, keyed by client IP."""

from fastapi import Request
from slowapi import Limiter


def _client_ip(request: Request) -> str:
    """Resolve the real client IP, respecting X-Forwarded-For behind Railway's proxy."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip)
