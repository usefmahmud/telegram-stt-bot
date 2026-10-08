"""In-memory token store for pending summarize callbacks."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


@dataclass
class _Entry:
    text: str
    owner_id: int
    expires_at: float
    used: bool = False


class TranscriptStore:
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, _Entry] = {}

    def put(self, text: str, owner_id: int) -> str:
        self._purge()
        while len(self._entries) >= self._max:
            self._entries.pop(next(iter(self._entries)))
        token = secrets.token_urlsafe(9)
        self._entries[token] = _Entry(
            text=text,
            owner_id=owner_id,
            expires_at=time.monotonic() + self._ttl,
        )
        return token

    def claim(self, token: str, user_id: int) -> tuple[str | None, str | None]:
        self._purge()
        entry = self._entries.get(token)
        if entry is None:
            return None, "expired"
        if entry.used:
            del self._entries[token]
            return None, "used"
        if entry.owner_id != user_id:
            return None, "not_owner"
        entry.used = True
        return entry.text, None

    def release(self, token: str, text: str, owner_id: int) -> None:
        self._entries[token] = _Entry(
            text=text,
            owner_id=owner_id,
            expires_at=time.monotonic() + self._ttl,
        )

    def _purge(self) -> None:
        now = time.monotonic()
        for token in [t for t, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[token]
