"""OpenAI-compatible chat client for transcript summaries."""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You summarize speech-to-text transcripts. Summarize the transcript the "
    "user sends. Be concise and keep key facts, decisions and action items. "
    "Reply in the same language as the transcript."
)


class Summarizer:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_chars: int = 60000,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._model = model
        self._max_chars = max_chars

    async def summarize(self, text: str) -> str:
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + "\n…[truncated]"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("empty summary from model")
        return content.strip()
