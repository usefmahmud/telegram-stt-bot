"""Local QwenCleo ASR behind a thread pool, with ffmpeg conversion."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when ffmpeg conversion or ASR inference fails."""


def probe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def convert_to_wav(src: Path, dst: Path) -> None:
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(src),
                "-ar", "16000", "-ac", "1",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranscriptionError(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise TranscriptionError(f"ffmpeg failed: {proc.stderr.strip()[:500]}")


class Transcriber:
    def __init__(
        self,
        device: str = "cpu",
        max_new_tokens: int = 2048,
        language: str | None = "Arabic",
        max_workers: int = 2,
    ) -> None:
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._language = language
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="asr")
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from qwencleo_asr import QwenCleoASR

                    logger.info("loading ASR model on %s", self._device)
                    self._model = QwenCleoASR(
                        device=self._device,
                        max_new_tokens=self._max_new_tokens,
                        default_language=self._language,
                    )
        return self._model

    def _transcribe_blocking(self, src: Path) -> tuple[str, str | None]:
        with tempfile.TemporaryDirectory(prefix="stt-") as tmp:
            wav = Path(tmp) / "audio.wav"
            convert_to_wav(src, wav)
            model = self._ensure_model()
            with self._infer_lock:
                result = model.transcribe(str(wav), language=self._language)
        text = (result.text or "").strip()
        language = getattr(result, "language", None)
        return text, language

    async def transcribe(self, src: Path) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._transcribe_blocking, src)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(str(exc)) from exc
