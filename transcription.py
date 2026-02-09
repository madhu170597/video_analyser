"""
Transcribe video using OpenAI Whisper (open source). Returns segments with start/end times and text.
"""
import ssl
from pathlib import Path
import urllib.error
import whisper


def _load_whisper_model(model_size: str):
    """Load Whisper model; retry with unverified SSL if download fails (e.g. corporate proxy)."""
    try:
        return whisper.load_model(model_size)
    except urllib.error.URLError as e:
        if "certificate" in str(e).lower() or "CERTIFICATE_VERIFY_FAILED" in str(e):
            _orig = ssl.create_default_context
            ssl.create_default_context = ssl._create_unverified_context
            try:
                return whisper.load_model(model_size)
            finally:
                ssl.create_default_context = _orig
        raise


def transcribe_video(video_path: Path, model_size: str = "base") -> list[dict]:
    """
    Returns list of {"start": sec, "end": sec, "text": str}.
    Use model_size "base" for speed; "small" or "medium" for better accuracy.
    """
    model = _load_whisper_model(model_size)
    result = model.transcribe(str(video_path), word_timestamps=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip(),
        })
    return segments
