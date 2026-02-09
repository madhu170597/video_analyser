"""
Download YouTube video using yt-dlp (open source). Saves video for analysis and Whisper.
"""
from pathlib import Path
import subprocess
import sys

from config import VIDEOS_DIR


def download_video(url: str, output_dir: Path | None = None) -> tuple[Path, float, str | None]:
    """
    Download video from URL. Returns (path_to_video_file, duration_sec, video_title_or_None).
    Title is used as context for Gemini to identify characters (e.g. "Avengers: Endgame - Official Trailer").
    """
    output_dir = output_dir or VIDEOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = "video"
    duration_sec = 0.0
    video_title: str | None = None

    # Get metadata first (no download). Timeout 90s for slow networks / YouTube.
    # Title is used as context for Gemini to identify characters (e.g. movie trailer).
    cmd_info = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--print", "%(id)s\n%(duration)s\n%(title)s",
        "--no-download",
        "--no-warnings",
        "--socket-timeout", "30",
        url,
    ]
    try:
        info = subprocess.run(cmd_info, capture_output=True, text=True, timeout=90)
        if info.returncode == 0:
            lines = (info.stdout or "").strip().splitlines()
            if lines:
                video_id = lines[0].strip()
            if len(lines) > 1:
                try:
                    duration_sec = float(lines[1].strip())
                except ValueError:
                    pass
            if len(lines) > 2:
                video_title = lines[2].strip() or None
    except subprocess.TimeoutExpired:
        # Fallback: download with generic template; yt-dlp will infer id from URL
        out_tpl = str(output_dir / "%(id)s.%(ext)s")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "--format", "best[ext=mp4]/best",
            "--output", out_tpl,
            "--no-warnings",
            "--socket-timeout", "30",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                "yt-dlp metadata fetch timed out and download failed. Try again or check network. "
                f"stderr: {result.stderr or result.stdout}"
            )
        candidates = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp4", ".mkv", ".webm")]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("yt-dlp did not create a file in " + str(output_dir))
        video_path = candidates[0]
        video_id = video_path.stem
        if not duration_sec:
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                duration_sec = frame_count / fps if fps else 0
            except Exception:
                pass
        return video_path, duration_sec, None  # no title on timeout fallback

    out_tpl = str(output_dir / f"{video_id}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "best[ext=mp4]/best",
        "--output", out_tpl,
        "--no-warnings",
        "--socket-timeout", "30",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr or result.stdout}")

    video_path = output_dir / f"{video_id}.mp4"
    if not video_path.exists():
        candidates = [p for p in output_dir.glob(f"{video_id}.*") if p.is_file()]
        video_path = candidates[0] if candidates else None
    if not video_path or not video_path.exists():
        raise FileNotFoundError("yt-dlp did not create a file in " + str(output_dir))

    if not duration_sec:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            duration_sec = frame_count / fps if fps else 0
        except Exception:
            duration_sec = 0
    return video_path, duration_sec, video_title
