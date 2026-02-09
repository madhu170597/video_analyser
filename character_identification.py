"""
Character Identification: Extract frames and identify all characters using Vertex AI in parallel.
Uses: Vertex AI (Google Cloud) with parallel frame processing (max 10 workers).
Output: Character appearance database with frame counts and timestamps.
Deduplicates actors by name and aggregates their on-screen character names.
"""
import json
import base64
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Optional
import warnings
from config import VERTEX_AI_PROJECT, VERTEX_AI_LOCATION, VERTEX_AI_MODEL
import cv2

# Vertex AI setup
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
except ImportError:
    print("Error: vertexai SDK not installed. Install with: pip install google-cloud-aiplatform")
    exit(1)

# Configuration
VERTEX_AI_PROJECT = VERTEX_AI_PROJECT
VERTEX_AI_LOCATION = VERTEX_AI_LOCATION
VERTEX_AI_MODEL = 'gemini-2.5-flash'  # Updated to latest model for better image understanding
MAX_PARALLEL_WORKERS = 3  # Reduced from 10 to avoid throttling on free tier
FRAME_SAMPLE_INTERVAL_SEC = 1.0


def _init_vertex_ai():
    """Initialize Vertex AI with project and location."""
    vertexai.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
    return GenerativeModel(VERTEX_AI_MODEL)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = FRAME_SAMPLE_INTERVAL_SEC,
) -> list[tuple[float, Path]]:
    """Extract frames from video at fixed intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    interval_frames = max(1, int(round(fps * interval_sec)))
    
    results = []
    frame_idx = 0
    video_name = video_path.stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            t_sec = frame_idx / fps
            out_name = f"{video_name}_t{t_sec:.1f}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            results.append((t_sec, out_path))
        frame_idx += 1
    
    cap.release()
    return results


def _upscale_frame(frame_path: Path) -> Path:
    """
    Upscale frame by 2x for better character recognition.
    Saves upscaled frame to temp location and returns the path.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return frame_path
    
    h, w = frame.shape[:2]

    # If the frame is already large, avoid upscaling to reduce payload and latency.
    MAX_DIMENSION_FOR_UPSCALE = 800  # pixels; if either side > this, skip upscaling
    if max(h, w) > MAX_DIMENSION_FOR_UPSCALE:
        # Also ensure we don't send extremely large images: cap max dimension
        MAX_DIMENSION = 1200
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            temp_path = frame_path.parent / f"{frame_path.stem}_resized.jpg"
            cv2.imwrite(str(temp_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return temp_path
        return frame_path

    # Upscale 2x using cubic interpolation (better quality than bilinear) for small frames
    upscaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Save upscaled frame to temp location with slightly lower quality to reduce size
    temp_path = frame_path.parent / f"{frame_path.stem}_upscaled.jpg"
    cv2.imwrite(str(temp_path), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return temp_path


def _frame_to_base64(frame_path: Path) -> str:
    """Convert frame image to base64."""
    with open(frame_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _extract_actor_name(full_entry: str) -> Optional[str]:
    """Extract actor name from 'Actor Name (Character1, Character2)' format."""
    full_entry = full_entry.strip()
    if "(" in full_entry and ")" in full_entry:
        actor_name = full_entry.split("(")[0].strip()
        return actor_name if actor_name else None
    return None


def _extract_on_screen_names(full_entry: str) -> list[str]:
    """Extract character names from parentheses."""
    full_entry = full_entry.strip()
    if "(" in full_entry and ")" in full_entry:
        char_part = full_entry[full_entry.find("(")+1:full_entry.rfind(")")]
        return [c.strip() for c in char_part.split(",") if c.strip()]
    return []


def identify_characters_in_frame(
    frame_path: Path,
    timestamp_sec: float,
    model: Optional[GenerativeModel] = None,
    video_context: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """Use Vertex AI to identify all characters in a frame."""
    if model is None:
        model = _init_vertex_ai()
    
    try:
        # Upscale frame for better character recognition
        image_data = _frame_to_base64(_upscale_frame(frame_path))
        
        context_line = ""
        if video_context and video_context.strip():
            context_line = f"Video context / title: \"{video_context.strip()}\". Use this to help identify characters correctly. "
        
        prompt = (
            "Analyze this video frame carefully and identify ALL people/characters visible in it. "
            "For EACH person visible, identify: (1) Actor's real name if recognizable, (2) Character name they're playing. "
            f"{context_line}"
            "Format: Return a JSON array where each entry is 'Actor Name (Character Name, Role)' format. "
            "Example: [\"Chris Hemsworth (Thor)\", \"Scarlett Johansson (Black Widow, Natasha Romanoff)\", \"Unknown Person (Guard)\"] "
            "If no people are visible, return an empty array []. "
            "Use consistent actor and character names across all frames. "
            "Be very careful to identify characters correctly based on their appearance and the context provided."
        )
        
        response = model.generate_content([
            Part.from_data(data=image_data, mime_type="image/jpeg"),
            prompt
        ])
        
        text = (response.text or "").strip()
        if debug:
            print(f"\n[DEBUG Frame {timestamp_sec}s] Raw response: {text[:200]}...")
        
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        try:
            characters = json.loads(text)
            if not isinstance(characters, list):
                characters = []
        except json.JSONDecodeError as e:
            if debug:
                print(f"[DEBUG Frame {timestamp_sec}s] JSON parse error: {e}")
                print(f"[DEBUG Frame {timestamp_sec}s] Text was: {text[:300]}")
            characters = []
        
        return {
            "timestamp": timestamp_sec,
            "frame_path": str(frame_path),
            "characters": characters,
            "error": None,
        }
    
    except Exception as e:
        if debug:
            print(f"[DEBUG Frame {timestamp_sec}s] Exception: {type(e).__name__}: {e}")
        return {
            "timestamp": timestamp_sec,
            "frame_path": str(frame_path),
            "characters": [],
            "error": str(e),
        }


def process_frames_parallel(
    frame_list: list[tuple[float, Path]],
    max_workers: int = MAX_PARALLEL_WORKERS,
    video_context: Optional[str] = None,
    debug: bool = False,
) -> list[dict]:
    """Process frames in parallel with optional video context."""
    model = _init_vertex_ai()
    results = []
    import time

    # Maximum time (seconds) we'll allow a single frame-identification task to run
    # Increased from 45s to 120s to handle Vertex AI latency on free tier
    TASK_TIMEOUT = 120
    # How long to wait in as_completed before we check for stuck tasks
    AS_COMPLETED_POLL = 10

    # We'll track two time maps:
    # - submission_times_by_id[id] = time when the task was submitted
    # - start_times_by_id[id] = time when the worker actually began executing
    submission_times_by_id: dict[int, float] = {}
    start_times_by_id: dict[int, float] = {}

    def _worker_wrapper(task_id: int, timestamp_sec: float, frame_path: Path, model, video_context, debug_flag: bool):
        """Wrapper run inside worker thread to record actual start time then call the real function."""
        start_times_by_id[task_id] = time.time()
        # Call the existing function (keeps behaviour identical)
        return identify_characters_in_frame(frame_path, timestamp_sec, model=model, video_context=video_context, debug=debug_flag)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {}
        for i, (timestamp_sec, frame_path) in enumerate(frame_list):
            task_id = i
            # Decide which frames get debug on (preserve prior behaviour)
            debug_flag = debug or (i < 3)
            future = executor.submit(_worker_wrapper, task_id, timestamp_sec, frame_path, model, video_context, debug_flag)
            future_to_frame[future] = (timestamp_sec, frame_path, task_id)
            submission_times_by_id[task_id] = time.time()

        pending = set(future_to_frame.keys())
        completed = 0
        total = len(frame_list)

        while pending:
            try:
                for future in as_completed(pending, timeout=AS_COMPLETED_POLL):
                    pending.remove(future)
                    completed += 1
                    timestamp_sec, frame_path, task_id = future_to_frame[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status = "✓" if not result.get("error") else "✗"
                        print(f"  [{completed}/{total}] {status} Frame {timestamp_sec:.1f}s: {len(result.get('characters', []))} characters")
                    except Exception as e:
                        print(f"  [{completed}/{total}] ✗ Frame {timestamp_sec:.1f}s: {type(e).__name__}: {e}")
                        results.append({
                            "timestamp": timestamp_sec,
                            "frame_path": str(frame_path),
                            "characters": [],
                            "error": str(e),
                        })
            except TimeoutError:
                # Check for stuck futures and cancel them if they've exceeded TASK_TIMEOUT
                now = time.time()
                for future in list(pending):
                    timestamp_sec, frame_path, task_id = future_to_frame[future]
                    # Prefer the real start time if available, otherwise fall back to submission time
                    st = start_times_by_id.get(task_id, submission_times_by_id.get(task_id, now))
                    if now - st > TASK_TIMEOUT:
                        cancelled = future.cancel()
                        pending.discard(future)
                        completed += 1
                        msg = f"Task timed out after {TASK_TIMEOUT}s and was cancelled" if cancelled else "Task exceeded timeout and could not be cancelled"
                        print(f"  [{completed}/{total}] ✗ Frame {timestamp_sec:.1f}s: {msg}")
                        results.append({
                            "timestamp": timestamp_sec,
                            "frame_path": str(frame_path),
                            "characters": [],
                            "error": msg,
                        })

    return results


def aggregate_characters(frame_results: list[dict]) -> dict:
    """Aggregate character appearances, deduplicating by actor name."""
    character_data = defaultdict(lambda: {
        "count": 0,
        "appearances": [],
        "frames": [],
        "on_screen_names": set(),
    })
    
    for result in frame_results:
        timestamp = result["timestamp"]
        frame_path = result["frame_path"]
        characters = result.get("characters", [])
        
        for char_entry in characters:
            if char_entry and isinstance(char_entry, str):
                char_entry = char_entry.strip()
                if char_entry:
                    actor_name = _extract_actor_name(char_entry) or char_entry
                    on_screen_names = _extract_on_screen_names(char_entry)
                    
                    character_data[actor_name]["count"] += 1
                    character_data[actor_name]["appearances"].append(timestamp)
                    character_data[actor_name]["frames"].append(frame_path)
                    
                    if on_screen_names:
                        for on_screen_name in on_screen_names:
                            character_data[actor_name]["on_screen_names"].add(on_screen_name)
                    else:
                        character_data[actor_name]["on_screen_names"].add(char_entry)
    
    result_dict = {}
    for actor_name, data in character_data.items():
        result_dict[actor_name] = {
            "count": data["count"],
            "appearances": data["appearances"],
            "frames": data["frames"],
            "on_screen_names": sorted(list(data["on_screen_names"])),
        }
    
    return result_dict


def print_character_summary(character_data: dict, top_n: int = 10):
    """Print summary with actor name and character names in separate columns."""
    if not character_data:
        print("No characters detected.")
        return
    
    sorted_chars = sorted(
        character_data.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )
    
    print(f"\n{'='*130}")
    print(f"CHARACTER SUMMARY (Top {min(top_n, len(sorted_chars))} out of {len(sorted_chars)})")
    print(f"{'='*130}")
    print(f"{'Rank':<6} {'Actor Name':<30} {'On-Screen Names/Characters':<70} {'Count':<12}")
    print(f"{'-'*130}")
    
    for rank, (actor_name, data) in enumerate(sorted_chars[:top_n], 1):
        count = data["count"]
        on_screen = ", ".join(data["on_screen_names"])
        if len(on_screen) > 65:
            on_screen = on_screen[:62] + "..."
        print(f"{rank:<6} {actor_name:<30} {on_screen:<70} {count:<12}")
    
    print(f"{'='*130}\n")


def save_character_database(character_data: dict, output_path: Path):
    """Save character database to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {}
    for actor_name, data in character_data.items():
        export_data[actor_name] = {
            "count": data["count"],
            "appearances": data["appearances"],
            "frames": data["frames"],
            "on_screen_names": data["on_screen_names"],
        }
    
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Character database saved: {output_path}")


def main():
    """Main: Extract frames and identify all characters using Vertex AI in parallel."""
    from argparse import ArgumentParser
    from config import FRAMES_DIR, OUTPUT_DIR
    
    parser = ArgumentParser(description="Identify characters in video frames using Vertex AI (parallel)")
    parser.add_argument("video_path", help="Path to video file or YouTube URL")
    parser.add_argument("--interval", type=float, default=FRAME_SAMPLE_INTERVAL_SEC, help="Frame sampling interval (sec)")
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_WORKERS, help="Max parallel workers")
    parser.add_argument("--no-save", action="store_true", help="Don't save character database to JSON")
    
    args = parser.parse_args()
    
    video_input = args.video_path.strip()
    
    if video_input.startswith("http://") or video_input.startswith("https://") or "youtube.com" in video_input or "youtu.be" in video_input:
        print(f"Downloading video from YouTube...")
        from download import download_video
        try:
            video_path, duration_sec, video_title = download_video(video_input)
            if video_title:
                print(f"Title: {video_title[:70]}")
            print(f"Duration: {duration_sec:.0f}s")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return
    else:
        video_path = Path(video_input)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
    
    frames_dir = FRAMES_DIR / video_path.stem
    output_json = OUTPUT_DIR / f"{video_path.stem}_characters.json"
    
    print(f"Processing video: {video_path}")
    print(f"Frame interval: {args.interval}s")
    print(f"Max parallel workers: {args.max_workers}")
    
    print("\n[1/3] Extracting frames...")
    frame_list = extract_frames(video_path, frames_dir, interval_sec=args.interval)
    print(f"Extracted {len(frame_list)} frames")
    
    print(f"\n[2/3] Identifying characters (parallel processing with {args.max_workers} workers)...")
    frame_results = process_frames_parallel(frame_list, max_workers=args.max_workers)
    
    print("\n[3/3] Aggregating character data (deduplicating by actor)...")
    character_data = aggregate_characters(frame_results)
    
    print_character_summary(character_data, top_n=20)
    
    if not args.no_save:
        save_character_database(character_data, output_json)
    
    return character_data, frame_results


if __name__ == "__main__":
    main()
