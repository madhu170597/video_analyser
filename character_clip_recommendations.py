#!/usr/bin/env python3
"""
Intelligent Clip Recommendation System:
1. Analyzes character appearances + transcript
2. Uses Vertex AI to recommend 3 best clips per character
3. Extracts clips using ffmpeg in parallel
4. Generates recommendations report
"""

import json
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import vertexai
from vertexai.generative_models import GenerativeModel
from transcription import transcribe_video
from config import VERTEX_AI_PROJECT, VERTEX_AI_LOCATION, VERTEX_AI_MODEL


def extract_character_appearances(character_data: dict, frame_interval_sec: float) -> dict:
    """
    Convert frame counts to video timestamps per character.
    
    Args:
        character_data: {actor: {count: int, appearances: [frame_indices], on_screen_names: [...]}}
        frame_interval_sec: Sampling interval used during frame extraction
    
    Returns:
        {actor: [(start_sec, end_sec, frame_count), ...]} sorted by time
    """
    appearances = {}
    
    for actor, data in character_data.items():
        frame_indices = data.get("appearances", [])
        
        if not frame_indices:
            continue
        
        # Group consecutive frames into segments
        segments = []
        current_group = [frame_indices[0]]
        
        for frame_idx in frame_indices[1:]:
            # If within 2 frames of previous (allowing brief gaps)
            if frame_idx - current_group[-1] <= 2:
                current_group.append(frame_idx)
            else:
                # Save current group as a segment
                start_frame = current_group[0]
                end_frame = current_group[-1]
                segments.append((
                    start_frame * frame_interval_sec,
                    end_frame * frame_interval_sec,
                    len(current_group)
                ))
                current_group = [frame_idx]
        
        # Add last group
        if current_group:
            start_frame = current_group[0]
            end_frame = current_group[-1]
            segments.append((
                start_frame * frame_interval_sec,
                end_frame * frame_interval_sec,
                len(current_group)
            ))
        
        appearances[actor] = sorted(segments, key=lambda x: x[0])
    
    return appearances


def _rank_appearance_segments(segments: list, top_n: int = 5) -> list:
    """
    Rank appearance segments by quality indicators:
    - Length (longer = more content)
    - Position (earlier segments more likely to be introductions)
    - Isolation (standalone segments better than continuous presence)
    
    Returns top_n segments ranked by quality score.
    """
    if not segments:
        return []
    
    scored = []
    for idx, (start, end, frame_count) in enumerate(segments):
        duration = end - start
        
        # Score based on:
        # - Duration (preferring 5-30 sec range)
        # - Position in video (first 20% gets boost for introductions)
        # - Frame density (more frames = more character visibility)
        
        duration_score = min(duration, 30) / 30 * 40  # Max 40 points
        position_score = (1 - (idx / max(1, len(segments)))) * 20  # Max 20 points
        density_score = min(frame_count / 20, 1.0) * 40  # Max 40 points
        
        total_score = duration_score + position_score + density_score
        scored.append((total_score, start, end, frame_count))
    
    # Sort by score descending, then by start time
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(s, e, f) for _, s, e, f in scored[:top_n]]


def prepare_clip_analysis_prompt(
    character_data: dict,
    transcript_segments: list,
    character_appearances: dict,
    top_n: int = 5
) -> str:
    """
    Build comprehensive prompt for LLM with character appearances + transcript.
    
    Args:
        character_data: Character identification data from task 1
        transcript_segments: List of {start, end, text} from transcription.py
        character_appearances: {actor: [(start_sec, end_sec, count), ...]}
        top_n: Number of top characters to analyze
    
    Returns:
        Formatted prompt string for Vertex AI
    """
    
    # Get top N characters by appearance count
    top_chars = sorted(
        character_data.items(),
        key=lambda x: x[1].get("count", 0),
        reverse=True
    )[:top_n]
    
    # Build character appearances section with ranked segments
    char_section = "CHARACTER APPEARANCES AND KEY MOMENTS:\n"
    char_section += "=" * 80 + "\n"
    
    for actor, data in top_chars:
        on_screen_names = data.get("on_screen_names", [])
        char_names_str = ", ".join(on_screen_names[:3]) if on_screen_names else "Unknown"
        count = data.get("count", 0)
        total_duration = sum(e - s for s, e, _ in character_appearances.get(actor, []))
        
        char_section += f"\n📌 {actor} (plays: {char_names_str})\n"
        char_section += f"   Appearances: {count} frames | Total on-screen: {total_duration:.0f}s\n"
        
        # Get top appearance segments (ranked by quality)
        appearances = character_appearances.get(actor, [])
        ranked_appearances = _rank_appearance_segments(appearances, top_n=5)
        
        if ranked_appearances:
            char_section += "   Top Moments:\n"
            for idx, (start_sec, end_sec, frame_count) in enumerate(ranked_appearances, 1):
                duration = end_sec - start_sec
                
                # Find transcript in this time range
                trans_in_range = [
                    s for s in transcript_segments
                    if s["start"] <= end_sec and s["end"] >= start_sec
                ]
                
                if trans_in_range:
                    # Build dialogue context
                    dialogue = " ".join([s["text"] for s in trans_in_range])
                    if len(dialogue) > 120:
                        dialogue = dialogue[:120] + "..."
                    char_section += f"   {idx}. [{start_sec:.0f}s-{end_sec:.0f}s, {duration:.0f}s] \"{dialogue}\"\n"
                else:
                    char_section += f"   {idx}. [{start_sec:.0f}s-{end_sec:.0f}s, {duration:.0f}s] (action/visual moment)\n"
    
    # Build comprehensive transcript section with better sampling
    transcript_section = "\nFULL TRANSCRIPT HIGHLIGHTS (for context):\n"
    transcript_section += "=" * 80 + "\n"
    
    if transcript_segments:
        # Sample more intelligently: include start, end, and key moments throughout
        sample_count = min(20, max(10, len(transcript_segments) // 5))
        sample_indices = set(range(0, len(transcript_segments), max(1, len(transcript_segments) // sample_count)))
        sample_indices.add(0)  # Always include start
        sample_indices.add(len(transcript_segments) - 1)  # Always include end
        
        for idx in sorted(sample_indices):
            if idx < len(transcript_segments):
                seg = transcript_segments[idx]
                text = seg["text"][:100]
                if text.strip():
                    transcript_section += f"  [{seg['start']:.0f}s] {text}\n"
    else:
        transcript_section += "  (No transcript available - focus on visual moments and character actions)\n"
    
    # Main prompt with better guidance
    prompt = f"""You are a professional video editor and content curator with expertise in creating engaging highlight reels.

Your task: Recommend exactly 3 EXCELLENT clips for each of the top 5 characters that would be perfect for:
- YouTube shorts / TikTok compilations
- Character highlight reels
- Fan engagement content
- Recommended watch moments

{char_section}

{transcript_section}

CRITICAL GUIDELINES FOR CLIP SELECTION:

1. CHARACTER PROMINENCE: Each clip MUST clearly showcase the character - they should be visible/speaking for most of the clip
2. NARRATIVE VARIETY: For each character, pick 3 different types of moments:
   - Dialogue/Emotional moment (character speaking or reacting strongly)
   - Action/Physical moment (character doing something notable)
   - Climactic/Impactful moment (most memorable scene they're in)
3. ENGAGEMENT: Pick clips that are inherently interesting, entertaining, or memorable to viewers
4. TIMING: Prefer mid-video moments over very start/end (which may lack context)
5. DURATION: 10-20 seconds is ideal for engagement (not too short, not too long)

FORMAT YOUR RESPONSE AS VALID JSON (no markdown, just raw JSON):
{{
  "clips": [
    {{
      "character": "Character Name",
      "actor": "Actor Name",
      "clip_number": 1,
      "start_second": 45,
      "duration_seconds": 15,
      "type": "dialogue",
      "title": "5-8 Word Engaging Title Here",
      "description": "Why this is a standout clip (1-2 sentences). What makes it memorable or entertaining?"
    }},
    ...
  ]
}}

IMPORTANT CONSTRAINTS:
- Minimum duration: 5 seconds
- Maximum duration: 25 seconds (prefer 10-20 seconds)
- Start times must be based on the moments listed above
- Provide exactly 3 clips per character (15 total for top 5 characters)
- Return ONLY valid JSON (no markdown, no explanations)
- Each character's 3 clips should have different "type" values
- Clips must be diverse in content and not repetitive
- Prioritize quality over quantity - only recommend clips that are truly compelling
"""
    
    return prompt


def recommend_clips_with_llm(prompt: str, model_name: str = VERTEX_AI_MODEL) -> dict:
    """
    Call Vertex AI to get clip recommendations.
    
    Args:
        prompt: Prepared prompt with character and transcript info
        model_name: Vertex AI model to use
    
    Returns:
        Parsed JSON with clip recommendations
    """
    vertexai.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
    model = GenerativeModel(model_name)
    
    print("\n  [LLM] Analyzing characters and transcript for clip recommendations...")
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Extract JSON from response (in case LLM adds markdown)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        clip_data = json.loads(response_text)
        print(f"  ✓ LLM recommended {len(clip_data.get('clips', []))} clips")
        return clip_data
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse LLM response as JSON: {e}")
        print(f"  Response preview: {response_text[:200]}")
        raise


def validate_clip(clip: dict, video_duration_sec: float) -> bool:
    """Validate clip has valid timestamps and minimum duration."""
    start = clip.get("start_second", 0)
    duration = clip.get("duration_seconds", 0)
    
    if duration < 5 or duration > 30:
        return False
    
    if start < 0 or start + duration > video_duration_sec:
        return False
    
    return True


def extract_clip_ffmpeg(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path
) -> bool:
    """
    Extract a single clip using ffmpeg.
    
    Args:
        video_path: Path to source video
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        output_path: Path to save extracted clip
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-q:v", "5",
            "-y",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            print(f"    ✗ ffmpeg error for {output_path.name}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout extracting {output_path.name}")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def extract_clips_parallel(
    video_path: Path,
    clip_recommendations: dict,
    output_dir: Path,
    video_duration_sec: float,
    max_workers: int = 10
) -> list:
    """
    Extract all recommended clips in parallel using ffmpeg.
    
    Args:
        video_path: Path to source video
        clip_recommendations: JSON from LLM with clip info
        output_dir: Directory to save clips
        video_duration_sec: Total video duration for validation
        max_workers: Max parallel ffmpeg processes
    
    Returns:
        List of successfully extracted clips with metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = clip_recommendations.get("clips", [])
    
    # Validate clips
    valid_clips = [c for c in clips if validate_clip(c, video_duration_sec)]
    invalid_count = len(clips) - len(valid_clips)
    
    if invalid_count > 0:
        print(f"  ⚠ Skipping {invalid_count} clips with invalid timestamps/duration")
    
    if not valid_clips:
        print("  ✗ No valid clips to extract")
        return []
    
    print(f"\n  Extracting {len(valid_clips)} clips in parallel ({max_workers} workers)...")
    
    # Prepare extraction tasks
    tasks = []
    extracted_clips = []
    
    for clip in valid_clips:
        # Generate safe filename
        actor = clip.get("actor", "Unknown").replace(" ", "_")
        clip_num = clip.get("clip_number", 0)
        title = clip.get("title", "Clip").replace(" ", "_")[:30]
        filename = f"{actor}_Clip{clip_num}_{title}.mp4"
        
        output_path = output_dir / filename
        start_sec = clip.get("start_second", 0)
        duration_sec = clip.get("duration_seconds", 0)
        
        tasks.append((video_path, start_sec, duration_sec, output_path, clip))
    
    # Extract in parallel
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_clip_ffmpeg, task[0], task[1], task[2], task[3])
            for task in tasks
        ]
        
        for future, task in zip(futures, tasks):
            if future.result():
                success_count += 1
                clip_info = task[4].copy()
                clip_info["output_file"] = str(task[3].name)
                clip_info["output_size_mb"] = task[3].stat().st_size / (1024*1024)
                extracted_clips.append(clip_info)
                print(f"    ✓ {task[3].name}")
            else:
                print(f"    ✗ Failed: {task[3].name}")
    
    print(f"\n  ✓ Successfully extracted {success_count}/{len(valid_clips)} clips")
    return extracted_clips


def save_clip_recommendations_report(
    clip_data: dict,
    output_path: Path
) -> None:
    """
    Save clip recommendations report as JSON.
    
    Args:
        clip_data: Extracted clip information with metadata
        output_path: Path to save JSON report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(clip_data, f, indent=2)
    
    print(f"\n  ✓ Clip recommendations saved: {output_path}")


def analyze_and_recommend_clips(
    video_path: Path,
    character_data: dict,
    frame_interval_sec: float = 1.0,
    output_dir: Path = None,
    top_n: int = 5,
    max_workers: int = 10
) -> dict:
    """
    End-to-end clip recommendation pipeline.
    
    Args:
        video_path: Path to video file
        character_data: Character identification data from task 1
        frame_interval_sec: Frame sampling interval from task 1
        output_dir: Directory to save clips
        top_n: Number of top characters to analyze
        max_workers: Max parallel workers for clip extraction
    
    Returns:
        Dictionary with clip recommendations and extraction results
    """
    
    if output_dir is None:
        output_dir = Path("output/clips")
    
    print("\n" + "=" * 80)
    print("TASK 2: CHARACTER CLIP RECOMMENDATIONS & EXTRACTION")
    print("=" * 80)
    
    # Step 1: Get video duration
    print(f"\n[Step 1/4] Analyzing video...")
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()
        print(f"  ✓ Video duration: {video_duration_sec:.1f} seconds ({video_duration_sec/60:.1f} minutes)")
    except Exception as e:
        print(f"  ✗ Error reading video: {e}")
        return {}
    
    # Step 2: Get character appearances
    print(f"\n[Step 2/4] Extracting character appearance timeline...")
    character_appearances = extract_character_appearances(character_data, frame_interval_sec)
    print(f"  ✓ Analyzed {len(character_appearances)} characters with {sum(len(v) for v in character_appearances.values())} appearance segments")
    
    # Step 3: Transcribe video
    print(f"\n[Step 3/4] Transcribing video...")
    try:
        transcript_segments = transcribe_video(video_path, model_size="base")
        print(f"  ✓ Transcription complete: {len(transcript_segments)} segments")
    except Exception as e:
        print(f"  ⚠ Transcription failed: {e}")
        print(f"    Continuing without transcript...")
        transcript_segments = []
    
    # Step 4: Build prompt and get LLM recommendations
    print(f"\n[Step 4/4] Getting LLM recommendations...")
    try:
        prompt = prepare_clip_analysis_prompt(
            character_data,
            transcript_segments,
            character_appearances,
            top_n=top_n
        )
        
        clip_recommendations = recommend_clips_with_llm(prompt)
        
        # Step 5: Extract clips
        print(f"\n[Step 5/5] Extracting recommended clips...")
        extracted_clips = extract_clips_parallel(
            video_path,
            clip_recommendations,
            output_dir,
            video_duration_sec,
            max_workers=max_workers
        )
        
        # Step 6: Save report
        report_data = {
            "video_file": str(video_path.name),
            "video_duration_sec": video_duration_sec,
            "character_data": character_data,
            "character_appearances": {
                actor: [{"start": s, "end": e, "frames": c} for s, e, c in segs]
                for actor, segs in character_appearances.items()
            },
            "transcript_segments": transcript_segments[:50],  # Save first 50 for reference
            "llm_recommendations": clip_recommendations,
            "extracted_clips": extracted_clips
        }
        
        report_path = output_dir.parent / f"{video_path.stem}_clip_recommendations.json"
        save_clip_recommendations_report(report_data, report_path)
        
        print("\n" + "=" * 80)
        print(f"✓ CLIP RECOMMENDATION COMPLETE")
        print("=" * 80)
        print(f"Extracted: {len(extracted_clips)} clips")
        print(f"Location: {output_dir}")
        print(f"Report: {report_path}")
        
        return report_data
        
    except Exception as e:
        print(f"  ✗ Error in clip recommendation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Recommend and extract character clips from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("character_json", help="Path to character data JSON from task 1")
    parser.add_argument("--frame-interval", type=float, default=1.0, help="Frame interval used in task 1")
    parser.add_argument("--output-dir", default="output/clips", help="Output directory for clips")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top characters")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    character_json_path = Path(args.character_json)
    
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        exit(1)
    
    if not character_json_path.exists():
        print(f"✗ Character JSON not found: {character_json_path}")
        exit(1)
    
    with open(character_json_path) as f:
        character_data = json.load(f)
    
    analyze_and_recommend_clips(
        video_path,
        character_data,
        frame_interval_sec=args.frame_interval,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        max_workers=args.max_workers
    )
