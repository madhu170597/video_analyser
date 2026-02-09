#!/usr/bin/env python3
"""
Identify top 5 most punchy dialogues from video transcript.
Uses LLM to analyze dialogue based on video genre and context.
"""

import json
import re
from pathlib import Path
from typing import Optional
import vertexai
from vertexai.generative_models import GenerativeModel
from transcription import transcribe_video
from config import VERTEX_AI_PROJECT, VERTEX_AI_LOCATION, VERTEX_AI_MODEL


def detect_video_genre(video_title: str) -> str:
    """
    Detect video genre from title using simple heuristics.
    
    Args:
        video_title: Title of the video
    
    Returns:
        Inferred genre string
    """
    title_lower = video_title.lower()
    
    genres = {
        "action": ["avengers", "fight", "battle", "war", "combat", "hero", "superhero"],
        "comedy": ["funny", "laugh", "comedy", "comedy special", "standup"],
        "drama": ["drama", "emotional", "sad", "grief", "heartbreak"],
        "thriller": ["thriller", "suspense", "murder", "crime", "mystery"],
        "sci-fi": ["sci-fi", "scifi", "space", "future", "alien", "robot"],
        "fantasy": ["fantasy", "magic", "wizard", "dragon", "spell"],
        "horror": ["horror", "scary", "haunted", "ghost", "monster"],
    }
    
    for genre, keywords in genres.items():
        if any(keyword in title_lower for keyword in keywords):
            return genre
    
    return "general entertainment"


def prepare_dialogue_analysis_prompt(
    transcript_segments: list,
    genre: str,
    video_title: str,
    character_data: Optional[dict] = None
) -> str:
    """
    Build prompt for LLM to identify punchy dialogues.
    
    Args:
        transcript_segments: List of {start, end, text} from transcription
        genre: Video genre (action, drama, comedy, etc.)
        video_title: Title of the video
        character_data: Optional character info from task 1
    
    Returns:
        Formatted prompt for Vertex AI
    """
    
    # Format transcript for analysis
    transcript_text = ""
    for seg in transcript_segments:
        transcript_text += f"[{seg['start']:.0f}s] {seg['text']}\n"
    
    # Build character info if available
    char_info = ""
    if character_data:
        char_list = sorted(
            character_data.items(),
            key=lambda x: x[1].get("count", 0),
            reverse=True
        )[:5]
        char_info = "Key characters: " + ", ".join([actor for actor, _ in char_list]) + "\n"
    
    prompt = f"""You are a film editor and content strategist. Analyze this video's transcript and identify the top 5 most punchy, memorable, or impactful dialogues.

VIDEO INFORMATION:
- Title: {video_title}
- Genre: {genre}
{char_info}

TRANSCRIPT:
{transcript_text}

TASK: Identify the 5 most punchy dialogues based on:
1. Memorability (lines people will remember and quote)
2. Emotional impact (dialogue that resonates emotionally)
3. Humor (funny or witty lines - if applicable for genre)
4. Significance (pivotal moments or key plot points)
5. Entertainment value (engaging and engaging for clips/trailers)

For EACH dialogue, provide:
- The exact dialogue line (as spoken in transcript)
- Start timestamp (in seconds)
- Character/speaker (if identifiable)
- Why it's punchy (1-2 sentence reasoning)

Format your response as VALID JSON (no markdown, just raw JSON):
{{
  "dialogues": [
    {{
      "rank": 1,
      "dialogue": "Exact dialogue text",
      "start_second": 15,
      "speaker": "Character/Actor name or 'Unknown'",
      "context": "Brief context of the scene",
      "reasoning": "Why this dialogue is punchy and memorable"
    }},
    ...
  ],
  "genre_detected": "{genre}",
  "summary": "Brief summary of why these dialogues were chosen"
}}

IMPORTANT:
- Provide exactly 5 dialogues
- Times must be realistic and match transcript
- No markdown formatting - return ONLY valid JSON
- Dialogue text must be EXACT excerpts from the transcript
- Reasoning should be specific to the genre and video context
"""
    
    return prompt


def recommend_dialogues_with_llm(
    prompt: str,
    model_name: str = VERTEX_AI_MODEL
) -> dict:
    """
    Call Vertex AI to identify punchy dialogues.
    
    Args:
        prompt: Prepared prompt with transcript and genre info
        model_name: Vertex AI model to use
    
    Returns:
        Parsed JSON with dialogue recommendations
    """
    vertexai.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
    model = GenerativeModel(model_name)
    
    print("\n  [LLM] Analyzing transcript for punchy dialogues...")
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Extract JSON from response (in case LLM adds markdown)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        dialogue_data = json.loads(response_text)
        print(f"  ✓ LLM identified {len(dialogue_data.get('dialogues', []))} punchy dialogues")
        return dialogue_data
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse LLM response as JSON: {e}")
        print(f"  Response preview: {response_text[:200]}")
        raise


def validate_dialogue(dialogue: dict, max_duration_sec: float) -> bool:
    """Validate dialogue has valid timestamp."""
    start = dialogue.get("start_second", 0)
    
    if start < 0 or start > max_duration_sec:
        return False
    
    return True


def analyze_punchy_dialogues(
    video_path: Path,
    character_data: Optional[dict] = None,
    video_title: Optional[str] = None,
    genre: Optional[str] = None
) -> dict:
    """
    End-to-end punchy dialogue analysis pipeline.
    
    Args:
        video_path: Path to video file
        character_data: Optional character identification data from task 1
        video_title: Video title (for genre detection)
        genre: Optional genre override
    
    Returns:
        Dictionary with dialogue recommendations and metadata
    """
    
    print("\n" + "=" * 80)
    print("TASK 3: TOP 5 PUNCHY DIALOGUES ANALYSIS")
    print("=" * 80)
    
    # Step 1: Get video duration
    print(f"\n[Step 1/3] Analyzing video...")
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()
        print(f"  ✓ Video duration: {video_duration_sec:.1f} seconds")
    except Exception as e:
        print(f"  ✗ Error reading video: {e}")
        return {}
    
    # Step 2: Transcribe video
    print(f"\n[Step 2/3] Transcribing video...")
    try:
        transcript_segments = transcribe_video(video_path, model_size="base")
        print(f"  ✓ Transcription complete: {len(transcript_segments)} segments")
    except Exception as e:
        print(f"  ✗ Transcription failed: {e}")
        return {}
    
    # Step 3: Detect genre and get LLM recommendations
    print(f"\n[Step 3/3] Getting LLM recommendations...")
    
    # Detect genre if not provided
    if genre is None:
        if video_title:
            genre = detect_video_genre(video_title)
        else:
            genre = "general entertainment"
    
    print(f"  Genre: {genre}")
    
    try:
        prompt = prepare_dialogue_analysis_prompt(
            transcript_segments,
            genre,
            video_title or video_path.stem,
            character_data
        )
        
        dialogue_data = recommend_dialogues_with_llm(prompt)
        
        # Validate dialogues
        dialogues = dialogue_data.get("dialogues", [])
        valid_dialogues = [
            d for d in dialogues
            if validate_dialogue(d, video_duration_sec)
        ]
        
        invalid_count = len(dialogues) - len(valid_dialogues)
        if invalid_count > 0:
            print(f"  ⚠ Skipping {invalid_count} dialogues with invalid timestamps")
        
        # Prepare output
        output_data = {
            "video_file": str(video_path.name),
            "video_duration_sec": video_duration_sec,
            "genre": genre,
            "transcript_segments": transcript_segments,
            "dialogue_analysis": dialogue_data,
            "validated_dialogues": valid_dialogues,
            "summary": dialogue_data.get("summary", "")
        }
        
        print("\n" + "=" * 80)
        print(f"✓ PUNCHY DIALOGUES ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Identified: {len(valid_dialogues)} punchy dialogues")
        
        if valid_dialogues:
            print("\nTop 5 Dialogues:")
            for i, d in enumerate(valid_dialogues[:5], 1):
                print(f"  {i}. [{d.get('start_second', 0):.0f}s] {d.get('dialogue', '')[:60]}...")
        
        return output_data
        
    except Exception as e:
        print(f"  ✗ Error in dialogue analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


def save_dialogues_report(
    dialogue_data: dict,
    output_path: Path
) -> None:
    """
    Save dialogue analysis report as JSON.
    
    Args:
        dialogue_data: Dialogue analysis output
        output_path: Path to save JSON report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(dialogue_data, f, indent=2)
    
    print(f"  ✓ Dialogue report saved: {output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Identify top 5 punchy dialogues from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--character-json", help="Path to character data JSON from task 1")
    parser.add_argument("--title", help="Video title (for genre detection)")
    parser.add_argument("--genre", help="Override genre (action, drama, comedy, etc.)")
    parser.add_argument("--output-dir", default="output", help="Output directory for report")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        exit(1)
    
    character_data = None
    if args.character_json:
        char_path = Path(args.character_json)
        if char_path.exists():
            with open(char_path) as f:
                character_data = json.load(f)
    
    dialogue_data = analyze_punchy_dialogues(
        video_path,
        character_data=character_data,
        video_title=args.title,
        genre=args.genre
    )
    
    if dialogue_data:
        safe_title = (args.title or video_path.stem).replace('/', '_').replace('\\', '_')[:100]
        report_path = Path(args.output_dir) / f"{safe_title}_dialogues.json"
        save_dialogues_report(dialogue_data, report_path)
