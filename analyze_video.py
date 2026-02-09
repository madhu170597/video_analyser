#!/usr/bin/env python3
"""
End-to-End Video Analysis Pipeline:
1. Download YouTube video
2. Extract frames and identify characters using Vertex AI (parallel)
3. Deduplicate by actor name and aggregate on-screen names
4. Generate Excel report with screen time analysis
5. Recommend and extract best clips for top characters
6. Identify top 5 punchy dialogues and add to Excel
"""
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

# Import modules
from config import FRAMES_DIR, OUTPUT_DIR, VIDEOS_DIR, FRAME_SAMPLE_INTERVAL_SEC
from download import download_video
from character_identification import (
    extract_frames,
    process_frames_parallel,
    aggregate_characters,
    print_character_summary,
    save_character_database,
    MAX_PARALLEL_WORKERS,
)
from character_screentime_report import (
    create_excel_report,
    add_dialogues_sheet,
    save_excel_workbook,
)
from character_clip_recommendations import analyze_and_recommend_clips
from punchy_dialogues import analyze_punchy_dialogues


def main():
    parser = ArgumentParser(
        description="End-to-end video analysis: download → character detection → screen time report"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--interval",
        type=float,
        default=FRAME_SAMPLE_INTERVAL_SEC,
        help="Frame sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_PARALLEL_WORKERS,
        help="Max parallel workers for frame processing (default: 10)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top characters to analyze (default: 5)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save character database JSON",
    )
    parser.add_argument(
        "--skip-clips",
        action="store_true",
        help="Skip clip recommendation and extraction (Task 2)",
    )
    parser.add_argument(
        "--skip-dialogues",
        action="store_true",
        help="Skip punchy dialogue analysis (Task 3)",
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("VIDEO ANALYSIS PIPELINE: Character Detection → Screen Time Report → Clip Recommendations")
    print("=" * 100)
    
    # Step 1: Download video
    print("\n[Step 1/4] Downloading video from YouTube...")
    try:
        video_path, video_duration_sec, video_title = download_video(args.url, VIDEOS_DIR)
        print(f"✓ Downloaded: {video_path}")
        if video_title:
            print(f"  Title: {video_title[:80]}")
        print(f"  Duration: {video_duration_sec:.0f} seconds ({video_duration_sec/60:.1f} minutes)")
    except Exception as e:
        print(f"✗ Error downloading video: {e}")
        return 1
    
    # Step 2: Extract frames and identify characters
    print(f"\n[Step 2/4] Extracting frames and identifying characters...")
    print(f"  Frame interval: {args.interval}s")
    print(f"  Max parallel workers: {args.max_workers}")
    
    # Use video title as directory name (fallback to video_path.stem if title not available)
    dir_name = video_title.replace('/', '_').replace('\\', '_')[:100] if video_title else video_path.stem
    frames_dir = FRAMES_DIR / dir_name
    
    try:
        print("\n  [2a] Extracting frames...")
        frame_list = extract_frames(video_path, frames_dir, interval_sec=args.interval)
        print(f"  ✓ Extracted {len(frame_list)} frames")
        
        print("\n  [2b] Identifying characters (Vertex AI parallel processing)...")
        frame_results = process_frames_parallel(frame_list, max_workers=args.max_workers, video_context=video_title)
        
        print(f"\n  [2c] Aggregating and deduplicating...")
        character_data = aggregate_characters(frame_results)
        print(f"  ✓ Identified {len(character_data)} unique actors")
        
    except Exception as e:
        print(f"✗ Error during character identification: {e}")
        return 1
    
    # Step 3: Print summary and save JSON
    print(f"\n[Step 3/6] Character Summary:")
    print_character_summary(character_data, top_n=20)
    
    if not args.no_json:
        safe_title = video_title.replace('/', '_').replace('\\', '_')[:100] if video_title else video_path.stem
        output_json = OUTPUT_DIR / f"{safe_title}_characters.json"
        save_character_database(character_data, output_json)
    
    # Step 4: Generate Excel report (will be updated later with dialogues)
    print(f"\n[Step 4/6] Generating Excel report...")
    safe_title = video_title.replace('/', '_').replace('\\', '_')[:100] if video_title else video_path.stem
    output_xlsx = OUTPUT_DIR / f"{safe_title}_screentime_report.xlsx"
    
    try:
        wb = create_excel_report(
            character_data,
            output_xlsx,
            video_duration_sec=video_duration_sec,
            frame_interval_sec=args.interval,
            top_n=args.top_n,
        )
        print(f"  ✓ Base report created")
    except Exception as e:
        print(f"✗ Error generating Excel report: {e}")
        return 1
    
    # Step 5: Analyze punchy dialogues and add to Excel
    print(f"\n[Step 5/6] Analyzing dialogues...")
    if not args.skip_dialogues:
        try:
            dialogues_data = analyze_punchy_dialogues(
                video_path,
                character_data=character_data,
                video_title=video_title,
            )
            
            if dialogues_data:
                wb = add_dialogues_sheet(wb, dialogues_data)
                print(f"  ✓ Dialogue sheet added to Excel")
        except Exception as e:
            print(f"  ⚠ Warning: Dialogue analysis failed: {e}")
    
    # Save updated Excel workbook
    try:
        save_excel_workbook(wb, output_xlsx)
        print(f"  ✓ Excel report saved: {output_xlsx}")
    except Exception as e:
        print(f"✗ Error saving Excel report: {e}")
        return 1
    
    # Step 6: Recommend and extract clips (Task 2)
    print(f"\n[Step 6/6] Recommending character clips...")
    if not args.skip_clips:
        try:
            output_clips_dir = OUTPUT_DIR / "clips"
            analyze_and_recommend_clips(
                video_path,
                character_data,
                frame_interval_sec=args.interval,
                output_dir=output_clips_dir,
                top_n=args.top_n,
                max_workers=args.max_workers,
            )
        except Exception as e:
            print(f"  ⚠ Warning: Clip recommendation failed: {e}")
    
    # Summary
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE ✓")
    print("=" * 100)
    print(f"Output files:")
    print(f"  • Frames: {frames_dir}")
    if not args.no_json:
        print(f"  • Character data: {output_json}")
    print(f"  • Excel report: {output_xlsx}")
    if not args.skip_clips:
        print(f"  • Clips: {OUTPUT_DIR / 'clips'}")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
