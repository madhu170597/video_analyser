
# Project: Video Analysis

This repository contains an end-to-end video analysis pipeline. The README below documents only the files you requested, in the exact sequence you provided. Each entry contains a short description and any usage notes relevant to that file.

1) analyze_video.py
- Purpose: Orchestrates the full pipeline: download → frame extraction → character detection → screentime report → dialogues → clip recommendations and extraction.
- Usage: primary CLI to run the full pipeline. Example:

```bash
python analyze_video.py "<youtube link>" --interval 1.0 --max-workers 10 --top-n 5
```

2) config.py
- Purpose: Centralized configuration for paths, constants, and model/API settings (frames directory, output directory, Vertex/Gemini model names, etc.).
- Notes: Edit this file to change `FRAMES_DIR`, `OUTPUT_DIR`, video/sample intervals, or default model names and project IDs.

3) download.py
- Purpose: Download YouTube videos (via `yt-dlp`) or return local file paths. Saves videos into the `data/videos/` folder.
- Notes: Ensure `yt-dlp` is installed (listed in `requirements.txt`).

4) character_identification.py
- Purpose: Extract frames, optionally upscale for better recognition, and call the LLM (Vertex AI / Gemini fallback) to identify actors/characters in each frame. Aggregates results.
- Usage notes: `process_frames_parallel()` accepts `max_workers`, `video_context`, and `debug`. If you hit Vertex AI rate limits, lower `max_workers` or increase task timeout in this module.

5) character_screentime_report.py
- Purpose: Build the Excel report (overall screen time sheet, per-character sheets, and Top Dialogues sheet). Writes an `.xlsx` file into `output/`.
- Notes: The Top Dialogues sheet preserves full context and wraps text for readability.

6) character_clip_recommendations.py
- Purpose: Analyze aggregated appearances and transcript to recommend clips per character, then extract clips via `ffmpeg` into `output/clips/`.
- Notes: `ffmpeg` is required and must be in PATH for clip extraction.

7) punchy_dialogues.py
- Purpose: Transcribe audio (via `transcription.py`) and use the LLM to select top punchy dialogues with timestamps, context, and reasoning.
- Notes: Model responses are parsed as JSON; enable `debug` if output parsing fails.

8) transcription.py
- Purpose: Whisper-based transcription wrapper that converts video → audio → text segments; returns a list of `{start, end, text}` segments used by dialogue and clip modules.
- Notes: Requires `ffmpeg` and the Whisper package installed.

9) requirements.txt
- Purpose: Lists Python dependencies needed to run the pipeline (yt-dlp, opencv, whisper, google-generativeai / google-cloud-aiplatform, deepface, etc.).
- Install: `pip install -r requirements.txt` inside your virtual environment.

Execution
- Run the full pipeline with the `analyze_video.py` command shown at the top. Adjust `--interval`, `--max-workers`, and `--top-n` as needed.

Only these files are documented here — the README intentionally omits other project files and explanatory sections per your request.

GCloud / Vertex AI setup (quick)
--------------------------------
If you plan to use Vertex AI (recommended for best results), ensure your Google Cloud project and credentials are configured. Replace `PROJECT_ID`, `SERVICE_ACCOUNT_EMAIL`, and `BILLING_ACCOUNT_ID` with your values.

1. Login (user and application default credentials):

```bash
gcloud auth login
gcloud auth application-default login
```

2. Set active project:

```bash
gcloud config set project PROJECT_ID
```

3. Enable Vertex AI API:

```bash
gcloud services enable aiplatform.googleapis.com
```

4. Grant `Vertex AI User` role to your service account (required permission: `aiplatform.endpoints.predict`):

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:SERVICE_ACCOUNT_EMAIL \
  --role=roles/aiplatform.user
```

5. (If billing not already enabled) Link a billing account:

```bash
gcloud beta billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

After these steps, your environment should be able to call Vertex AI from the code. If you still see `403` errors, double-check the `SERVICE_ACCOUNT_EMAIL` and IAM bindings in the Cloud Console.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **ffmpeg** (required): used by Whisper (video→audio) and for clip extraction. Install before running:
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo pacman -S ffmpeg` (Arch)
  - **Windows**: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- **Gemini API key**: get a free key at [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
export GEMINI_API_KEY="your-key"
# or create a .env file with GEMINI_API_KEY=your-key
```

## Usage

```bash
python analyze_video.py "<youtube link>" --interval 1.0 --max-workers 10 --top-n 5
```

Output:

- `data/videos/` — downloaded video
- `data/frames/` — sampled frames (optional, for debugging)
- `output/` — **_screentime_report.xlsx** and extracted clips in `output/clips/`

## Tips

- **Videos with people**: Use a video with clear human faces for character detection. With `ENFORCE_FACE_DETECTION=True` (default), only real faces are counted (objects like puzzle pieces won’t be named as “characters”).
- **No characters detected**: If you see “Open Box Piece”-style names or no faces, the video may have no visible people. Try a talk head, interview, or movie clip. Optionally set `ENFORCE_FACE_DETECTION=False` in `config.py` for difficult/small faces.
- **Clips**: Long segments are split into sub-clips (max 45 s by default, see `MAX_CLIP_DURATION_SEC`) so the top 3 recommended clips are distinct parts of the video, not the full video three times.

## Character identification (face detection + Gemini)

The pipeline keeps **local face detection and clustering** to compute who appears when (screen time, occurrences). It then sends **one representative frame per character** to Gemini with the **video title** (or `--context`) so Gemini can name them as the audience would know them—e.g. **Iron Man**, **Thor**, **Captain America** for an Avengers trailer—instead of generic labels. All downstream outputs (report, clip titles, sheet names) use these names.

## How Gemini is used (to save credits)

1. **Character names**: one call with up to 5 representative face images.
2. **Clip recommendations**: one batched call per character (images for candidate clips) to get top 3 + title + reasoning.
3. **Punchy dialogues**: one text-only call with transcript and genre.

Face detection, clustering, screen-time computation, transcription (Whisper), and Excel generation are all local/open source.

## License

Open source. Dependencies: see their respective licenses (e.g. Whisper MIT, yt-dlp Unlicense, deepface MIT).
