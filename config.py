"""
Configuration for video analysis pipeline.
Tuned for 1-2 min videos to conserve Gemini credits; scale up intervals for longer videos.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
CLIPS_DIR = OUTPUT_DIR / "clips"

# Create dirs
for d in (DATA_DIR, OUTPUT_DIR, VIDEOS_DIR, FRAMES_DIR, CLIPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Video analysis (for 10-20 min videos; reduce intervals for shorter videos)
FRAME_SAMPLE_INTERVAL_SEC = 1.0   # Extract one frame every N seconds (1.0 = every second for better character coverage)
MIN_CLIP_DURATION_SEC = 5.0      # Minimum clip length for recommendations
MAX_CLIP_DURATION_SEC = 20.0     # Split long segments into sub-clips so we get variety (not full video 3x)
TOP_N_CHARACTERS = 5
TOP_N_CLIPS_PER_CHARACTER = 3
TOP_N_DIALOGUES = 5

# Face clustering
FACE_CLUSTER_MIN_SAMPLES = 2     # Min faces to form a character cluster
MAX_FACES_PER_FRAME = 10         # Cap for performance
ENFORCE_FACE_DETECTION = False   # False = detect both real faces and objects; set True to enforce only real faces

# Vertex AI Configuration (recommended for production, better quotas)
VERTEX_AI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
VERTEX_AI_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
VERTEX_AI_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-flash")  # Pre-trained face detection model
VERTEX_AI_MAX_PARALLEL_WORKERS = 10

# Gemini API (fallback, limited quota)
GEMINI_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")  # Stable, supports images; or "gemini-1.5-pro", "gemini-2.5-flash"
GEMINI_MAX_IMAGES_PER_CALL = 16        # Batch images to reduce request count
GEMINI_ENV_KEY = "GEMINI_API_KEY"

# def get_gemini_api_key() -> str:
#     key = os.environ.get(GEMINI_ENV_KEY) or os.environ.get("GOOGLE_API_KEY")
#     if not key:
#         raise ValueError(
#             f"Set {GEMINI_ENV_KEY} or GOOGLE_API_KEY in environment or .env file. "
#             "Get a key at https://aistudio.google.com/apikey"
#         )
#     return key
