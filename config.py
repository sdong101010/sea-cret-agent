from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 2
AUDIO_DEVICE_NAME = os.getenv("AUDIO_DEVICE", "BlackHole")
MIC_DEVICE_NAME = os.getenv("MIC_DEVICE", "")

QUESTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("QUESTION_CONFIDENCE_THRESHOLD", "0.6")
)
TRANSCRIPT_WINDOW_SECONDS = 60
THOUGHT_PAUSE_SECONDS = float(os.getenv("THOUGHT_PAUSE_SECONDS", "0.75"))

SUMMARIES_DIR = DATA_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
