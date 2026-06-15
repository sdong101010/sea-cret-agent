from pathlib import Path
from dotenv import load_dotenv
import os
import truststore

# Use the macOS system trust store so we can reach the Salesforce
# AI Model Gateway, which is signed by an internal Salesforce CA.
truststore.inject_into_ssl()

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# (Whisper config retired -- transcription now uses bin/speech_sidecar)

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 2
AUDIO_DEVICE_NAME = os.getenv("AUDIO_DEVICE", "BlackHole")
MIC_DEVICE_NAME = os.getenv("MIC_DEVICE", "")
# Linear gain applied to the mic before it's mixed into the transcription
# stream. Built-in laptop mics often capture at very low levels (RMS < 0.01),
# which produces a mostly-silent mixed signal that the speech engine can't
# transcribe well. Default 8x is a reasonable starting point; bump up if your
# voice is still barely audible to the sidecar.
MIC_GAIN = float(os.getenv("MIC_GAIN", "8.0"))

QUESTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("QUESTION_CONFIDENCE_THRESHOLD", "0.6")
)
TRANSCRIPT_WINDOW_SECONDS = 60
THOUGHT_PAUSE_SECONDS = float(os.getenv("THOUGHT_PAUSE_SECONDS", "0.75"))

SUMMARIES_DIR = DATA_DIR / "summaries"
SUMMARIES_DIR.mkdir(exist_ok=True)

ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BEDROCK_BASE_URL", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
