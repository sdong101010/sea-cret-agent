# Meeting Copilot

A local meeting assistant that captures system audio from MS Teams / Google Meet calls, transcribes in real-time, detects customer questions via Gemini, and researches answers from the web. Speaker diarization separates "You" from other speakers so only customer questions are surfaced.

## How It Works

1. **BlackHole** captures system audio (what you hear in the meeting)
2. **faster-whisper** transcribes speech to text locally
3. **Gemini** detects customer questions from the transcript
4. **Web search** finds relevant Salesforce docs and pages
5. **Gemini** synthesizes a concise answer from the web research
6. A browser-based sidebar shows the live transcript and Q&A cards on your second monitor

## Prerequisites

### 1. BlackHole (virtual audio device)

```bash
brew install blackhole-2ch
```

Then open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup"):
1. Click **+** → **Create Multi-Output Device**
2. Check both your real speakers/headphones AND **BlackHole 2ch**
3. Set this Multi-Output Device as your system output (System Settings → Sound → Output)

This lets you hear the meeting audio AND capture it simultaneously.

### 2. PortAudio (required by sounddevice)

```bash
brew install portaudio
```

### 3. Gemini API Key

Get a free key from https://aistudio.google.com/apikey — this powers question detection, answer generation, and meeting summaries.

## Setup

```bash
cd meeting-copilot

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit config (add your Gemini API key)
cp .env.example .env
```

To find your BlackHole device name:
```bash
python -m sounddevice
```

## Usage

```bash
source .venv/bin/activate
python -m app.main
```

Open **http://localhost:8765** in your browser (second monitor).

1. Click **Start Session** before your meeting begins
2. Click **Calibrate Voice** and speak for 3 seconds so the copilot can identify you vs. other speakers
3. The transcript appears on the left as people speak
4. Customer questions are detected and shown on the right with web-researched answers
5. Click **Stop Session** when the meeting ends — a summary is generated and saved to `data/summaries/`

## Configuration

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | From https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for Q&A and detection |
| `AUDIO_DEVICE` | `BlackHole` | Audio input device name |
| `MIC_DEVICE` | (auto) | Microphone device for voice calibration |
| `WHISPER_MODEL_SIZE` | `small` | Whisper model: tiny, base, small, medium, large-v3 |
| `QUESTION_CONFIDENCE_THRESHOLD` | `0.6` | Min confidence to surface a question (0.0-1.0) |
