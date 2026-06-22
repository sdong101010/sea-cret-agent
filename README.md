# Sea-cret Agent

A local meeting assistant that captures system audio from MS Teams / Google Meet calls, transcribes in real-time, detects questions via Claude, and researches answers using Slack MCP and the public web. Speaker labels separate "Me" (mic) from remote participants ("Speaker 1", "Speaker 2", ...).

## How It Works

1. **Background Music** captures system audio (what you hear in the meeting) while still letting the system volume slider and F11/F12 keys work normally
2. **Apple SpeechAnalyzer** (macOS 26+) transcribes speech to text on-device via a Swift sidecar binary
3. **Claude** detects customer questions from the transcript
4. **Web search** finds relevant Salesforce docs and pages
5. **Claude** synthesizes a concise answer from the web research
6. A browser-based sidebar shows the live transcript and Q&A cards on your second monitor

## Prerequisites

### 1. Background Music (virtual audio device)

```bash
brew install --cask background-music
```

After install, launch the app once (`open -a "Background Music"`) and:
1. Set **Background Music** as your system output (System Settings → Sound → Output)
2. Click the Background Music menu-bar icon → **Preferences → Output Device → your real speakers/headphones**

This lets you hear the meeting audio AND capture it simultaneously, while keeping the system volume slider and F11/F12 keys working — unlike a BlackHole + Multi-Output Device setup, which greys out the volume controls.

### 2. PortAudio (required by sounddevice)

```bash
brew install portaudio
```

### 3. Claude (Salesforce AI Model Gateway)

The app expects `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BEDROCK_BASE_URL` to be set in your environment (these are typically already configured globally if you use Claude Code). Override the model with `ANTHROPIC_MODEL` if needed (defaults to `claude-opus-4-7`).

### 4. macOS 26+ (Tahoe)

The transcription sidecar uses Apple's `SpeechAnalyzer` API which requires macOS 26 or newer. The Swift binary builds automatically on first run; you can also build it manually with `bin/build_sidecar.sh`.

## Setup

```bash
cd sea-cret-agent

# Create virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit config (add your Gemini API key)
cp .env.example .env
```

To find your Background Music device name:
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
| `ANTHROPIC_AUTH_TOKEN` | (inherited) | Bearer token for the Salesforce AI Model Gateway |
| `ANTHROPIC_BEDROCK_BASE_URL` | (inherited) | Gateway base URL |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Claude model for detection, answers, summaries |
| `AUDIO_DEVICE` | `Background Music` | Audio input device name |
| `MIC_DEVICE` | (auto) | Microphone device for voice calibration |
| `QUESTION_CONFIDENCE_THRESHOLD` | `0.6` | Min confidence to surface a question (0.0-1.0) |
