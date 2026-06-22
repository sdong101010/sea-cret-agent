"""Capture system audio (Background Music) and the local microphone.

The callback receives three parallel streams per chunk:

    on_audio_chunk(mixed, mic, system)

  - mixed: float32 mono signal sent to the transcriber so it hears everyone
  - mic:   float32 mono mic-only signal (zeros if no mic configured)
  - system: float32 mono system-audio signal

Speaker identification uses (mic, system) separately to label "Me" via mic
energy and remote participants via embeddings on the system stream.
"""

import asyncio
import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

import config

logger = logging.getLogger(__name__)


def find_device_by_name(name: str) -> int | None:
    """Find an audio input device index by substring match on its name."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None


def list_audio_devices() -> list[dict]:
    """List all available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"]})
    return inputs


class AudioCapture:
    """Captures from the system audio device and, when configured, the mic.

    Both streams are delivered separately to the callback (along with a mixed
    version for the transcriber) so speaker identification can distinguish
    your voice (from the mic) from remote participants (from system audio).
    """

    def __init__(self, on_audio_chunk: Callable[[np.ndarray, np.ndarray, np.ndarray], None]):
        self._on_chunk = on_audio_chunk
        self._system_stream: sd.InputStream | None = None
        self._mic_stream: sd.InputStream | None = None
        self._running = False

        self._chunk_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_SECONDS)
        self._lock = threading.Lock()
        self._system_buf = np.zeros(self._chunk_samples, dtype=np.float32)
        self._mic_buf = np.zeros(self._chunk_samples, dtype=np.float32)
        self._system_ready = False
        self._mic_ready = False
        self._mic_enabled = False

    def _system_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("System audio status: %s", status)
        if not self._running:
            return
        with self._lock:
            self._system_buf[:frames] = indata[:frames, 0]
            self._system_ready = True
            self._maybe_flush()

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("Mic audio status: %s", status)
        if not self._running:
            return
        with self._lock:
            self._mic_buf[:frames] = indata[:frames, 0]
            self._mic_ready = True
            self._maybe_flush()

    def _maybe_flush(self):
        """Emit a chunk once all active sources have delivered data."""
        if not self._system_ready:
            return
        if self._mic_enabled and not self._mic_ready:
            return

        system = self._system_buf.copy()
        if self._mic_enabled:
            mic = self._mic_buf.copy()
            # Built-in laptop mics run quiet (~0.005 RMS while talking).
            # Boost into the range the speech engine expects so the user's
            # voice doesn't get drowned out by system audio in the mixed
            # signal. We pass the un-boosted mic to the speaker identifier
            # so the energy gate uses raw levels.
            mixed = system + (mic * config.MIC_GAIN)
            np.clip(mixed, -1.0, 1.0, out=mixed)
        else:
            mic = np.zeros_like(system)
            mixed = system

        self._system_ready = False
        self._mic_ready = False
        self._on_chunk(mixed, mic, system)

    async def start(self):
        system_idx = find_device_by_name(config.AUDIO_DEVICE_NAME)
        if system_idx is None:
            available = list_audio_devices()
            device_list = "\n".join(f"  [{d['index']}] {d['name']}" for d in available)
            raise RuntimeError(
                f"Audio device '{config.AUDIO_DEVICE_NAME}' not found. "
                f"Available input devices:\n{device_list}\n"
                f"Set AUDIO_DEVICE in .env to match your device name."
            )

        system_info = sd.query_devices(system_idx)
        logger.info("System audio device: %s (index %d)", system_info["name"], system_idx)

        self._system_stream = sd.InputStream(
            device=system_idx,
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype="float32",
            blocksize=self._chunk_samples,
            callback=self._system_callback,
        )

        mic_name = config.MIC_DEVICE_NAME.strip()
        if mic_name:
            mic_idx = find_device_by_name(mic_name)
            if mic_idx is None:
                logger.warning("Mic device '%s' not found — running without mic capture", mic_name)
            else:
                mic_info = sd.query_devices(mic_idx)
                logger.info("Mic device: %s (index %d)", mic_info["name"], mic_idx)
                self._mic_stream = sd.InputStream(
                    device=mic_idx,
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=self._chunk_samples,
                    callback=self._mic_callback,
                )
                self._mic_enabled = True

        self._running = True
        self._system_stream.start()
        if self._mic_stream:
            self._mic_stream.start()

        sources = "system + mic" if self._mic_enabled else "system only"
        logger.info("Audio capture started (%s, chunk=%ds, rate=%dHz)",
                     sources, config.AUDIO_CHUNK_SECONDS, config.AUDIO_SAMPLE_RATE)

    async def stop(self):
        self._running = False
        for stream in (self._system_stream, self._mic_stream):
            if stream:
                stream.stop()
                stream.close()
        self._system_stream = None
        self._mic_stream = None
        self._mic_enabled = False
        logger.info("Audio capture stopped")

    @property
    def mic_enabled(self) -> bool:
        return self._mic_enabled
