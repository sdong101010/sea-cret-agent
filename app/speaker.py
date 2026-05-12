"""Speaker identification using resemblyzer voice embeddings."""

import logging
import threading

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

import config

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.75


class SpeakerIdentifier:
    def __init__(self):
        self._encoder: VoiceEncoder | None = None
        self._self_embedding: np.ndarray | None = None
        self._lock = threading.Lock()
        self._calibrated = False

    def load_model(self):
        logger.info("Loading speaker embedding model...")
        self._encoder = VoiceEncoder(device="cpu")
        logger.info("Speaker embedding model loaded")

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, audio_samples: np.ndarray) -> bool:
        """Record the user's voice embedding from a calibration sample.

        Args:
            audio_samples: float32 mono audio at 16kHz, at least 2 seconds.

        Returns:
            True if calibration succeeded.
        """
        if self._encoder is None:
            return False

        wav = preprocess_wav(audio_samples, source_sr=config.AUDIO_SAMPLE_RATE)
        if len(wav) < config.AUDIO_SAMPLE_RATE:
            logger.warning("Calibration audio too short (%.1fs)", len(wav) / config.AUDIO_SAMPLE_RATE)
            return False

        with self._lock:
            self._self_embedding = self._encoder.embed_utterance(wav)
            self._calibrated = True

        logger.info("Speaker calibration complete (%.1f seconds of audio)", len(wav) / config.AUDIO_SAMPLE_RATE)
        return True

    def identify(self, audio_segment: np.ndarray) -> str:
        """Classify an audio segment as 'You' or 'Other'.

        Returns 'Unknown' if not calibrated or audio is too short.
        """
        if not self._calibrated or self._encoder is None:
            return "Unknown"

        wav = preprocess_wav(audio_segment, source_sr=config.AUDIO_SAMPLE_RATE)
        if len(wav) < config.AUDIO_SAMPLE_RATE * 0.5:
            return "Unknown"

        with self._lock:
            seg_embedding = self._encoder.embed_utterance(wav)
            similarity = np.dot(self._self_embedding, seg_embedding)

        if similarity >= SIMILARITY_THRESHOLD:
            return "You"
        return "Other"

    def identify_batch(self, audio: np.ndarray, segments: list[dict]) -> list[str]:
        """Identify speakers for multiple segments within an audio buffer.

        Args:
            audio: The full audio buffer (float32 mono, 16kHz).
            segments: List of dicts with 'start' and 'end' in seconds.

        Returns:
            List of speaker labels, one per segment.
        """
        labels = []
        sr = config.AUDIO_SAMPLE_RATE
        for seg in segments:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            end_sample = min(end_sample, len(audio))
            if end_sample - start_sample < sr * 0.5:
                labels.append("Unknown")
                continue
            snippet = audio[start_sample:end_sample]
            labels.append(self.identify(snippet))
        return labels
