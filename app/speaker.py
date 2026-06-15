"""Speaker identification with mic gating + online clustering.

Two signals decide who's talking for each segment:

  1. Mic energy. If the local mic RMS over the segment's time range is above
     MIC_ENERGY_THRESHOLD, the user was speaking -> label "Me".
  2. Otherwise, run a resemblyzer voice embedding on the SYSTEM audio for
     that range and either match it to an existing remote speaker (cosine
     similarity > SIMILARITY_THRESHOLD) or spawn a new "Speaker N" identity.

Speakers are renameable -- the UI can call rename_speaker("Speaker 1", "Mike")
and historical labels are not retroactively rewritten (they're just a label
on already-broadcast transcript segments), but every subsequent match to that
speaker uses the new name.
"""

import logging
import threading

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

import config

logger = logging.getLogger(__name__)


# Matching threshold for "this segment belongs to existing speaker N".
# Resemblyzer same-speaker similarities typically sit in 0.78-0.95; the
# 0.6-0.78 band overlaps with cross-speaker matches, so anything below 0.78
# tends to absorb new voices into the first cluster instead of spawning new
# ones.
SIMILARITY_THRESHOLD = 0.78
# Margin within which we accept the best match but DON'T update its centroid.
# Borderline matches still get a label, but we refuse to pollute the cluster
# with a possibly-wrong embedding.
CONFIDENT_MATCH_MARGIN = 0.04
# Below this RMS the mic is considered silent for that segment.
# Real-world mic levels for typical laptop mics often run 0.003-0.02 when
# the user is speaking from a normal distance, so the threshold needs to
# be quite low to avoid mislabeling them as a remote speaker.
MIC_ENERGY_THRESHOLD = 0.002
# Below this RMS the system stream is considered silent (no remote audio).
# Used so we don't embed pure silence and produce garbage clusters.
SYSTEM_ENERGY_THRESHOLD = 0.005
# Min segment length to embed; anything shorter is too noisy for clustering
MIN_EMBED_SECONDS = 0.5
# Max simultaneous remote speakers we'll track. After this we just reuse
# the closest existing one rather than creating new clusters forever.
MAX_REMOTE_SPEAKERS = 8


class SpeakerIdentifier:
    def __init__(self):
        self._encoder: VoiceEncoder | None = None
        self._lock = threading.Lock()

        # Remote speakers, keyed by their public label ("Speaker 1", or whatever
        # the user renamed it to). Each value is a centroid embedding plus a
        # count for running-mean updates.
        self._speakers: dict[str, dict] = {}
        self._next_id: int = 1

    def load_model(self):
        logger.info("Loading speaker embedding model...")
        self._encoder = VoiceEncoder(device="cpu")
        logger.info("Speaker embedding model loaded")

    def _slice(self, audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        sr = config.AUDIO_SAMPLE_RATE
        s = max(0, int(start_s * sr))
        e = min(len(audio), int(end_s * sr))
        if e <= s:
            return np.zeros(0, dtype=np.float32)
        return audio[s:e]

    def _is_me(self, mic: np.ndarray | None, start_s: float, end_s: float) -> bool:
        """Return True if the mic was loud during this time range."""
        if mic is None or mic.size == 0:
            return False
        snippet = self._slice(mic, start_s, end_s)
        if snippet.size == 0:
            return False
        rms = float(np.sqrt(np.mean(snippet ** 2)))
        return rms >= MIC_ENERGY_THRESHOLD

    def _classify_remote(self, system: np.ndarray, start_s: float, end_s: float) -> str:
        """Return a label for a remote-speaker segment.

        Matches against existing speakers by centroid cosine similarity;
        creates a new "Speaker N" if no match.
        """
        if self._encoder is None:
            return "Unknown"
        snippet = self._slice(system, start_s, end_s)
        if snippet.size < int(MIN_EMBED_SECONDS * config.AUDIO_SAMPLE_RATE):
            return "Unknown"

        # Don't embed silent system audio -- it produces useless centroids
        # and resemblyzer log10(0) warnings.
        rms = float(np.sqrt(np.mean(snippet ** 2)))
        if rms < SYSTEM_ENERGY_THRESHOLD:
            return "Unknown"

        wav = preprocess_wav(snippet, source_sr=config.AUDIO_SAMPLE_RATE)
        if len(wav) < int(MIN_EMBED_SECONDS * config.AUDIO_SAMPLE_RATE):
            return "Unknown"

        with self._lock:
            embedding = self._encoder.embed_utterance(wav)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

            # Find best existing speaker by cosine similarity to centroid.
            best_label: str | None = None
            best_sim: float = -1.0
            for label, sp in self._speakers.items():
                centroid = sp["centroid"]
                sim = float(np.dot(centroid, embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            if best_label is not None and best_sim >= SIMILARITY_THRESHOLD:
                # Only mutate the centroid for CONFIDENT matches. A borderline
                # match still gets the label, but pulling its embedding into
                # the running mean would drift the centroid toward an "average
                # voice" and start absorbing every subsequent speaker.
                if best_sim >= SIMILARITY_THRESHOLD + CONFIDENT_MATCH_MARGIN:
                    sp = self._speakers[best_label]
                    sp["count"] += 1
                    sp["centroid"] = (
                        sp["centroid"] * (sp["count"] - 1) + embedding
                    ) / sp["count"]
                    sp["centroid"] /= np.linalg.norm(sp["centroid"]) + 1e-9
                return best_label

            # Cluster cap reached -- assign to closest existing speaker.
            if len(self._speakers) >= MAX_REMOTE_SPEAKERS and best_label is not None:
                return best_label

            # New speaker.
            label = f"Speaker {self._next_id}"
            self._next_id += 1
            self._speakers[label] = {
                "centroid": embedding,
                "count": 1,
            }
            logger.info("New remote speaker: %s (sim to nearest=%.2f)", label, best_sim)
            return label

    def identify_batch(
        self,
        mic: np.ndarray | None,
        system: np.ndarray | None,
        segments: list[dict],
    ) -> list[str]:
        """Identify speakers for multiple segments within an audio batch.

        Args:
            mic: rolling mic-only audio (float32 mono, 16kHz). May be None.
            system: rolling system-only audio (float32 mono, 16kHz). May be None.
            segments: list of dicts with 'start' and 'end' in seconds, relative
                to the rolling buffers.

        Returns:
            List of labels: "Me", "Speaker N", or "Unknown".
        """
        if system is None and mic is None:
            return ["Unknown"] * len(segments)
        labels: list[str] = []
        for seg in segments:
            start, end = seg["start"], seg["end"]
            duration = end - start

            mic_rms = 0.0
            if mic is not None:
                snippet = self._slice(mic, start, end)
                if snippet.size > 0:
                    mic_rms = float(np.sqrt(np.mean(snippet ** 2)))

            sys_rms = 0.0
            if system is not None:
                snippet = self._slice(system, start, end)
                if snippet.size > 0:
                    sys_rms = float(np.sqrt(np.mean(snippet ** 2)))

            # Channel-based attribution. If the system stream has audio, it's
            # a remote speaker -- even if the mic is also picking up that
            # audio as acoustic leak from the speakers. Only when the system
            # channel is silent do we attribute to "Me". Cross-talk gets
            # labeled as the remote speaker, which is fine -- in real life
            # whoever was talked over usually repeats themselves anyway.
            if sys_rms >= SYSTEM_ENERGY_THRESHOLD and duration >= MIN_EMBED_SECONDS:
                if system is None:
                    labels.append("Unknown")
                    continue
                label = self._classify_remote(system, start, end)
                logger.info("seg %.2f-%.2f mic_rms=%.4f sys_rms=%.4f -> %s", start, end, mic_rms, sys_rms, label)
                labels.append(label)
                continue

            if mic_rms >= MIC_ENERGY_THRESHOLD:
                logger.info("seg %.2f-%.2f mic_rms=%.4f sys_rms=%.4f -> Me", start, end, mic_rms, sys_rms)
                labels.append("Me")
                continue

            logger.info("seg %.2f-%.2f mic_rms=%.4f sys_rms=%.4f -> Unknown (silent)", start, end, mic_rms, sys_rms)
            labels.append("Unknown")
        return labels

    def rename_speaker(self, old_label: str, new_label: str) -> bool:
        """Rename a remote speaker. Returns True if the rename was applied."""
        new_label = new_label.strip()
        if not new_label or old_label == new_label:
            return False
        with self._lock:
            if old_label not in self._speakers:
                return False
            if new_label in self._speakers:
                # Merge: combine centroids by their counts.
                old = self._speakers.pop(old_label)
                target = self._speakers[new_label]
                total = old["count"] + target["count"]
                target["centroid"] = (
                    target["centroid"] * target["count"] + old["centroid"] * old["count"]
                ) / total
                target["centroid"] /= np.linalg.norm(target["centroid"]) + 1e-9
                target["count"] = total
            else:
                self._speakers[new_label] = self._speakers.pop(old_label)
        logger.info("Renamed speaker %s -> %s", old_label, new_label)
        return True

    def get_speakers(self) -> list[str]:
        with self._lock:
            return list(self._speakers.keys())

    def reset(self):
        """Clear all learned speakers (called between sessions)."""
        with self._lock:
            self._speakers.clear()
            self._next_id = 1
