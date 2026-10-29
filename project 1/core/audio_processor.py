"""Multimodal audio processing and energy-based Voice Activity Detection (VAD).

This module processes microphone audio streams or synthetic audio buffers to measure
ambient noise levels, detect speech activity, and assess vocal participation during class.
"""

from collections import deque
from datetime import datetime
import math
import struct
from typing import Any, Deque, Dict, Optional, Union

import numpy as np

from config import get_config
from utils.logger import get_logger

logger = get_logger("AudioProcessor")


class AudioProcessor:
    """Processes real-time audio streams, computes energy and speech activity metrics."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        noise_threshold: Optional[float] = None,
        mock_mode: bool = False,
    ) -> None:
        """Initialize the AudioProcessor.

        Args:
            sample_rate: Audio sampling frequency in Hz (default 16000).
            chunk_size: Number of frames per buffer chunk (default 1024).
            noise_threshold: Minimum RMS energy threshold for speech consideration.
            mock_mode: If True, operates in simulated audio mode without hardware input.
        """
        cfg = get_config()
        self.sample_rate = sample_rate or cfg.audio.sample_rate
        self.chunk_size = chunk_size or cfg.audio.chunk_size
        self.noise_threshold = noise_threshold or cfg.audio.noise_threshold
        self.mock_mode = mock_mode or cfg.audio.mock_audio

        self.noise_floor: float = 0.005
        self.recent_rms: Deque[float] = deque(maxlen=50)
        self.recent_speech_flags: Deque[bool] = deque(maxlen=50)
        self.last_speech_time: Optional[datetime] = None
        self.current_speaking_rate: float = 0.0
        self.total_chunks_processed: int = 0

    def process_chunk(self, audio_data: Union[bytes, np.ndarray, None]) -> Dict[str, Any]:
        """Process an incoming audio buffer chunk and compute speech activity indicators.

        Args:
            audio_data: Raw 16-bit PCM byte string or 1D numpy array of samples.

        Returns:
            Dict[str, Any]: Audio feature metrics including rms, speech_detected, and noise_floor.
        """
        if self.mock_mode or audio_data is None:
            return self._generate_mock_features()

        try:
            # Convert raw bytes to normalized float array [-1.0, 1.0]
            if isinstance(audio_data, bytes):
                if len(audio_data) == 0:
                    return self._generate_mock_features()
                fmt = f"{len(audio_data) // 2}h"
                samples = np.array(struct.unpack(fmt, audio_data), dtype=np.float32) / 32768.0
            elif isinstance(audio_data, np.ndarray):
                samples = audio_data.astype(np.float32)
                if samples.max() > 1.0 or samples.min() < -1.0:
                    samples = samples / 32768.0
            else:
                return self._generate_mock_features()

            if len(samples) == 0:
                return self._generate_mock_features()

            # 1. Compute Root Mean Square (RMS) energy
            rms = float(np.sqrt(np.mean(samples**2) + 1e-8))
            self.recent_rms.append(rms)

            # 2. Voice Activity Detection (VAD) via adaptive energy thresholding
            adaptive_threshold = max(self.noise_threshold, self.noise_floor * 2.2)
            speech_detected = bool(rms > adaptive_threshold)
            self.recent_speech_flags.append(speech_detected)

            # 3. Dynamic background noise floor estimation
            if not speech_detected:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            else:
                self.last_speech_time = datetime.now()

            # 4. Speaking rate estimation based on zero-crossing rate (ZCR)
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(samples))))
            zcr = float(zero_crossings / (2.0 * len(samples)))

            # If speech is detected, map ZCR to approximate syllables/speaking rate
            if speech_detected:
                self.current_speaking_rate = float(np.clip(zcr * 8.0, 1.0, 6.0))
            else:
                self.current_speaking_rate = 0.0

            self.total_chunks_processed += 1

            return {
                "rms_energy": round(rms, 4),
                "speech_detected": speech_detected,
                "noise_level": round(self.noise_floor, 4),
                "speaking_rate": round(self.current_speaking_rate, 2),
                "zcr": round(zcr, 4),
                "is_mock": False,
            }

        except Exception as err:
            logger.error(f"Error processing audio buffer: {err}")
            return self._generate_mock_features()

    def _generate_mock_features(self) -> Dict[str, Any]:
        """Generate safe fallback audio features when in mock or silent mode."""
        return {
            "rms_energy": 0.008,
            "speech_detected": False,
            "noise_level": 0.005,
            "speaking_rate": 0.0,
            "zcr": 0.05,
            "is_mock": True,
        }

    def detect_speech_activity(self) -> bool:
        """Query whether active speech is occurring right now.

        Returns:
            bool: True if recent buffers indicate active vocalization.
        """
        if not self.recent_speech_flags:
            return False
        # True if at least 30% of the recent chunks had speech
        recent_window = list(self.recent_speech_flags)[-10:]
        return sum(recent_window) >= max(1, len(recent_window) // 3)

    def get_audio_features(self) -> Dict[str, Any]:
        """Retrieve aggregated summary of recent audio environment.

        Returns:
            Dict[str, Any]: Summary dictionary of audio metrics.
        """
        avg_rms = float(np.mean(self.recent_rms)) if self.recent_rms else 0.0
        speech_ratio = (
            float(sum(self.recent_speech_flags) / len(self.recent_speech_flags))
            if self.recent_speech_flags
            else 0.0
        )

        return {
            "average_rms": round(avg_rms, 4),
            "speech_activity_ratio": round(speech_ratio, 3),
            "noise_floor": round(self.noise_floor, 4),
            "speaking_rate": round(self.current_speaking_rate, 2),
            "is_speech_active": self.detect_speech_activity(),
        }

    def get_engagement_contribution(self) -> float:
        """Calculate vocal activity modifier for the overall engagement score.

        Returns:
            float: Score multiplier or addition (0.0 to 1.0).
        """
        features = self.get_audio_features()
        ratio = features["speech_activity_ratio"]

        # Moderate vocal participation (interactive Q&A) contributes positively
        if 0.15 <= ratio <= 0.65:
            return 0.90
        elif ratio > 0.65:
            # Continuous high noise/cacophony
            return 0.50
        else:
            # Quiet class (lecture mode or attentive listening)
            return 0.75
