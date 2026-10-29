"""Configuration settings for Multimodal Attendance & Engagement Agent.

This module provides structured configuration dataclasses for video capture,
audio analysis, engagement scoring, attendance tracking, AI recommendations,
and local storage persistence.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class VideoConfig:
    """Configuration for video capture and face processing."""
    width: int = 640
    height: int = 480
    fps: int = 30
    face_detection_confidence: float = 0.7
    face_tracking_threshold: float = 0.6
    camera_index: int = 0
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: Tuple[int, int] = (40, 40)


@dataclass
class AudioConfig:
    """Configuration for microphone capture and speech activity detection."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    noise_threshold: float = 0.02
    channels: int = 1
    mock_audio: bool = False
    vad_energy_threshold: float = 0.015
    smoothing_window: int = 5


@dataclass
class EngagementConfig:
    """Configuration for engagement analysis thresholds and weights."""
    inactivity_timeout: int = 30
    head_pose_threshold: float = 30.0  # degrees deviation
    movement_sensitivity: float = 0.3
    optimal_movement_min: float = 0.15
    optimal_movement_max: float = 0.65
    history_window_seconds: int = 60
    head_pose_weight: float = 0.45
    movement_weight: float = 0.35
    inactivity_weight: float = 0.20


@dataclass
class AttendanceConfig:
    """Configuration for attendance registry and presence evaluation."""
    min_presence_frames: int = 30
    absence_timeout: int = 60
    session_save_path: str = field(
        default_factory=lambda: str(BASE_DIR / "data" / "sessions")
    )
    face_match_threshold: float = 0.72
    late_threshold_minutes: float = 10.0


@dataclass
class AIAgentConfig:
    """Configuration for rule-based teaching advisor and analytics."""
    suggestion_interval: int = 300  # seconds between suggestions
    engagement_drop_threshold: float = 0.40
    high_engagement_threshold: float = 0.80
    fatigue_session_minutes: int = 45
    sudden_drop_rate: float = 0.25
    cooldown_seconds: int = 180


@dataclass
class StorageConfig:
    """Configuration for SQLite storage, logs, and reports."""
    db_path: str = field(
        default_factory=lambda: str(BASE_DIR / "data" / "attendance_engagement.db")
    )
    reports_dir: str = field(
        default_factory=lambda: str(BASE_DIR / "data" / "reports")
    )
    logs_dir: str = field(
        default_factory=lambda: str(BASE_DIR / "data" / "logs")
    )
    sessions_dir: str = field(
        default_factory=lambda: str(BASE_DIR / "data" / "sessions")
    )


@dataclass
class AppConfig:
    """Root configuration holding all sub-system configurations."""
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    ai_agent: AIAgentConfig = field(default_factory=AIAgentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        for directory in [
            self.storage.reports_dir,
            self.storage.logs_dir,
            self.storage.sessions_dir,
            os.path.dirname(self.storage.db_path),
        ]:
            if directory:
                os.makedirs(directory, exist_ok=True)


# Singleton instance for quick access
config = AppConfig()
config.ensure_directories()


def get_config() -> AppConfig:
    """Retrieve the application configuration instance.

    Returns:
        AppConfig: The active application configuration instance.
    """
    return config
