"""Multimodal behavioral engagement analysis engine.

This module computes real-time engagement indicators per participant and across the entire
class by combining head pose orientation, physical movement dynamics, and activity patterns.
"""

from collections import deque
from datetime import datetime, timedelta
import math
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from config import get_config
from utils.face_utils import compute_movement_score, estimate_head_pose
from utils.logger import get_logger

logger = get_logger("EngagementAnalyzer")


class ParticipantEngagementState:
    """Tracks behavioral engagement metrics for a single participant over time."""

    def __init__(self, participant_id: str) -> None:
        """Initialize state tracking for participant.

        Args:
            participant_id: Unique participant ID.
        """
        self.participant_id = participant_id
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_movement_time: datetime = datetime.now()
        self.head_pose_score: float = 1.0
        self.movement_score: float = 0.8
        self.inactivity_score: float = 0.0
        self.composite_score: float = 0.85
        self.is_frontal: bool = True
        self.history: Deque[float] = deque(maxlen=120)


class EngagementAnalyzer:
    """Analyzes and aggregates engagement metrics from vision and audio telemetry."""

    def __init__(
        self,
        inactivity_timeout: Optional[int] = None,
        head_pose_threshold: Optional[float] = None,
        movement_sensitivity: Optional[float] = None,
    ) -> None:
        """Initialize the EngagementAnalyzer.

        Args:
            inactivity_timeout: Seconds of immobility before penalizing engagement.
            head_pose_threshold: Degrees of head deviation considered looking away.
            movement_sensitivity: Sensitivity factor for physical movements.
        """
        cfg = get_config()
        self.inactivity_timeout = inactivity_timeout or cfg.engagement.inactivity_timeout
        self.head_pose_threshold = head_pose_threshold or cfg.engagement.head_pose_threshold
        self.movement_sensitivity = movement_sensitivity or cfg.engagement.movement_sensitivity
        self.head_pose_weight = cfg.engagement.head_pose_weight
        self.movement_weight = cfg.engagement.movement_weight
        self.inactivity_weight = cfg.engagement.inactivity_weight

        self.participants: Dict[str, ParticipantEngagementState] = {}
        # Timeline buffer: list of dicts with timestamp, class_score, participant_scores
        self.timeline: Deque[Dict[str, Any]] = deque(maxlen=7200)  # Up to 2 hours of per-second ticks
        self.last_tick_time: Optional[datetime] = None

    def analyze_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
        audio_level: float = 0.0,
    ) -> Dict[str, Any]:
        """Analyze a video frame and compute updated engagement scores.

        Args:
            detections: List of detection dicts (participant_id, bbox, etc.) from AttendanceTracker.
            frame_shape: Height and width of the frame.
            frame: Optional full image frame for landmark head pose estimation.
            timestamp: Timestamp of the frame. Defaults to datetime.now().
            audio_level: Current ambient or speech audio RMS level.

        Returns:
            Dict[str, Any]: Snapshot containing class_engagement and per-participant breakdown.
        """
        ts = timestamp or datetime.now()
        participant_results: Dict[str, Dict[str, Any]] = {}
        active_scores: List[float] = []

        for det in detections:
            pid = det["participant_id"]
            bbox = det["bbox"]

            if pid not in self.participants:
                self.participants[pid] = ParticipantEngagementState(pid)
            state = self.participants[pid]

            # 1. Compute head pose orientation score
            if frame is not None:
                yaw, pitch, roll, is_frontal = estimate_head_pose(frame, bbox)
            else:
                # Fallback estimation based on detection metadata
                yaw, pitch, roll, is_frontal = 0.0, 0.0, 0.0, True

            state.is_frontal = is_frontal
            deviation = math.sqrt(yaw**2 + pitch**2)

            if deviation <= 15.0:
                head_pose_score = 1.0
            elif deviation <= self.head_pose_threshold:
                # Smooth decrease from 1.0 to 0.7
                fraction = (deviation - 15.0) / max(1.0, (self.head_pose_threshold - 15.0))
                head_pose_score = 1.0 - 0.3 * fraction
            else:
                # Decreases down to 0.15 when turning away significantly
                excess = deviation - self.head_pose_threshold
                head_pose_score = max(0.15, 0.7 - (excess / 60.0) * 0.55)

            state.head_pose_score = float(head_pose_score)

            # 2. Compute movement dynamics score
            raw_movement = compute_movement_score(state.last_bbox, bbox, frame_shape)
            state.last_bbox = bbox

            # Update inactivity timer if meaningful motion detected
            if raw_movement > 0.08:
                state.last_movement_time = ts

            # Moderate movement is ideal; stillness or wild flailing drops score
            if 0.10 <= raw_movement <= 0.60:
                # Optimal attentive engagement zone
                movement_score = 0.85 + 0.15 * (1.0 - abs(raw_movement - 0.35) / 0.25)
            elif raw_movement < 0.10:
                # Static / still posture
                movement_score = 0.40 + (raw_movement / 0.10) * 0.45
            else:
                # Excessive motion / restless distraction
                movement_score = max(0.20, 0.85 - ((raw_movement - 0.60) / 0.40) * 0.65)

            state.movement_score = float(movement_score)

            # 3. Compute inactivity penalty
            idle_seconds = max(0.0, (ts - state.last_movement_time).total_seconds())
            if idle_seconds > self.inactivity_timeout:
                inactivity_score = min(1.0, (idle_seconds - self.inactivity_timeout) / 60.0)
            else:
                inactivity_score = 0.0

            state.inactivity_score = float(inactivity_score)

            # 4. Composite participant engagement score
            composite = (
                self.head_pose_weight * state.head_pose_score
                + self.movement_weight * state.movement_score
                + self.inactivity_weight * (1.0 - state.inactivity_score)
            )
            composite = float(np.clip(composite, 0.0, 1.0))
            state.composite_score = composite
            state.history.append(composite)

            active_scores.append(composite)
            participant_results[pid] = {
                "participant_id": pid,
                "engagement_score": round(composite, 3),
                "head_pose_score": round(state.head_pose_score, 3),
                "movement_score": round(state.movement_score, 3),
                "inactivity_score": round(state.inactivity_score, 3),
                "is_frontal": state.is_frontal,
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
            }

        # Class aggregate engagement
        if active_scores:
            class_score = float(np.mean(active_scores))
        else:
            class_score = 0.0

        snapshot = {
            "timestamp": ts.isoformat(),
            "class_engagement": round(class_score, 3),
            "participant_count": len(active_scores),
            "participants": participant_results,
            "audio_level": round(audio_level, 4),
        }

        # Store in timeline
        self.timeline.append(snapshot)
        self.last_tick_time = ts

        return snapshot

    def get_current_engagement(self) -> float:
        """Get the latest class-wide average engagement score.

        Returns:
            float: Engagement score from 0.0 to 1.0.
        """
        if not self.timeline:
            return 0.0
        return self.timeline[-1].get("class_engagement", 0.0)

    def get_participant_engagement(self, participant_id: str) -> Optional[float]:
        """Get the current engagement score of a specific participant.

        Args:
            participant_id: ID of the participant.

        Returns:
            Optional[float]: Engagement score or None if not found.
        """
        if participant_id in self.participants:
            return self.participants[participant_id].composite_score
        return None

    def get_engagement_trend(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Compute the trajectory and statistical trend of engagement over a recent window.

        Args:
            window_seconds: Time horizon in seconds to analyze.

        Returns:
            Dict[str, Any]: Trend dictionary containing average, slope, and direction.
        """
        if len(self.timeline) < 2:
            return {"slope": 0.0, "average": self.get_current_engagement(), "direction": "stable"}

        now = self.timeline[-1]["timestamp"]
        now_dt = datetime.fromisoformat(now)
        cutoff = now_dt - timedelta(seconds=window_seconds)

        window_scores = [
            t["class_engagement"]
            for t in self.timeline
            if datetime.fromisoformat(t["timestamp"]) >= cutoff
        ]

        if not window_scores:
            return {"slope": 0.0, "average": self.get_current_engagement(), "direction": "stable"}

        avg_score = float(np.mean(window_scores))

        # Linear regression slope over the window
        if len(window_scores) >= 3:
            x = np.arange(len(window_scores))
            y = np.array(window_scores)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = window_scores[-1] - window_scores[0]

        if slope > 0.005:
            direction = "rising"
        elif slope < -0.005:
            direction = "falling"
        else:
            direction = "stable"

        return {
            "slope": round(slope, 5),
            "average": round(avg_score, 3),
            "direction": direction,
            "sample_count": len(window_scores),
        }

    def detect_engagement_drop(
        self, threshold: float = 0.40, window_seconds: int = 120
    ) -> bool:
        """Detect whether engagement has dropped below critical levels or declined rapidly.

        Args:
            threshold: Absolute score threshold denoting low engagement.
            window_seconds: Seconds window over which to measure sustained drop.

        Returns:
            bool: True if an engagement drop condition is detected.
        """
        if len(self.timeline) < 5:
            return False

        trend = self.get_engagement_trend(window_seconds=window_seconds)
        current = self.get_current_engagement()

        # Condition 1: Average in window is below critical threshold
        if trend["average"] < threshold and current < threshold:
            return True

        # Condition 2: Steep downward slope
        if trend["direction"] == "falling" and trend["slope"] < -0.015:
            return True

        return False
