"""Rule-based intelligent teaching advisor for real-time classroom orchestration.

This module evaluates multimodal telemetry patterns across engagement, attendance,
audio activity, and lesson duration to deliver timely, actionable pedagogy suggestions
without requiring external cloud LLM dependencies.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config import get_config
from utils.logger import get_logger

logger = get_logger("TeachingAgent")


class TeachingSuggestion:
    """Represents a generated teaching recommendation."""

    def __init__(
        self,
        category: str,
        priority: str,
        message: str,
        trigger_reason: str,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Initialize suggestion record.

        Args:
            category: Suggestion category (PACING, INTERACTION, CONTENT, BREAK, MANAGEMENT).
            priority: Priority level (LOW, MEDIUM, HIGH, URGENT).
            message: Guidance text presented to the instructor.
            trigger_reason: Pedagogical rationale and trigger condition.
            metrics_snapshot: Dictionary of metrics at the time of trigger.
            timestamp: Datetime of suggestion creation.
        """
        self.category = category
        self.priority = priority
        self.message = message
        self.trigger_reason = trigger_reason
        self.metrics_snapshot = metrics_snapshot or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion object to serializable dictionary.

        Returns:
            Dict[str, Any]: Formatted suggestion dictionary.
        """
        return {
            "category": self.category,
            "priority": self.priority,
            "message": self.message,
            "trigger_reason": self.trigger_reason,
            "timestamp": self.timestamp.isoformat(),
            "metrics_snapshot": self.metrics_snapshot,
        }


class TeachingAgent:
    """Evaluates class state and outputs contextual instructional recommendations."""

    def __init__(
        self,
        cooldown_seconds: Optional[int] = None,
        drop_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the TeachingAgent.

        Args:
            cooldown_seconds: Minimum seconds before repeating a suggestion of the same category.
            drop_threshold: Engagement threshold below which warnings are triggered.
            high_threshold: Engagement threshold above which positive reinforcement is given.
        """
        cfg = get_config()
        self.cooldown_seconds = cooldown_seconds or cfg.ai_agent.cooldown_seconds
        self.drop_threshold = drop_threshold or cfg.ai_agent.engagement_drop_threshold
        self.high_threshold = high_threshold or cfg.ai_agent.high_engagement_threshold
        self.fatigue_minutes = cfg.ai_agent.fatigue_session_minutes

        self.suggestion_history: List[TeachingSuggestion] = []
        self._last_category_timestamps: Dict[str, datetime] = {}
        self._last_generated_hash: Optional[str] = None

    def analyze(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate current classroom metrics and generate a suggestion if appropriate.

        Args:
            metrics: Dictionary containing:
                - current_engagement: float (0.0 to 1.0)
                - engagement_trend: dict(slope, average, direction)
                - session_duration_minutes: float
                - audio_features: dict(speech_activity_ratio, is_speech_active, noise_floor)
                - active_students_count: int

        Returns:
            Optional[Dict[str, Any]]: Suggestion dictionary if triggered, else None.
        """
        current_engagement = float(metrics.get("current_engagement", 0.0))
        trend = metrics.get("engagement_trend", {"direction": "stable", "slope": 0.0, "average": current_engagement})
        duration_minutes = float(metrics.get("session_duration_minutes", 0.0))
        audio = metrics.get("audio_features", {})
        active_count = int(metrics.get("active_students_count", 0))

        suggestion = self.generate_suggestion(
            current_engagement=current_engagement,
            trend=trend,
            session_duration_minutes=duration_minutes,
            audio_activity=audio,
            active_students_count=active_count,
        )

        return suggestion.to_dict() if suggestion else None

    def generate_suggestion(
        self,
        current_engagement: float,
        trend: Dict[str, Any],
        session_duration_minutes: float,
        audio_activity: Dict[str, Any],
        active_students_count: int,
    ) -> Optional[TeachingSuggestion]:
        """Core rule engine to synthesize classroom telemetry into instructional interventions.

        Args:
            current_engagement: Real-time class engagement score.
            trend: Trajectory statistics.
            session_duration_minutes: Elapsed minutes of active session.
            audio_activity: Acoustic data from microphone analysis.
            active_students_count: Number of participating students.

        Returns:
            Optional[TeachingSuggestion]: Actionable suggestion or None if conditions not met.
        """
        now = datetime.now()
        snapshot = {
            "current_engagement": current_engagement,
            "trend": trend,
            "duration_minutes": session_duration_minutes,
            "audio": audio_activity,
            "active_students": active_students_count,
        }

        # Do not emit recommendations if no students are detected
        if active_students_count == 0:
            return None

        # Rule 1: Natural Attention Fatigue (Duration > 45 mins + Dropping or low engagement)
        if session_duration_minutes >= self.fatigue_minutes and (
            current_engagement < 0.55 or trend.get("direction") == "falling"
        ):
            if self._can_trigger("BREAK", now):
                return self._create_suggestion(
                    category="BREAK",
                    priority="HIGH",
                    message="Session has reached a natural attention fatigue point (>45 min). Consider a 5-minute break or brief physical stretch.",
                    reason=f"Duration is {session_duration_minutes:.0f} mins and engagement is dropping.",
                    snapshot=snapshot,
                    timestamp=now,
                )

        # Rule 2: Sudden Sharp Drop in Engagement
        if trend.get("direction") == "falling" and trend.get("slope", 0.0) < -0.012:
            if self._can_trigger("PACING", now):
                return self._create_suggestion(
                    category="PACING",
                    priority="URGENT",
                    message="Rapid drop in student attention detected. Current pacing may be too fast or too slow; pause for a quick comprehension check.",
                    reason=f"Engagement slope dropped sharply: {trend.get('slope')}.",
                    snapshot=snapshot,
                    timestamp=now,
                )

        # Rule 3: High Audio Noise / Side-Chatter with Low Visual Engagement
        speech_ratio = audio_activity.get("speech_activity_ratio", 0.0)
        noise_floor = audio_activity.get("noise_floor", 0.0)
        if (speech_ratio > 0.50 or noise_floor > 0.04) and current_engagement < 0.45:
            if self._can_trigger("MANAGEMENT", now):
                return self._create_suggestion(
                    category="MANAGEMENT",
                    priority="MEDIUM",
                    message="High ambient chatter detected alongside low visual focus. Redirect attention with a focused cold-call question.",
                    reason=f"High noise ratio ({speech_ratio:.0%}) combined with low visual attention ({current_engagement:.0%}).",
                    snapshot=snapshot,
                    timestamp=now,
                )

        # Rule 4: Sustained Low Engagement
        if current_engagement < self.drop_threshold:
            if self._can_trigger("INTERACTION", now):
                return self._create_suggestion(
                    category="INTERACTION",
                    priority="HIGH",
                    message="Classroom engagement has dropped below 40%. Consider introducing an interactive activity, think-pair-share, or live poll.",
                    reason=f"Class average engagement is {current_engagement:.0%} (<{self.drop_threshold:.0%}).",
                    snapshot=snapshot,
                    timestamp=now,
                )

        # Rule 5: Sustained High Engagement Window (Capitalize on focus)
        if current_engagement >= self.high_threshold and trend.get("direction") in ("rising", "stable"):
            if self._can_trigger("CONTENT", now):
                return self._create_suggestion(
                    category="CONTENT",
                    priority="LOW",
                    message="Class is highly focused and engaged (>80%). This is an ideal prime window to introduce complex or challenging concepts.",
                    reason=f"High sustained focus at {current_engagement:.0%}.",
                    snapshot=snapshot,
                    timestamp=now,
                )

        return None

    def _can_trigger(self, category: str, now: datetime) -> bool:
        """Check whether a category has passed its mandatory cooldown duration.

        Args:
            category: Category string.
            now: Current timestamp.

        Returns:
            bool: True if allowed to trigger, False otherwise.
        """
        last_time = self._last_category_timestamps.get(category)
        if last_time is None:
            return True
        elapsed = (now - last_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def _create_suggestion(
        self,
        category: str,
        priority: str,
        message: str,
        reason: str,
        snapshot: Dict[str, Any],
        timestamp: datetime,
    ) -> TeachingSuggestion:
        """Instantiate, log, and register a new recommendation."""
        suggestion = TeachingSuggestion(
            category=category,
            priority=priority,
            message=message,
            trigger_reason=reason,
            metrics_snapshot=snapshot,
            timestamp=timestamp,
        )
        self._last_category_timestamps[category] = timestamp
        self.suggestion_history.append(suggestion)
        logger.info(f"TeachingAgent Recommendation [{priority}] ({category}): {message}")
        return suggestion

    def get_recent_suggestions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent recommendations formatted as dictionaries.

        Args:
            limit: Maximum count to return.

        Returns:
            List[Dict[str, Any]]: List of recent suggestions.
        """
        recent = self.suggestion_history[-limit:]
        return [s.to_dict() for s in reversed(recent)]

    def clear_history(self) -> None:
        """Clear the historical recommendations and cooldown state."""
        self.suggestion_history.clear()
        self._last_category_timestamps.clear()
        self._last_generated_hash = None
