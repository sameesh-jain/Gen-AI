"""Unit tests for TeachingAgent and rule-based pedagogical intervention engine."""

from datetime import datetime
import os
import sys
import unittest

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.teaching_agent import TeachingAgent


class TestTeachingAgent(unittest.TestCase):
    """Test suite for TeachingAgent rule execution, priority assignments, and cooldowns."""

    def setUp(self) -> None:
        """Initialize fresh TeachingAgent instance for each test."""
        self.agent = TeachingAgent(
            cooldown_seconds=60,
            drop_threshold=0.40,
            high_threshold=0.80,
        )

    def test_low_engagement_suggestion(self) -> None:
        """Verify that engagement below 40% triggers an INTERACTION suggestion with HIGH priority."""
        metrics = {
            "current_engagement": 0.32,
            "engagement_trend": {"direction": "stable", "slope": 0.0, "average": 0.32},
            "session_duration_minutes": 15.0,
            "audio_features": {"speech_activity_ratio": 0.1, "noise_floor": 0.01},
            "active_students_count": 5,
        }

        suggestion = self.agent.analyze(metrics)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "INTERACTION")
        self.assertEqual(suggestion["priority"], "HIGH")
        self.assertIn("interactive", suggestion["message"].lower())

    def test_high_engagement_positive_reinforcement(self) -> None:
        """Verify that engagement above 80% triggers a CONTENT suggestion for complex concepts."""
        metrics = {
            "current_engagement": 0.88,
            "engagement_trend": {"direction": "rising", "slope": 0.01, "average": 0.88},
            "session_duration_minutes": 20.0,
            "audio_features": {"speech_activity_ratio": 0.2, "noise_floor": 0.01},
            "active_students_count": 5,
        }

        suggestion = self.agent.analyze(metrics)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "CONTENT")
        self.assertEqual(suggestion["priority"], "LOW")
        self.assertIn("complex", suggestion["message"].lower())

    def test_suggestion_priority_levels(self) -> None:
        """Verify correct mapping of priority levels across distinct pedagogical scenarios."""
        # Scenario 1: Sharp sudden engagement drop -> URGENT
        urgent_metrics = {
            "current_engagement": 0.45,
            "engagement_trend": {"direction": "falling", "slope": -0.02, "average": 0.50},
            "session_duration_minutes": 10.0,
            "audio_features": {},
            "active_students_count": 4,
        }
        sugg_urgent = self.agent.analyze(urgent_metrics)
        self.assertIsNotNone(sugg_urgent)
        self.assertEqual(sugg_urgent["priority"], "URGENT")
        self.assertEqual(sugg_urgent["category"], "PACING")

        # Scenario 2: Fatigue past 45 min -> HIGH
        self.agent.clear_history()
        fatigue_metrics = {
            "current_engagement": 0.48,
            "engagement_trend": {"direction": "falling", "slope": -0.005, "average": 0.50},
            "session_duration_minutes": 50.0,
            "audio_features": {},
            "active_students_count": 6,
        }
        sugg_fatigue = self.agent.analyze(fatigue_metrics)
        self.assertIsNotNone(sugg_fatigue)
        self.assertEqual(sugg_fatigue["priority"], "HIGH")
        self.assertEqual(sugg_fatigue["category"], "BREAK")

        # Scenario 3: Chatter + low focus -> MEDIUM
        self.agent.clear_history()
        chatter_metrics = {
            "current_engagement": 0.38,
            "engagement_trend": {"direction": "stable", "slope": 0.0, "average": 0.38},
            "session_duration_minutes": 25.0,
            "audio_features": {"speech_activity_ratio": 0.65, "noise_floor": 0.05},
            "active_students_count": 5,
        }
        sugg_chatter = self.agent.analyze(chatter_metrics)
        self.assertIsNotNone(sugg_chatter)
        self.assertEqual(sugg_chatter["priority"], "MEDIUM")
        self.assertEqual(sugg_chatter["category"], "MANAGEMENT")

    def test_no_duplicate_suggestions(self) -> None:
        """Verify that the cooldown mechanism prevents duplicate suggestions of the same category."""
        metrics = {
            "current_engagement": 0.30,
            "engagement_trend": {"direction": "stable", "slope": 0.0, "average": 0.30},
            "session_duration_minutes": 15.0,
            "audio_features": {"speech_activity_ratio": 0.1, "noise_floor": 0.01},
            "active_students_count": 5,
        }

        # First trigger should succeed
        first = self.agent.analyze(metrics)
        self.assertIsNotNone(first)
        self.assertEqual(first["category"], "INTERACTION")

        # Immediate second trigger with same conditions must be suppressed by cooldown
        second = self.agent.analyze(metrics)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
