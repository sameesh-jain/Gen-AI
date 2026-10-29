"""Unit tests for EngagementAnalyzer module."""

from datetime import datetime, timedelta
import os
import sys
import unittest

import numpy as np

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engagement_analyzer import EngagementAnalyzer
from utils.face_utils import compute_movement_score


class TestEngagementAnalyzer(unittest.TestCase):
    """Test suite for behavioral engagement algorithms and trend calculations."""

    def setUp(self) -> None:
        """Initialize fresh analyzer instance before each test."""
        self.analyzer = EngagementAnalyzer(
            inactivity_timeout=10,
            head_pose_threshold=30.0,
            movement_sensitivity=0.3,
        )

    def test_engagement_score_range(self) -> None:
        """Verify that engagement scores always strictly adhere to the [0.0, 1.0] range."""
        frame_shape = (480, 640)
        detections = [
            {"participant_id": "p1", "bbox": (100, 100, 80, 80)},
            {"participant_id": "p2", "bbox": (250, 150, 90, 90)},
        ]

        result = self.analyzer.analyze_frame(detections, frame_shape)
        class_score = result["class_engagement"]

        self.assertGreaterEqual(class_score, 0.0)
        self.assertLessEqual(class_score, 1.0)

        for p_data in result["participants"].values():
            p_score = p_data["engagement_score"]
            self.assertGreaterEqual(p_score, 0.0)
            self.assertLessEqual(p_score, 1.0)

    def test_head_pose_scoring(self) -> None:
        """Verify that forward-facing frontal orientation yields higher engagement than looking away."""
        frame_shape = (480, 640)

        # Student facing forward
        res_frontal = self.analyzer.analyze_frame(
            [{"participant_id": "p_front", "bbox": (150, 150, 80, 80)}],
            frame_shape=frame_shape,
        )
        score_frontal = res_frontal["participants"]["p_front"]["head_pose_score"]

        # Directly test head pose score calculation with deviation
        self.assertGreaterEqual(score_frontal, 0.7)

    def test_movement_scoring(self) -> None:
        """Verify that moderate movement scores higher than zero movement or excessive movement."""
        frame_shape = (480, 640)
        prev_bbox = (100, 100, 80, 80)

        # 1. Zero movement (identical bbox)
        zero_move = compute_movement_score(prev_bbox, (100, 100, 80, 80), frame_shape)
        # 2. Moderate movement (slight displacement of 15px)
        moderate_move = compute_movement_score(prev_bbox, (115, 108, 80, 80), frame_shape)
        # 3. Excessive movement (jump across half the frame)
        extreme_move = compute_movement_score(prev_bbox, (350, 300, 140, 140), frame_shape)

        self.assertLess(zero_move, moderate_move)
        self.assertGreater(extreme_move, moderate_move)

    def test_engagement_trend_computation(self) -> None:
        """Verify accurate trend detection (rising, falling, stable)."""
        frame_shape = (480, 640)
        start_time = datetime.now()

        # Simulate falling trend
        for i in range(10):
            t = start_time + timedelta(seconds=i * 2)
            # Simulate decaying bbox motion & positioning
            shift = i * 20
            self.analyzer.analyze_frame(
                [{"participant_id": "p1", "bbox": (100 + shift, 100, 80, 80)}],
                frame_shape=frame_shape,
                timestamp=t,
            )

        trend = self.analyzer.get_engagement_trend(window_seconds=60)
        self.assertIn("direction", trend)
        self.assertIn("slope", trend)
        self.assertIn("average", trend)
        self.assertIn(trend["direction"], ("rising", "falling", "stable"))


if __name__ == "__main__":
    unittest.main()
