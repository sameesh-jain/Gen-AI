"""Unit tests for AttendanceTracker module."""

from datetime import datetime, timedelta
import os
import sys
import unittest

import numpy as np

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.attendance_tracker import AttendanceTracker


class TestAttendanceTracker(unittest.TestCase):
    """Test suite for attendance verification, sustained presence, and registry exports."""

    def setUp(self) -> None:
        """Initialize a fresh tracker instance for each test case."""
        self.tracker = AttendanceTracker(min_presence_frames=5, absence_timeout=10)
        self.tracker.register_participant("student_01", "Alice")
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_mark_attendance_with_valid_face(self) -> None:
        """Verify manual and programmatic attendance marking for a valid registered participant."""
        # Initial status should be absent
        self.assertEqual(self.tracker.registry["student_01"].status, "absent")

        # Mark as present
        success = self.tracker.mark_attendance("student_01", "present")
        self.assertTrue(success)
        self.assertEqual(self.tracker.registry["student_01"].status, "present")

        # Mark as late
        self.tracker.mark_attendance("student_01", "late")
        self.assertEqual(self.tracker.registry["student_01"].status, "late")

        # Marking an unknown participant should return False
        unknown_success = self.tracker.mark_attendance("unknown_id", "present")
        self.assertFalse(unknown_success)

    def test_sustained_presence_required(self) -> None:
        """Verify that sustained presence over min_presence_frames is required before marking present."""
        test_bbox = [(100, 100, 80, 80)]
        start_time = datetime.now()

        # Feed 3 frames (below min_presence_frames=5)
        for i in range(3):
            t = start_time + timedelta(seconds=i)
            self.tracker.process_frame(self.dummy_frame, timestamp=t, detections_override=test_bbox)

        # After only 3 detections, participant status should be 'pending'
        rec = list(self.tracker.registry.values())[0]
        self.assertEqual(rec.status, "pending")

        # Feed 2 more frames to reach 5 frames
        for i in range(3, 5):
            t = start_time + timedelta(seconds=i)
            self.tracker.process_frame(self.dummy_frame, timestamp=t, detections_override=test_bbox)

        # Now sustained presence requirement is met
        self.assertIn(rec.status, ("present", "late"))

    def test_absence_timeout(self) -> None:
        """Verify that prolonged absence transitions pending status back to absent."""
        test_bbox = [(100, 100, 80, 80)]
        start_time = datetime.now()

        # Step 1: Detect once so participant becomes pending
        self.tracker.process_frame(self.dummy_frame, timestamp=start_time, detections_override=test_bbox)
        rec = list(self.tracker.registry.values())[0]
        self.assertEqual(rec.status, "pending")

        # Step 2: Simulate elapsed time greater than absence_timeout (15 seconds > 10 seconds)
        future_time = start_time + timedelta(seconds=15)
        # Empty frame with no detections
        self.tracker.process_frame(self.dummy_frame, timestamp=future_time, detections_override=[])

        # Should revert from pending to absent
        self.assertEqual(rec.status, "absent")

    def test_export_log_format(self) -> None:
        """Verify that export_log returns properly structured records with all required fields."""
        self.tracker.mark_attendance("student_01", "present")
        logs = self.tracker.export_log()

        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1)

        record = logs[0]
        expected_keys = {
            "participant_id",
            "name",
            "status",
            "first_seen",
            "last_seen",
            "total_frames",
            "confidence",
        }
        self.assertTrue(expected_keys.issubset(record.keys()))
        self.assertEqual(record["participant_id"], "student_01")
        self.assertEqual(record["name"], "Alice")
        self.assertEqual(record["status"], "present")


if __name__ == "__main__":
    unittest.main()
