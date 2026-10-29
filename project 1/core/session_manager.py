"""Central session lifecycle coordinator and multimodal intelligence orchestrator.

This module coordinates camera frame ingestion, microphone audio processing,
telemetry persistence, rule-based teaching interventions, and event notifications.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import uuid

import numpy as np

from agents.teaching_agent import TeachingAgent
from config import get_config
from core.attendance_tracker import AttendanceTracker
from core.audio_processor import AudioProcessor
from core.engagement_analyzer import EngagementAnalyzer
from utils.data_store import DataStore
from utils.logger import get_logger, setup_session_logger

logger = get_logger("SessionManager")


class SessionManager:
    """Manages the lifecycle of an interactive classroom observation session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        subject: str = "Computer Science",
        instructor: str = "Instructor",
        data_store: Optional[DataStore] = None,
        mock_audio: bool = False,
    ) -> None:
        """Initialize the SessionManager.

        Args:
            session_id: Optional unique identifier. Defaults to generated UUID.
            subject: Subject or course code.
            instructor: Name of teacher or professor.
            data_store: Optional database storage instance.
            mock_audio: Whether to run audio processing in simulated mode.
        """
        self.session_id: str = session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.subject: str = subject
        self.instructor: str = instructor
        self.session_title: str = f"{subject} with {instructor}"

        # Initialize logging for this session
        self.logger = setup_session_logger(self.session_id)
        self.data_store: DataStore = data_store or DataStore()

        # Core Engines
        self.attendance_tracker = AttendanceTracker()
        self.engagement_analyzer = EngagementAnalyzer()
        self.audio_processor = AudioProcessor(mock_mode=mock_audio)
        self.teaching_agent = TeachingAgent()

        # Lifecycle state
        self.is_active: bool = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.frame_count: int = 0
        self.last_suggestion_time: Optional[datetime] = None
        self.last_persisted_engagement_time: Optional[datetime] = None

        # Milestone markers (e.g. 15, 30, 45, 60 minutes)
        self.milestones_reached: set = set()

        # Event hooks
        self._event_listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {
            "on_engagement_drop": [],
            "on_attendance_change": [],
            "on_milestone_reached": [],
            "on_suggestion_generated": [],
        }

        self._previous_attendee_ids: set = set()

    def register_event_listener(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback function for a specific session event.

        Args:
            event_name: Event identifier ('on_engagement_drop', 'on_attendance_change', etc.).
            callback: Callable accepting event payload dictionary.
        """
        if event_name in self._event_listeners:
            self._event_listeners[event_name].append(callback)
        else:
            self._event_listeners[event_name] = [callback]

    def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Dispatch event notifications to all registered listeners."""
        for callback in self._event_listeners.get(event_name, []):
            try:
                callback(payload)
            except Exception as err:
                self.logger.error(f"Error executing listener for {event_name}: {err}")

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Start the classroom monitoring session.

        Args:
            metadata: Optional dictionary with custom session parameters.

        Returns:
            bool: True if started successfully, False otherwise.
        """
        if self.is_active:
            self.logger.warning(f"Session {self.session_id} is already active.")
            return True

        self.start_time = datetime.now()
        self.is_active = True
        self.attendance_tracker.set_session_start(self.start_time)

        # Preload registered attendees from database
        known_attendees = self.data_store.get_all_attendees()
        for att in known_attendees:
            emb = np.array(att["embedding"], dtype=np.float32) if att["embedding"] else None
            self.attendance_tracker.register_participant(att["id"], att["name"], emb)

        # Create session record in database
        success = self.data_store.create_session(
            session_id=self.session_id,
            subject=self.subject,
            instructor=self.instructor,
            title=self.session_title,
            metadata=metadata,
        )

        self.logger.info(f"Session '{self.session_id}' started for {self.subject} by {self.instructor}.")
        return success

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
        detections_override: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Process a video frame through attendance, engagement, and teaching agent loops.

        Args:
            frame: Video frame image.
            timestamp: Optional timestamp for this frame. Defaults to datetime.now().
            detections_override: Optional explicit list of face bounding boxes for synthetic tests.

        Returns:
            Dict[str, Any]: Consolidated snapshot of detections, engagement, and any new suggestions.
        """
        if not self.is_active:
            return {"error": "Session is not active"}

        ts = timestamp or datetime.now()
        self.frame_count += 1
        frame_shape = frame.shape[:2] if frame is not None else (480, 640)

        # 1. Attendance processing
        detections = self.attendance_tracker.process_frame(
            frame, timestamp=ts, detections_override=detections_override
        )
        current_attendees = self.attendance_tracker.get_current_attendees()
        current_attendee_ids = {a["participant_id"] for a in current_attendees}

        # Check for attendance changes
        if current_attendee_ids != self._previous_attendee_ids:
            newly_joined = current_attendee_ids - self._previous_attendee_ids
            self._previous_attendee_ids = current_attendee_ids
            self._emit_event("on_attendance_change", {
                "session_id": self.session_id,
                "current_count": len(current_attendee_ids),
                "new_attendees": list(newly_joined),
                "timestamp": ts.isoformat(),
            })

        # 2. Audio features retrieval
        audio_features = self.audio_processor.get_audio_features()
        audio_level = audio_features.get("average_rms", 0.0)

        # 3. Engagement analysis
        engagement_snapshot = self.engagement_analyzer.analyze_frame(
            detections=detections,
            frame_shape=frame_shape,
            frame=frame,
            timestamp=ts,
            audio_level=audio_level,
        )

        class_engagement = engagement_snapshot["class_engagement"]
        trend = self.engagement_analyzer.get_engagement_trend(window_seconds=60)

        # 4. Periodic persistence to SQLite (e.g. every 1-2 seconds)
        if (
            self.last_persisted_engagement_time is None
            or (ts - self.last_persisted_engagement_time).total_seconds() >= 1.0
        ):
            self.data_store.update_engagement(
                session_id=self.session_id,
                participant_id="CLASS_AVERAGE",
                head_pose_score=0.9,
                movement_score=0.8,
                inactivity_score=0.1,
                engagement_score=class_engagement,
                audio_level=audio_level,
                timestamp=ts.isoformat(),
            )

            # Persist per-participant records
            for pid, pdata in engagement_snapshot.get("participants", {}).items():
                self.data_store.update_engagement(
                    session_id=self.session_id,
                    participant_id=pid,
                    head_pose_score=pdata["head_pose_score"],
                    movement_score=pdata["movement_score"],
                    inactivity_score=pdata["inactivity_score"],
                    engagement_score=pdata["engagement_score"],
                    audio_level=audio_level,
                    timestamp=ts.isoformat(),
                )

            # Persist attendance records periodically
            for rec in self.attendance_tracker.export_log():
                self.data_store.record_attendance(
                    session_id=self.session_id,
                    participant_id=rec["participant_id"],
                    status=rec["status"],
                    first_seen=rec["first_seen"] or ts.isoformat(),
                    last_seen=rec["last_seen"] or ts.isoformat(),
                    total_frames=rec["total_frames"],
                    confidence=rec["confidence"],
                )

            self.last_persisted_engagement_time = ts

        # 5. Milestone tracking (15, 30, 45, 60 minutes)
        elapsed_minutes = (ts - self.start_time).total_seconds() / 60.0
        for milestone in [15, 30, 45, 60]:
            if elapsed_minutes >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.add(milestone)
                self._emit_event("on_milestone_reached", {
                    "session_id": self.session_id,
                    "milestone_minutes": milestone,
                    "current_engagement": class_engagement,
                })

        # 6. Engagement drop event
        if self.engagement_analyzer.detect_engagement_drop():
            self._emit_event("on_engagement_drop", {
                "session_id": self.session_id,
                "current_engagement": class_engagement,
                "trend": trend,
            })

        # 7. AI Teaching Agent recommendations
        metrics = {
            "current_engagement": class_engagement,
            "engagement_trend": trend,
            "session_duration_minutes": elapsed_minutes,
            "audio_features": audio_features,
            "active_students_count": len(current_attendees),
        }

        new_suggestion = self.teaching_agent.analyze(metrics)
        if new_suggestion:
            self.last_suggestion_time = ts
            self.data_store.store_suggestion(
                session_id=self.session_id,
                category=new_suggestion["category"],
                priority=new_suggestion["priority"],
                message=new_suggestion["message"],
                trigger_reason=new_suggestion["trigger_reason"],
                metrics_snapshot=new_suggestion.get("metrics_snapshot"),
                timestamp=new_suggestion.get("timestamp"),
            )
            self._emit_event("on_suggestion_generated", new_suggestion)

        return {
            "session_id": self.session_id,
            "timestamp": ts.isoformat(),
            "frame_index": self.frame_count,
            "detections": detections,
            "current_attendees": current_attendees,
            "engagement": engagement_snapshot,
            "trend": trend,
            "audio": audio_features,
            "latest_suggestion": new_suggestion,
        }

    def process_audio_chunk(self, audio_data: Any) -> Dict[str, Any]:
        """Process microphone audio chunk and update acoustic features.

        Args:
            audio_data: Raw audio byte buffer or numpy waveform array.

        Returns:
            Dict[str, Any]: Computed audio metrics.
        """
        return self.audio_processor.process_chunk(audio_data)

    def end(self) -> Dict[str, Any]:
        """Finalize and close the monitoring session.

        Returns:
            Dict[str, Any]: Comprehensive final session summary.
        """
        if not self.is_active:
            self.logger.warning(f"Session {self.session_id} is not active.")
            return self.get_summary()

        self.end_time = datetime.now()
        self.is_active = False

        # Final attendance flush to database
        for rec in self.attendance_tracker.export_log():
            self.data_store.record_attendance(
                session_id=self.session_id,
                participant_id=rec["participant_id"],
                status=rec["status"],
                first_seen=rec["first_seen"] or self.end_time.isoformat(),
                last_seen=rec["last_seen"] or self.end_time.isoformat(),
                total_frames=rec["total_frames"],
                confidence=rec["confidence"],
            )

        # Mark completed in database
        self.data_store.end_session(self.session_id, end_time=self.end_time.isoformat())

        self.logger.info(f"Session '{self.session_id}' concluded successfully.")
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level summary of the session state and statistics.

        Returns:
            Dict[str, Any]: Statistical overview of attendance and engagement.
        """
        elapsed_seconds = (
            (self.end_time or datetime.now()) - (self.start_time or datetime.now())
        ).total_seconds()

        attendance_records = self.attendance_tracker.export_log()
        present_count = sum(1 for a in attendance_records if a["status"] in ("present", "late"))

        return {
            "session_id": self.session_id,
            "subject": self.subject,
            "instructor": self.instructor,
            "is_active": self.is_active,
            "duration_seconds": round(elapsed_seconds, 1),
            "total_frames_processed": self.frame_count,
            "total_registered_attendees": len(attendance_records),
            "active_present_attendees": present_count,
            "current_engagement": self.engagement_analyzer.get_current_engagement(),
            "recent_suggestions_count": len(self.teaching_agent.suggestion_history),
        }
