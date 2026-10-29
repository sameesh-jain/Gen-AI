"""Privacy-preserving attendance tracking engine with sustained presence verification.

This module detects, matches, and tracks student presence using sliding-window verification
to prevent false-positive transient detections and accurately record attendance.
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import get_config
from utils.face_utils import detect_faces, encode_face, match_face
from utils.logger import get_logger

logger = get_logger("AttendanceTracker")


class ParticipantRecord:
    """Internal state tracking structure for an individual participant."""

    def __init__(self, participant_id: str, name: str, embedding: Optional[np.ndarray] = None) -> None:
        """Initialize participant tracking state.

        Args:
            participant_id: Unique participant ID.
            name: Display name.
            embedding: Facial feature vector.
        """
        self.participant_id = participant_id
        self.name = name
        self.embedding = embedding
        self.first_seen: Optional[datetime] = None
        self.last_seen: Optional[datetime] = None
        self.total_frames: int = 0
        self.confidence: float = 0.0
        self.status: str = "absent"  # 'absent', 'pending', 'present', 'late'
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.recent_detections: Deque[int] = deque(maxlen=60)
        self.missed_consecutive_frames: int = 0


class AttendanceTracker:
    """Tracks attendance in real-time using face detection, feature matching, and sliding windows."""

    def __init__(
        self,
        known_participants: Optional[Dict[str, Dict[str, Any]]] = None,
        min_presence_frames: Optional[int] = None,
        absence_timeout: Optional[int] = None,
    ) -> None:
        """Initialize the AttendanceTracker.

        Args:
            known_participants: Dictionary mapping participant_id to dict(name=..., embedding=...).
            min_presence_frames: Minimum frame detections required for sustained presence.
            absence_timeout: Number of seconds without detection before marking inactive.
        """
        cfg = get_config()
        self.min_presence_frames = min_presence_frames or cfg.attendance.min_presence_frames
        self.absence_timeout = absence_timeout or cfg.attendance.absence_timeout
        self.match_threshold = cfg.attendance.face_match_threshold
        self.late_threshold_minutes = cfg.attendance.late_threshold_minutes

        self.session_start_time: datetime = datetime.now()
        self.registry: Dict[str, ParticipantRecord] = {}
        self._next_guest_index: int = 1

        if known_participants:
            for pid, info in known_participants.items():
                emb = np.array(info["embedding"], dtype=np.float32) if info.get("embedding") else None
                self.register_participant(pid, info.get("name", pid), emb)

    def set_session_start(self, start_time: datetime) -> None:
        """Set or update the official session start time for calculating lateness.

        Args:
            start_time: Datetime object representing session kick-off.
        """
        self.session_start_time = start_time

    def register_participant(
        self,
        participant_id: str,
        name: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Register a known student in the tracker registry.

        Args:
            participant_id: Unique participant identifier.
            name: Display name of the participant.
            embedding: Normalized 256-D face descriptor.
        """
        if participant_id in self.registry:
            self.registry[participant_id].name = name
            if embedding is not None:
                self.registry[participant_id].embedding = embedding
        else:
            self.registry[participant_id] = ParticipantRecord(participant_id, name, embedding)
        logger.debug(f"Registered participant '{name}' ({participant_id})")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[datetime] = None,
        detections_override: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process a single video frame, update detections, and compute attendance status.

        Args:
            frame: BGR video frame from camera or simulator.
            timestamp: Optional timestamp for this frame. Defaults to datetime.now().
            detections_override: Optional explicit list of face bounding boxes (x, y, w, h)
                used for synthetic simulations and automated tests.

        Returns:
            List[Dict[str, Any]]: List of active detections with participant details, bboxes, and embeddings.
        """
        ts = timestamp or datetime.now()
        detected_pids: List[str] = []
        frame_detections: List[Dict[str, Any]] = []

        if frame is None or frame.size == 0:
            self._handle_no_detections(ts)
            return []

        try:
            # 1. Detect faces in frame or use override
            if detections_override is not None:
                bboxes = detections_override
            else:
                bboxes = detect_faces(frame)

            if not bboxes:
                self._handle_no_detections(ts)
                return []

            # 2. Extract embeddings and match each detected face
            active_embeddings = {
                pid: r.embedding
                for pid, r in self.registry.items()
                if r.embedding is not None
            }

            for bbox in bboxes:
                x, y, w, h = bbox
                # Safe crop
                fh, fw = frame.shape[:2]
                rx, ry = max(0, x), max(0, y)
                rw, rh = min(w, fw - rx), min(h, fh - ry)

                if rw < 20 or rh < 20:
                    continue

                face_roi = frame[ry : ry + rh, rx : rx + rw]
                embedding = encode_face(face_roi)

                # Match against known registry
                matched_id, match_conf = match_face(
                    embedding, active_embeddings, threshold=self.match_threshold
                )

                if matched_id is not None:
                    pid = matched_id
                    rec = self.registry[pid]
                    # Update running embedding prototype with running average
                    if rec.embedding is not None:
                        rec.embedding = 0.85 * rec.embedding + 0.15 * embedding
                        norm = np.linalg.norm(rec.embedding)
                        if norm > 0:
                            rec.embedding /= norm
                else:
                    # Assign new guest participant
                    pid = f"student_{self._next_guest_index:02d}"
                    self._next_guest_index += 1
                    name = f"Student {self._next_guest_index - 1}"
                    self.register_participant(pid, name, embedding)
                    rec = self.registry[pid]
                    match_conf = 1.0

                detected_pids.append(pid)

                # Update participant metrics
                if rec.first_seen is None:
                    rec.first_seen = ts

                rec.last_seen = ts
                rec.total_frames += 1
                rec.confidence = float(match_conf)
                rec.last_bbox = bbox
                rec.recent_detections.append(1)
                rec.missed_consecutive_frames = 0

                # Evaluate sustained presence requirement
                sustained_count = sum(rec.recent_detections)
                if sustained_count >= self.min_presence_frames:
                    if rec.status in ("absent", "pending"):
                        # Check lateness threshold
                        elapsed_min = (rec.first_seen - self.session_start_time).total_seconds() / 60.0
                        if elapsed_min > self.late_threshold_minutes:
                            rec.status = "late"
                        else:
                            rec.status = "present"
                        logger.info(f"Participant '{rec.name}' verified as {rec.status.upper()}.")
                elif rec.status == "absent":
                    rec.status = "pending"

                frame_detections.append({
                    "participant_id": pid,
                    "name": rec.name,
                    "bbox": bbox,
                    "embedding": embedding,
                    "confidence": match_conf,
                    "status": rec.status,
                    "is_present": rec.status in ("present", "late"),
                })

            # 3. Handle participants not detected in this frame
            for pid, rec in self.registry.items():
                if pid not in detected_pids:
                    rec.recent_detections.append(0)
                    rec.missed_consecutive_frames += 1

                    # Check absence timeout if previously present
                    if rec.last_seen is not None and rec.status in ("present", "late", "pending"):
                        idle_seconds = (ts - rec.last_seen).total_seconds()
                        if idle_seconds > self.absence_timeout:
                            # Participant marked absent after prolonged timeout
                            if rec.status == "pending":
                                rec.status = "absent"

            return frame_detections

        except Exception as err:
            logger.error(f"Error processing frame in AttendanceTracker: {err}")
            return []

    def _handle_no_detections(self, timestamp: datetime) -> None:
        """Handle state update when zero faces are detected in the current frame."""
        for rec in self.registry.values():
            rec.recent_detections.append(0)
            rec.missed_consecutive_frames += 1
            if rec.last_seen is not None and rec.status == "pending":
                idle_seconds = (timestamp - rec.last_seen).total_seconds()
                if idle_seconds > self.absence_timeout:
                    rec.status = "absent"

    def mark_attendance(self, participant_id: str, status: str) -> bool:
        """Manually mark or override the attendance status of a participant.

        Args:
            participant_id: ID of the participant.
            status: Status string ('present', 'late', 'absent').

        Returns:
            bool: True if updated, False if participant not found.
        """
        valid_statuses = {"present", "late", "absent", "pending"}
        if status not in valid_statuses:
            logger.warning(f"Invalid attendance status: '{status}'")
            return False

        if participant_id in self.registry:
            self.registry[participant_id].status = status
            logger.info(f"Manually set attendance for '{participant_id}' to '{status}'")
            return True

        logger.warning(f"Participant '{participant_id}' not found in registry.")
        return False

    def get_current_attendees(self) -> List[Dict[str, Any]]:
        """Retrieve list of participants who are currently actively present in the session.

        Returns:
            List[Dict[str, Any]]: List of attendees with recent active status.
        """
        attendees = []
        for rec in self.registry.values():
            if rec.status in ("present", "late"):
                attendees.append({
                    "participant_id": rec.participant_id,
                    "name": rec.name,
                    "status": rec.status,
                    "first_seen": rec.first_seen.isoformat() if rec.first_seen else None,
                    "last_seen": rec.last_seen.isoformat() if rec.last_seen else None,
                    "total_frames": rec.total_frames,
                    "confidence": rec.confidence,
                })
        return attendees

    def get_all_attendance_status(self) -> List[Dict[str, Any]]:
        """Retrieve the attendance records for all participants in the registry.

        Returns:
            List[Dict[str, Any]]: Complete list of all participants with attendance metrics.
        """
        return self.export_log()

    def export_log(self) -> List[Dict[str, Any]]:
        """Export full attendance records for database persistence or reporting.

        Returns:
            List[Dict[str, Any]]: Serialized list of participant attendance data.
        """
        records = []
        for rec in self.registry.values():
            records.append({
                "participant_id": rec.participant_id,
                "name": rec.name,
                "status": rec.status,
                "first_seen": rec.first_seen.isoformat() if rec.first_seen else None,
                "last_seen": rec.last_seen.isoformat() if rec.last_seen else None,
                "total_frames": rec.total_frames,
                "confidence": round(rec.confidence, 3),
            })
        return records
