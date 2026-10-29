"""SQLite-based data persistence layer for sessions, attendance, engagement, and AI suggestions.

This module provides thread-safe access to a local SQLite database storing all
multimodal classroom telemetry, historical records, and generated analytical reports.
"""

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from config import get_config
from utils.logger import get_logger

logger = get_logger("DataStore")


class DataStore:
    """Manages SQLite database storage for sessions, attendees, engagement, and suggestions."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the DataStore and ensure database schema existence.

        Args:
            db_path: Optional file path to the SQLite database. Defaults to config path.
        """
        cfg = get_config()
        self.db_path = db_path or cfg.storage.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for acquiring and safely closing a SQLite connection.

        Yields:
            sqlite3.Connection: An open SQLite connection.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as err:
            conn.rollback()
            logger.error(f"Database transaction error: {err}")
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create necessary database tables and indices if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    subject TEXT NOT NULL,
                    instructor TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata TEXT
                )
            """)

            # Registered attendees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendees (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    embedding TEXT
                )
            """)

            # Attendance records per session
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_frames INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, participant_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)

            # Engagement time-series records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engagement_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    head_pose_score REAL NOT NULL,
                    movement_score REAL NOT NULL,
                    inactivity_score REAL NOT NULL,
                    engagement_score REAL NOT NULL,
                    audio_level REAL NOT NULL DEFAULT 0.0,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)

            # AI teaching suggestions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    metrics_snapshot TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_engagement_session ON engagement_records(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_session ON suggestions(session_id)")

    def create_session(
        self,
        session_id: str,
        subject: str,
        instructor: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new classroom session record.

        Args:
            session_id: Unique identifier for the session.
            subject: Subject or course title.
            instructor: Name of the teacher or instructor.
            title: Optional friendly title. Defaults to '{subject} Session'.
            metadata: Optional dictionary with session parameters or settings.

        Returns:
            bool: True if created successfully, False otherwise.
        """
        now = datetime.now().isoformat()
        session_title = title or f"{subject} - {now[:10]}"
        meta_json = json.dumps(metadata or {})

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO sessions (id, title, subject, instructor, start_time, status, metadata)
                    VALUES (?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (session_id, session_title, subject, instructor, now, meta_json),
                )
            logger.info(f"Session '{session_id}' created in database.")
            return True
        except Exception as err:
            logger.error(f"Failed to create session {session_id}: {err}")
            return False

    def end_session(self, session_id: str, end_time: Optional[str] = None) -> bool:
        """Mark a session as completed with an end timestamp.

        Args:
            session_id: Identifier of the session to end.
            end_time: Optional ISO timestamp. Defaults to current time.

        Returns:
            bool: True if updated successfully, False otherwise.
        """
        now = end_time or datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE sessions
                    SET end_time = ?, status = 'completed'
                    WHERE id = ?
                    """,
                    (now, session_id),
                )
            logger.info(f"Session '{session_id}' marked as completed.")
            return True
        except Exception as err:
            logger.error(f"Failed to end session {session_id}: {err}")
            return False

    def register_attendee(
        self, participant_id: str, name: str, embedding: Optional[List[float]] = None
    ) -> bool:
        """Register a known student or participant in the master registry.

        Args:
            participant_id: Unique participant ID.
            name: Human-readable name.
            embedding: Optional facial feature vector for recognition.

        Returns:
            bool: True if registered or updated, False otherwise.
        """
        now = datetime.now().isoformat()
        emb_json = json.dumps(embedding) if embedding is not None else None
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendees (id, name, registered_at, embedding)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        embedding = COALESCE(excluded.embedding, attendees.embedding)
                    """,
                    (participant_id, name, now, emb_json),
                )
            return True
        except Exception as err:
            logger.error(f"Failed to register attendee {participant_id}: {err}")
            return False

    def get_all_attendees(self) -> List[Dict[str, Any]]:
        """Retrieve all registered participants from the attendees table.

        Returns:
            List[Dict[str, Any]]: List of attendee dicts with id, name, and embedding.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, registered_at, embedding FROM attendees")
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    emb = json.loads(row["embedding"]) if row["embedding"] else None
                    results.append({
                        "id": row["id"],
                        "name": row["name"],
                        "registered_at": row["registered_at"],
                        "embedding": emb,
                    })
                return results
        except Exception as err:
            logger.error(f"Failed to fetch attendees: {err}")
            return []

    def record_attendance(
        self,
        session_id: str,
        participant_id: str,
        status: str,
        first_seen: str,
        last_seen: str,
        total_frames: int,
        confidence: float,
    ) -> bool:
        """Insert or update attendance status for a participant in a session.

        Args:
            session_id: Session identifier.
            participant_id: Participant identifier.
            status: Attendance state ('present', 'late', 'absent').
            first_seen: ISO timestamp of first appearance.
            last_seen: ISO timestamp of latest appearance.
            total_frames: Number of frames participant was detected.
            confidence: Recognition or presence confidence score.

        Returns:
            bool: True if written successfully, False otherwise.
        """
        now = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendance_records
                    (session_id, participant_id, status, first_seen, last_seen, total_frames, confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, participant_id) DO UPDATE SET
                        status = excluded.status,
                        last_seen = excluded.last_seen,
                        total_frames = excluded.total_frames,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, participant_id, status, first_seen, last_seen, total_frames, confidence, now),
                )
            return True
        except Exception as err:
            logger.error(f"Failed to record attendance for {participant_id} in {session_id}: {err}")
            return False

    def get_session_attendance(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetch all attendance records for a given session.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: List of attendance dictionaries.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT a.participant_id, COALESCE(att.name, a.participant_id) as name,
                           a.status, a.first_seen, a.last_seen, a.total_frames, a.confidence, a.updated_at
                    FROM attendance_records a
                    LEFT JOIN attendees att ON a.participant_id = att.id
                    WHERE a.session_id = ?
                    ORDER BY a.participant_id ASC
                    """,
                    (session_id,),
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as err:
            logger.error(f"Failed to get attendance for session {session_id}: {err}")
            return []

    def update_engagement(
        self,
        session_id: str,
        participant_id: str,
        head_pose_score: float,
        movement_score: float,
        inactivity_score: float,
        engagement_score: float,
        audio_level: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Record an engagement telemetry snapshot for a participant.

        Args:
            session_id: Session identifier.
            participant_id: Participant identifier or 'CLASS_AVERAGE'.
            head_pose_score: Score from 0.0 to 1.0.
            movement_score: Score from 0.0 to 1.0.
            inactivity_score: Score from 0.0 to 1.0.
            engagement_score: Aggregated composite score from 0.0 to 1.0.
            audio_level: Ambient/speech audio RMS level.
            timestamp: ISO timestamp of record. Defaults to now.

        Returns:
            bool: True if recorded successfully, False otherwise.
        """
        ts = timestamp or datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO engagement_records
                    (session_id, timestamp, participant_id, head_pose_score, movement_score,
                     inactivity_score, engagement_score, audio_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        ts,
                        participant_id,
                        float(head_pose_score),
                        float(movement_score),
                        float(inactivity_score),
                        float(engagement_score),
                        float(audio_level),
                    ),
                )
            return True
        except Exception as err:
            logger.error(f"Failed to record engagement: {err}")
            return False

    def get_engagement_history(
        self, session_id: str, participant_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve engagement history records for a session.

        Args:
            session_id: Session identifier.
            participant_id: Optional filter for specific participant or 'CLASS_AVERAGE'.
            limit: Optional limit on the number of records returned.

        Returns:
            List[Dict[str, Any]]: Chronological engagement records.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM engagement_records WHERE session_id = ?"
                params: List[Any] = [session_id]

                if participant_id:
                    query += " AND participant_id = ?"
                    params.append(participant_id)

                query += " ORDER BY timestamp ASC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as err:
            logger.error(f"Failed to fetch engagement history: {err}")
            return []

    def store_suggestion(
        self,
        session_id: str,
        category: str,
        priority: str,
        message: str,
        trigger_reason: str,
        metrics_snapshot: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Store an AI teaching suggestion generated during the session.

        Args:
            session_id: Session identifier.
            category: Category (e.g. PACING, INTERACTION, CONTENT, BREAK, MANAGEMENT).
            priority: Priority level (LOW, MEDIUM, HIGH, URGENT).
            message: Suggestion guidance text for the instructor.
            trigger_reason: Reason why rule was triggered.
            metrics_snapshot: Dictionary containing current telemetry indicators.
            timestamp: ISO timestamp.

        Returns:
            bool: True if stored successfully, False otherwise.
        """
        ts = timestamp or datetime.now().isoformat()
        snapshot_json = json.dumps(metrics_snapshot or {})
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO suggestions
                    (session_id, timestamp, category, priority, message, trigger_reason, metrics_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, ts, category, priority, message, trigger_reason, snapshot_json),
                )
            logger.info(f"Teaching suggestion stored for session {session_id} [{priority}]: {message}")
            return True
        except Exception as err:
            logger.error(f"Failed to store suggestion: {err}")
            return False

    def get_session_suggestions(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetch all teaching suggestions generated during a session.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: List of suggestion records.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM suggestions WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    r = dict(row)
                    r["metrics_snapshot"] = json.loads(r["metrics_snapshot"]) if r["metrics_snapshot"] else {}
                    results.append(r)
                return results
        except Exception as err:
            logger.error(f"Failed to fetch suggestions for session {session_id}: {err}")
            return []

    def get_session_report(self, session_id: str) -> Dict[str, Any]:
        """Aggregate comprehensive report data for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            Dict[str, Any]: Dictionary containing session info, attendance, engagement metrics,
            and teaching recommendations.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
                session_row = cursor.fetchone()

                if not session_row:
                    return {"error": f"Session '{session_id}' not found"}

                session_info = dict(session_row)
                session_info["metadata"] = json.loads(session_info["metadata"]) if session_info["metadata"] else {}

            attendance = self.get_session_attendance(session_id)
            engagement = self.get_engagement_history(session_id, participant_id="CLASS_AVERAGE")
            suggestions = self.get_session_suggestions(session_id)

            return {
                "session": session_info,
                "attendance": attendance,
                "engagement_timeline": engagement,
                "suggestions": suggestions,
            }
        except Exception as err:
            logger.error(f"Failed to compile report for {session_id}: {err}")
            return {"error": str(err)}

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all past and ongoing classroom sessions.

        Returns:
            List[Dict[str, Any]]: List of summary session records.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT s.id, s.title, s.subject, s.instructor, s.start_time, s.end_time, s.status,
                           COUNT(DISTINCT a.participant_id) as attendee_count
                    FROM sessions s
                    LEFT JOIN attendance_records a ON s.id = a.session_id
                    GROUP BY s.id
                    ORDER BY s.start_time DESC
                    """
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as err:
            logger.error(f"Failed to list sessions: {err}")
            return []
