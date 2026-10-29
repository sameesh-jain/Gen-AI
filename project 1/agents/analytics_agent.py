"""Post-session classroom analytics and longitudinal intelligence engine.

This module computes key performance indicators, attention fatigue curves,
participation distributions, and comparative historical performance across sessions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from utils.data_store import DataStore
from utils.logger import get_logger

logger = get_logger("AnalyticsAgent")


class AnalyticsAgent:
    """Performs deep post-session analytical evaluations on stored classroom telemetry."""

    def __init__(self, data_store: Optional[DataStore] = None) -> None:
        """Initialize the AnalyticsAgent.

        Args:
            data_store: Optional DataStore instance. Defaults to default SQLite database.
        """
        self.data_store = data_store or DataStore()

    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """Perform comprehensive analytics for a completed or ongoing session.

        Args:
            session_id: Unique identifier for the classroom session.

        Returns:
            Dict[str, Any]: Comprehensive analytical evaluation with metrics, insights, and distributions.
        """
        report_data = self.data_store.get_session_report(session_id)
        if "error" in report_data:
            return report_data

        session_info = report_data["session"]
        attendance = report_data["attendance"]
        timeline = report_data["engagement_timeline"]
        suggestions = report_data["suggestions"]

        # 1. Attendance Metrics
        total_attendees = len(attendance)
        present_count = sum(1 for a in attendance if a["status"] in ("present", "late"))
        late_count = sum(1 for a in attendance if a["status"] == "late")
        absent_count = sum(1 for a in attendance if a["status"] == "absent")

        all_registered = self.data_store.get_all_attendees()
        registered_count = max(total_attendees, len(all_registered))
        attendance_rate = float(present_count / max(1, registered_count))

        # 2. Engagement Metrics
        if timeline:
            scores = [float(r["engagement_score"]) for r in timeline]
            avg_engagement = float(np.mean(scores))
            std_engagement = float(np.std(scores))
            max_idx = int(np.argmax(scores))
            min_idx = int(np.argmin(scores))

            peak_time = timeline[max_idx]["timestamp"]
            lowest_time = timeline[min_idx]["timestamp"]
            peak_score = float(scores[max_idx])
            lowest_score = float(scores[min_idx])
        else:
            avg_engagement = 0.0
            std_engagement = 0.0
            peak_time = "N/A"
            lowest_time = "N/A"
            peak_score = 0.0
            lowest_score = 0.0

        # 3. Attention Fatigue Curve (time-bucketed into 5-minute intervals)
        fatigue_curve = self._compute_fatigue_curve(timeline)

        # 4. Synthesize Key Pedagogical Insights
        insights = self._generate_key_insights(
            avg_engagement=avg_engagement,
            attendance_rate=attendance_rate,
            late_count=late_count,
            peak_score=peak_score,
            lowest_score=lowest_score,
            fatigue_curve=fatigue_curve,
            suggestion_count=len(suggestions),
        )

        return {
            "session_id": session_id,
            "session_info": session_info,
            "summary_metrics": {
                "attendance_rate": round(attendance_rate, 3),
                "total_students": total_attendees,
                "present_count": present_count,
                "late_count": late_count,
                "absent_count": absent_count,
                "average_engagement": round(avg_engagement, 3),
                "engagement_std_dev": round(std_engagement, 3),
                "peak_engagement_score": round(peak_score, 3),
                "peak_engagement_time": peak_time,
                "lowest_engagement_score": round(lowest_score, 3),
                "lowest_engagement_time": lowest_time,
                "total_suggestions_given": len(suggestions),
            },
            "attention_fatigue_curve": fatigue_curve,
            "key_insights": insights,
            "attendance_records": attendance,
            "suggestions": suggestions,
        }

    def _compute_fatigue_curve(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate time-bucketed engagement averages to model attention fatigue.

        Args:
            timeline: Chronological engagement records.

        Returns:
            List[Dict[str, Any]]: Bucketed time series with interval index and average score.
        """
        if not timeline:
            return []

        # Bucket size: 300 seconds (5 minutes)
        buckets: Dict[int, List[float]] = {}
        first_ts = datetime.fromisoformat(timeline[0]["timestamp"])

        for item in timeline:
            ts = datetime.fromisoformat(item["timestamp"])
            elapsed_sec = max(0, (ts - first_ts).total_seconds())
            bucket_idx = int(elapsed_sec // 300)

            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(float(item["engagement_score"]))

        fatigue_curve = []
        for idx in sorted(buckets.keys()):
            mean_score = float(np.mean(buckets[idx]))
            fatigue_curve.append({
                "minute_start": idx * 5,
                "minute_end": (idx + 1) * 5,
                "average_engagement": round(mean_score, 3),
                "sample_points": len(buckets[idx]),
            })

        return fatigue_curve

    def _generate_key_insights(
        self,
        avg_engagement: float,
        attendance_rate: float,
        late_count: int,
        peak_score: float,
        lowest_score: float,
        fatigue_curve: List[Dict[str, Any]],
        suggestion_count: int,
    ) -> List[str]:
        """Generate human-readable analytical observations for the educator.

        Returns:
            List[str]: Bulleted insights and pedagogical suggestions.
        """
        insights = []

        # Attendance insight
        if attendance_rate >= 0.90:
            insights.append(f"Outstanding attendance rate of {attendance_rate:.0%} with high student turnout.")
        elif attendance_rate < 0.70:
            insights.append(f"Low attendance rate ({attendance_rate:.0%}); follow-up recommended for absent students.")

        if late_count > 0:
            insights.append(f"{late_count} student(s) arrived late after the grace period.")

        # Overall engagement insight
        if avg_engagement >= 0.75:
            insights.append(f"High overall engagement level ({avg_engagement:.0%}); teaching style resonated effectively.")
        elif avg_engagement <= 0.45:
            insights.append(f"Engagement was lower than usual ({avg_engagement:.0%}); pacing or difficulty may need adjustment.")

        # Attention fatigue pattern
        if len(fatigue_curve) >= 3:
            first_third = fatigue_curve[0]["average_engagement"]
            last_third = fatigue_curve[-1]["average_engagement"]
            drop = first_third - last_third
            if drop > 0.15:
                insights.append(
                    f"Noticeable attention decline (-{drop:.0%}) detected toward the end of session; consider scheduling an active recall break."
                )

        if suggestion_count > 0:
            insights.append(f"{suggestion_count} real-time teaching intervention recommendations were triggered during class.")

        return insights

    def compare_sessions(self, session_id_1: str, session_id_2: str) -> Dict[str, Any]:
        """Compare performance metrics between two distinct classroom sessions.

        Args:
            session_id_1: Baseline session ID.
            session_id_2: Target session ID.

        Returns:
            Dict[str, Any]: Side-by-side comparison of engagement and attendance deltas.
        """
        s1 = self.analyze_session(session_id_1)
        s2 = self.analyze_session(session_id_2)

        if "error" in s1 or "error" in s2:
            return {"error": "One or both sessions could not be loaded."}

        m1 = s1["summary_metrics"]
        m2 = s2["summary_metrics"]

        return {
            "session_1": {
                "id": session_id_1,
                "title": s1["session_info"].get("title"),
                "attendance_rate": m1["attendance_rate"],
                "avg_engagement": m1["average_engagement"],
            },
            "session_2": {
                "id": session_id_2,
                "title": s2["session_info"].get("title"),
                "attendance_rate": m2["attendance_rate"],
                "avg_engagement": m2["average_engagement"],
            },
            "deltas": {
                "attendance_delta": round(m2["attendance_rate"] - m1["attendance_rate"], 3),
                "engagement_delta": round(m2["average_engagement"] - m1["average_engagement"], 3),
            },
        }

    def generate_report_data(self, session_id: str) -> Dict[str, Any]:
        """Alias to analyze_session for standard report generation pipelines.

        Args:
            session_id: Unique session ID.

        Returns:
            Dict[str, Any]: Complete report dataset.
        """
        return self.analyze_session(session_id)
