"""Interactive real-time Streamlit dashboard for multimodal classroom telemetry.

This module renders a live responsive two-column dashboard with webcam feed,
presence roster, real-time engagement gauge, Plotly timeline charts, audio indicators,
and color-coded AI teaching suggestions.
"""

from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import get_config
from core.session_manager import SessionManager
from utils.data_store import DataStore
from utils.face_utils import draw_face_annotations


class StreamlitDashboard:
    """Renders modular UI components for classroom attendance, engagement, and AI interventions."""

    def __init__(self, data_store: Optional[DataStore] = None) -> None:
        """Initialize the Streamlit dashboard controller.

        Args:
            data_store: Optional DataStore instance.
        """
        self.data_store = data_store or DataStore()

    def render_live_view(self, container: Any, frame: np.ndarray, participant_data: List[Dict[str, Any]]) -> None:
        """Render annotated video frame inside specified Streamlit container.

        Args:
            container: Streamlit element/column container.
            frame: Video image frame in BGR format.
            participant_data: List of detected participant metadata dictionaries.
        """
        if frame is not None and frame.size > 0:
            annotated = draw_face_annotations(frame, participant_data)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            container.image(rgb, caption="Real-time Camera Ingestion", use_container_width=True)
        else:
            container.info("Camera feed standby / no video input.")

    def render_attendance_panel(self, container: Any, attendees: List[Dict[str, Any]]) -> None:
        """Render current active student roster and attendance statuses.

        Args:
            container: Streamlit element/column container.
            attendees: List of active attendee records.
        """
        container.subheader("Verified Attendance Roster")
        if not attendees:
            container.write("No students currently detected.")
            return

        df = pd.DataFrame(attendees)
        cols_to_show = ["participant_id", "name", "status", "confidence"]
        available_cols = [c for c in cols_to_show if c in df.columns]
        display_df = df[available_cols].copy()
        if "confidence" in display_df.columns:
            display_df["confidence"] = display_df["confidence"].apply(lambda v: f"{float(v):.1%}")

        container.dataframe(display_df, use_container_width=True, hide_index=True)

    def render_engagement_gauge(self, container: Any, score: float) -> None:
        """Render Plotly radial gauge showing class-wide engagement level (0-100%).

        Args:
            container: Streamlit element/column container.
            score: Floating-point engagement score between 0.0 and 1.0.
        """
        pct = max(0.0, min(100.0, score * 100.0))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                title={"text": "Classroom Engagement (%)", "font": {"size": 18}},
                number={"suffix": "%", "font": {"size": 28, "color": "#1e3a8a"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#2563eb"},
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 75], "color": "#fef9c3"},
                        {"range": [75, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 4},
                        "thickness": 0.75,
                        "value": 40,
                    },
                },
            )
        )
        fig.update_layout(height=230, margin=dict(l=15, r=15, t=30, b=15))
        container.plotly_chart(fig, use_container_width=True)

    def render_engagement_chart(self, container: Any, history: List[Dict[str, Any]]) -> None:
        """Render Plotly timeline chart displaying class engagement trajectory over time.

        Args:
            container: Streamlit element/column container.
            history: List of timeline dicts with timestamp and engagement_score.
        """
        container.subheader("Engagement Progression Timeline")
        if not history:
            container.info("Accumulating engagement telemetry...")
            return

        df = pd.DataFrame(history)
        if "timestamp" in df.columns and "engagement_score" in df.columns:
            df["time_label"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")
            df["score_pct"] = df["engagement_score"].astype(float) * 100.0

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["time_label"],
                    y=df["score_pct"],
                    mode="lines+markers",
                    name="Class Engagement",
                    line=dict(color="#2563eb", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(37, 99, 235, 0.15)",
                )
            )
            # Reference threshold line at 40%
            fig.add_hline(y=40, line_dash="dash", line_color="#ef4444", annotation_text="Attention Alert (40%)")
            fig.add_hline(y=80, line_dash="dot", line_color="#10b981", annotation_text="Target Focus (80%)")

            fig.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(range=[0, 105], title="Score (%)"),
                xaxis=dict(title="Time"),
            )
            container.plotly_chart(fig, use_container_width=True)

    def render_suggestions(self, container: Any, suggestions: List[Dict[str, Any]]) -> None:
        """Render card-based AI teaching recommendations color-coded by priority.

        Args:
            container: Streamlit element/column container.
            suggestions: List of recommendation dicts.
        """
        container.subheader("AI Teaching Interventions")
        if not suggestions:
            container.success("Classroom flow is optimal. No active intervention required.")
            return

        for s in reversed(suggestions[-5:]):
            prio = s.get("priority", "MEDIUM")
            cat = s.get("category", "GENERAL")
            msg = s.get("message", "")
            ts = s.get("timestamp", "")[:19].replace("T", " ")

            if prio == "URGENT":
                container.error(f"🚨 **[{prio}] {cat}** ({ts})\n\n{msg}")
            elif prio == "HIGH":
                container.warning(f"⚠️ **[{prio}] {cat}** ({ts})\n\n{msg}")
            elif prio == "LOW":
                container.success(f"💡 **[{prio}] {cat}** ({ts})\n\n{msg}")
            else:
                container.info(f"ℹ️ **[{prio}] {cat}** ({ts})\n\n{msg}")

    def render_audio_indicator(self, container: Any, audio_features: Dict[str, Any]) -> None:
        """Render audio activity bar and acoustic stats.

        Args:
            container: Streamlit element/column container.
            audio_features: Acoustic features dict.
        """
        container.subheader("Acoustic Activity")
        rms = float(audio_features.get("average_rms", 0.0))
        is_speech = bool(audio_features.get("is_speech_active", False))
        rate = float(audio_features.get("speaking_rate", 0.0))

        # Normalized meter from 0 to 100
        vol_pct = min(1.0, rms / 0.05)
        container.progress(vol_pct, text=f"Ambient Noise / Vocal Energy: {vol_pct:.0%}")

        c1, c2 = container.columns(2)
        if is_speech:
            c1.markdown("🗣️ **Speech Detected: Active**")
        else:
            c1.markdown("🤫 **Speech: Quiet / Listening**")
        c2.markdown(f"Speaking Cadence: **{rate:.1f}** syl/sec")


def run_streamlit_app() -> None:
    """Main execution function when launched with `streamlit run ui/dashboard.py`."""
    st.set_page_config(page_title="Multimodal Classroom Intelligence", layout="wide", page_icon="🎓")
    st.title("🎓 Multimodal Classroom Intelligence Dashboard")
    st.markdown("Privacy-preserving real-time computer vision, acoustic telemetry, and AI teaching guidance.")

    # Initialize dashboard controller
    dashboard = StreamlitDashboard()

    # Sidebar controls
    st.sidebar.header("Session Controls")
    mode = st.sidebar.radio("Input Source Mode", ["Demo Simulation", "Live Camera Feed"], index=0)
    subject = st.sidebar.text_input("Course Subject", "CS101: Artificial Intelligence")
    instructor = st.sidebar.text_input("Instructor", "Prof. Alan Turing")

    if "session_manager" not in st.session_state:
        sm = SessionManager(subject=subject, instructor=instructor, mock_audio=(mode == "Demo Simulation"))
        sm.start()
        st.session_state["session_manager"] = sm
    sm: SessionManager = st.session_state["session_manager"]

    # Top KPI Metrics row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    summary = sm.get_summary()

    kpi1.metric("Course Subject", sm.subject)
    kpi2.metric("Active Present", summary["active_present_attendees"])
    kpi3.metric("Current Engagement", f"{summary['current_engagement']:.0%}")
    kpi4.metric("Frames Processed", summary["total_frames_processed"])

    # 2-Column Responsive Layout
    col_left, col_right = st.columns([6, 5])

    # Left Column: Video View + Attendance Table
    with col_left:
        video_placeholder = st.empty()
        roster_placeholder = st.empty()

    # Right Column: Engagement Gauge + Timeline + Suggestions + Audio
    with col_right:
        gauge_placeholder = st.empty()
        chart_placeholder = st.empty()
        audio_placeholder = st.empty()
        suggestions_placeholder = st.empty()

    # Simulation / Frame Generation Step
    if mode == "Demo Simulation":
        # Generate synthetic test frame
        synth_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        synth_frame[:] = (35, 30, 30)

        # Draw 3 simulated faces
        cv2.rectangle(synth_frame, (100, 150), (220, 290), (200, 180, 140), -1)
        cv2.rectangle(synth_frame, (260, 130), (380, 270), (180, 200, 160), -1)
        cv2.rectangle(synth_frame, (420, 160), (540, 300), (160, 160, 210), -1)

        result = sm.process_frame(synth_frame)
    else:
        # Camera feed
        cap = cv2.VideoCapture(0)
        ret, cam_frame = cap.read()
        cap.release()
        if ret:
            result = sm.process_frame(cam_frame)
            synth_frame = cam_frame
        else:
            synth_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = sm.process_frame(synth_frame)

    # Render components
    participant_data = result.get("detections", [])
    current_attendees = result.get("current_attendees", [])
    engagement = result.get("engagement", {}).get("class_engagement", 0.75)
    timeline = sm.data_store.get_engagement_history(sm.session_id, participant_id="CLASS_AVERAGE")
    suggestions = sm.data_store.get_session_suggestions(sm.session_id)
    audio = result.get("audio", {})

    dashboard.render_live_view(video_placeholder, synth_frame, participant_data)
    dashboard.render_attendance_panel(roster_placeholder, current_attendees)
    dashboard.render_engagement_gauge(gauge_placeholder, engagement)
    dashboard.render_engagement_chart(chart_placeholder, timeline)
    dashboard.render_audio_indicator(audio_placeholder, audio)
    dashboard.render_suggestions(suggestions_placeholder, suggestions)

    st.sidebar.divider()
    if st.sidebar.button("Conclude & End Session"):
        sm.end()
        st.success(f"Session {sm.session_id} finalized and saved to database.")


if __name__ == "__main__":
    run_streamlit_app()
