"""Standalone offline simulation and demonstration of Multimodal Attendance & Engagement Agent.

This script executes a full synthetic classroom session without requiring hardware camera
or microphone access. It models realistic student behavior patterns, updates attendance,
computes multimodal engagement metrics, triggers AI teaching interventions, generates
an HTML analytical report, and displays rich formatted console telemetry.
"""

from datetime import datetime, timedelta
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from core.session_manager import SessionManager
from ui.report_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger("RunDemo")
console = Console()


def create_synthetic_frame(
    width: int,
    height: int,
    active_students: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Render a synthetic classroom frame containing participant avatars and features.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        active_students: List of student configurations currently in view.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int, int, int]]]: (synthesized_frame, face_bounding_boxes).
    """
    frame = np.full((height, width, 3), (45, 40, 38), dtype=np.uint8)

    # Draw classroom background decor (e.g. whiteboard boundary and desk line)
    cv2.rectangle(frame, (30, 20), (width - 30, 100), (70, 65, 60), 2)
    cv2.putText(
        frame,
        "CS 204: Advanced Artificial Intelligence & Agents",
        (50, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )
    cv2.line(frame, (0, 380), (width, 380), (60, 55, 50), 2)

    bboxes: List[Tuple[int, int, int, int]] = []

    for s in active_students:
        bx, by = s["pos_x"], s["pos_y"]
        bw, bh = s["width"], s["height"]
        color = s["color"]
        name = s["name"]

        # Face region rectangle
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

        # Draw eyes and nose to facilitate facial feature extraction
        eye_y = by + int(bh * 0.35)
        left_eye_x = bx + int(bw * 0.30)
        right_eye_x = bx + int(bw * 0.70)
        eye_radius = max(3, int(bw * 0.08))

        # Gaze shift based on whether looking away
        gaze_offset = 0 if s.get("is_frontal", True) else int(bw * 0.15)
        cv2.circle(frame, (left_eye_x + gaze_offset, eye_y), eye_radius, (30, 30, 30), -1)
        cv2.circle(frame, (right_eye_x + gaze_offset, eye_y), eye_radius, (30, 30, 30), -1)

        # Mouth
        mouth_y = by + int(bh * 0.75)
        mouth_w = int(bw * 0.4)
        cv2.line(frame, (bx + int(bw * 0.3), mouth_y), (bx + int(bw * 0.7), mouth_y), (40, 40, 40), 2)

        # Name label
        cv2.putText(
            frame,
            name,
            (bx, by - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        bboxes.append((bx, by, bw, bh))

    return frame, bboxes


def run_synthetic_session(ticks: int = 40) -> Dict[str, Any]:
    """Execute a realistic synthetic session simulation over a number of ticks.

    Args:
        ticks: Number of discrete simulation time steps to execute.

    Returns:
        Dict[str, Any]: Final session analytics summary.
    """
    cfg = get_config()
    cfg.ensure_directories()

    console.print(
        Panel.fit(
            "[bold cyan]Multimodal Attendance & Engagement Intelligence[/bold cyan]\n"
            "[white]Running Autonomous Offline Simulation Demo (No Camera/Microphone Required)[/white]",
            border_style="cyan",
        )
    )

    # 1. Define 5 simulated participants with distinct behavioural archetypes
    participants = [
        {
            "id": "p_alice",
            "name": "Alice Johnson",
            "role": "High Engagement Student",
            "color": (210, 180, 140),  # BGR
            "base_x": 60,
            "base_y": 140,
            "width": 80,
            "height": 95,
            "join_tick": 0,
        },
        {
            "id": "p_bob",
            "name": "Bob Smith",
            "role": "Moderate Engagement Student",
            "color": (160, 200, 170),
            "base_x": 180,
            "base_y": 150,
            "width": 80,
            "height": 95,
            "join_tick": 0,
        },
        {
            "id": "p_charlie",
            "name": "Charlie Davis",
            "role": "Attention Fatigue Student",
            "color": (180, 160, 220),
            "base_x": 300,
            "base_y": 145,
            "width": 80,
            "height": 95,
            "join_tick": 0,
        },
        {
            "id": "p_diana",
            "name": "Diana Miller",
            "role": "Distracted / Restless Student",
            "color": (200, 220, 150),
            "base_x": 420,
            "base_y": 155,
            "width": 80,
            "height": 95,
            "join_tick": 0,
        },
        {
            "id": "p_ethan",
            "name": "Ethan Hunt",
            "role": "Late Arrival Student",
            "color": (150, 190, 230),
            "base_x": 530,
            "base_y": 140,
            "width": 80,
            "height": 95,
            "join_tick": 12,  # Joins at tick 12
        },
    ]

    # Pre-register known students in DataStore
    ds = DataStore()
    for p in participants:
        # Generate stable dummy embedding for known matching
        np.random.seed(hash(p["id"]) % 100000)
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        ds.register_attendee(p["id"], p["name"], embedding=emb.tolist())

    # 2. Instantiate SessionManager
    session_id = f"demo_sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sm = SessionManager(
        session_id=session_id,
        subject="CS 204: Intelligent Systems",
        instructor="Dr. Sarah Connor",
        data_store=ds,
        mock_audio=True,
    )
    sm.start(metadata={"environment": "synthetic_simulation", "ticks": ticks})

    # Event tracking
    generated_suggestions: List[Dict[str, Any]] = []

    def on_suggestion(sugg: Dict[str, Any]) -> None:
        generated_suggestions.append(sugg)

    sm.register_event_listener("on_suggestion_generated", on_suggestion)

    sim_start = datetime.now()

    # 3. Step through simulated session ticks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Simulating classroom progression...", total=ticks)

        for tick in range(ticks):
            # Simulated timestamp advancing 30 seconds per tick (totaling 20-30 minutes)
            curr_sim_time = sim_start + timedelta(seconds=tick * 30)

            # Determine participant status and motion dynamics
            active_render_students: List[Dict[str, Any]] = []

            for p in participants:
                if tick < p["join_tick"]:
                    continue  # Student hasn't joined yet

                pid = p["id"]
                pos_x = p["base_x"]
                pos_y = p["base_y"]
                is_frontal = True

                # Dynamic behavior simulation per archetype:
                if pid == "p_alice":
                    # Alice: steady, slight gentle nodding
                    pos_y += int(2 * math.sin(tick * 0.5))
                    is_frontal = True
                elif pid == "p_bob":
                    # Bob: takes notes every 5 ticks (downward pitch / slight shift)
                    if tick % 6 in (2, 3):
                        pos_y += 6
                        is_frontal = False
                    else:
                        is_frontal = True
                elif pid == "p_charlie":
                    # Charlie: fatigue drops attention after tick 15
                    if tick > 15:
                        pos_x += 15
                        is_frontal = False
                    else:
                        is_frontal = True
                elif pid == "p_diana":
                    # Diana: frequently moves and looks around
                    pos_x += int(8 * math.sin(tick * 1.2))
                    pos_y += int(6 * math.cos(tick * 1.0))
                    is_frontal = (tick % 3 == 0)
                elif pid == "p_ethan":
                    # Ethan: joins late, settles in
                    pos_x += int(3 * math.sin(tick * 0.3))
                    is_frontal = True

                active_render_students.append({
                    "id": pid,
                    "name": p["name"],
                    "color": p["color"],
                    "pos_x": max(10, min(pos_x, 550)),
                    "pos_y": max(100, min(pos_y, 350)),
                    "width": p["width"],
                    "height": p["height"],
                    "is_frontal": is_frontal,
                })

            # Render synthetic frame and obtain bounding boxes
            frame, bboxes = create_synthetic_frame(640, 480, active_render_students)

            # Simulate occasional vocal interaction (every ~8 ticks)
            if tick in (8, 18, 28):
                # Simulated speech audio waveform chunk
                t_arr = np.linspace(0, 0.1, 1600, endpoint=False)
                speech_chunk = (0.05 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
                sm.process_audio_chunk(speech_chunk)

            # Ingest frame into SessionManager
            step_result = sm.process_frame(
                frame, timestamp=curr_sim_time, detections_override=bboxes
            )

            progress.update(task, advance=1)
            time.sleep(0.04)  # Fast-forward simulation speed

    # 4. Finalize session
    summary = sm.end()

    # 5. Generate Standalone HTML Report
    rg = ReportGenerator(ds)
    html_report = rg.generate_html_report(session_id)
    saved_report_path = rg.save_report(html_report)

    # 6. Print Rich Formatted Console Summary
    _print_rich_session_summary(session_id, ds, summary, saved_report_path)

    return summary


def _print_rich_session_summary(
    session_id: str,
    data_store: DataStore,
    summary: Dict[str, Any],
    report_path: str,
) -> None:
    """Display comprehensive post-demo analytics using Rich formatting."""
    from agents.analytics_agent import AnalyticsAgent
    analytics = AnalyticsAgent(data_store).analyze_session(session_id)

    # Summary Panel
    metrics = analytics.get("summary_metrics", {})
    console.print()
    console.print(
        Panel(
            f"[bold green]Classroom Intelligence Session Completed[/bold green]\n"
            f"[white]Session ID:[/white] [cyan]{session_id}[/cyan]\n"
            f"[white]Duration Tracked:[/white] {summary.get('duration_seconds', 0):.1f}s (simulated ~20 mins)\n"
            f"[white]Attendance Rate:[/white] [bold blue]{metrics.get('attendance_rate', 0.0):.0%}[/bold blue] "
            f"({metrics.get('present_count', 0)} of {metrics.get('total_students', 0)} students present)\n"
            f"[white]Average Class Engagement:[/white] [bold magenta]{metrics.get('average_engagement', 0.0):.0%}[/bold magenta] "
            f"(Peak: {metrics.get('peak_engagement_score', 0.0):.0%})\n"
            f"[white]AI Suggestions Issued:[/white] {metrics.get('total_suggestions_given', 0)}\n"
            f"[white]HTML Report Generated:[/white] [underline cyan]{report_path}[/underline cyan]",
            title="[bold yellow]Executive Summary[/bold yellow]",
            border_style="green",
        )
    )

    # Attendance Roster Table
    att_table = Table(title="Verified Student Attendance Roster", header_style="bold blue")
    att_table.add_column("Participant ID", style="cyan")
    att_table.add_column("Student Name", style="white")
    att_table.add_column("Status", style="bold")
    att_table.add_column("First Seen", style="magenta")
    att_table.add_column("Last Seen", style="magenta")
    att_table.add_column("Confidence", justify="right")

    for att in analytics.get("attendance_records", []):
        st = att.get("status", "absent").upper()
        st_style = "[green]PRESENT[/green]" if st == "PRESENT" else ("[yellow]LATE[/yellow]" if st == "LATE" else "[red]ABSENT[/red]")
        first_s = att.get("first_seen", "")[:19].replace("T", " ") if att.get("first_seen") else "-"
        last_s = att.get("last_seen", "")[:19].replace("T", " ") if att.get("last_seen") else "-"
        att_table.add_row(
            att.get("participant_id"),
            att.get("name"),
            st_style,
            first_s,
            last_s,
            f"{att.get('confidence', 1.0):.1%}",
        )
    console.print(att_table)

    # Teaching Interventions Table
    sugg_list = analytics.get("suggestions", [])
    if sugg_list:
        sugg_table = Table(title="AI Teaching Recommendations Triggered", header_style="bold magenta")
        sugg_table.add_column("Category", style="cyan")
        sugg_table.add_column("Priority", style="bold")
        sugg_table.add_column("Recommendation Guidance", style="white")
        sugg_table.add_column("Trigger Context", style="dim")

        for s in sugg_list:
            prio = s.get("priority", "MEDIUM")
            prio_style = "[red]URGENT[/red]" if prio == "URGENT" else ("[yellow]HIGH[/yellow]" if prio == "HIGH" else "[blue]INFO[/blue]")
            sugg_table.add_row(
                s.get("category"),
                prio_style,
                s.get("message"),
                s.get("trigger_reason"),
            )
        console.print(sugg_table)

    # Key Insights List
    insights = analytics.get("key_insights", [])
    if insights:
        console.print("\n[bold yellow]Key Insights & Longitudinal Observations:[/bold yellow]")
        for idx, ins in enumerate(insights, 1):
            console.print(f"  [cyan]{idx}.[/cyan] {ins}")
    console.print()


if __name__ == "__main__":
    run_synthetic_session()
