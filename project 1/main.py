"""Command-line interface (CLI) for Multimodal Attendance & Engagement Intelligence.

This entry point provides commands to run live classroom monitoring sessions,
launch the Streamlit dashboard, execute standalone offline demos, list historical sessions,
and generate post-session reports.
"""

from datetime import datetime
import os
import subprocess
import sys
import time
from typing import Optional

import click
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.analytics_agent import AnalyticsAgent
from config import get_config
from core.session_manager import SessionManager
from ui.report_generator import ReportGenerator
from utils.data_store import DataStore
from utils.logger import get_logger

logger = get_logger("CLI")
console = Console()


@click.group()
def cli() -> None:
    """Multimodal Attendance & Engagement Intelligence System."""
    pass


@cli.command("start")
@click.option("--subject", "-s", default="Computer Science", help="Course subject name.")
@click.option("--instructor", "-i", default="Instructor", help="Instructor name.")
@click.option("--camera", "-c", default=0, type=int, help="Webcam device index.")
@click.option("--launch-dashboard", "-d", is_flag=True, default=False, help="Launch Streamlit dashboard.")
def start(subject: str, instructor: str, camera: int, launch_dashboard: bool) -> None:
    """Start an interactive live classroom monitoring session."""
    cfg = get_config()
    cfg.ensure_directories()

    console.print(
        Panel.fit(
            f"[bold blue]Starting Classroom Monitoring Session[/bold blue]\n"
            f"[white]Subject:[/white] {subject}\n"
            f"[white]Instructor:[/white] {instructor}\n"
            f"[white]Camera Index:[/white] {camera}",
            border_style="blue",
        )
    )

    if launch_dashboard:
        dashboard_script = os.path.join(os.path.dirname(__file__), "ui", "dashboard.py")
        console.print("[green]Launching Streamlit Dashboard in browser...[/green]")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", dashboard_script])

    # Initialize live session manager
    sm = SessionManager(subject=subject, instructor=instructor)
    sm.start()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        console.print(f"[bold red]Error: Could not open camera {camera}. Switching to synthetic demo mode...[/bold red]")
        from run_demo import run_synthetic_session
        run_synthetic_session()
        return

    console.print("[cyan]Monitoring active. Press Ctrl+C or 'q' in video window to end session.[/cyan]")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = sm.process_frame(frame)
            current_engagement = result["engagement"]["class_engagement"]
            attendee_count = len(result["current_attendees"])

            # Annotate frame for display
            from utils.face_utils import draw_face_annotations
            annotated = draw_face_annotations(frame, result.get("detections", []))
            cv2.putText(
                annotated,
                f"Engagement: {current_engagement:.0%} | Students: {attendee_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Multimodal Attendance & Engagement Monitor", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]KeyboardInterrupt received. Concluding session cleanly...[/yellow]")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = sm.end()

        # Generate report automatically
        rg = ReportGenerator(sm.data_store)
        report_path = rg.save_report(rg.generate_html_report(sm.session_id))
        console.print(
            Panel.fit(
                f"[bold green]Session Completed Successfully[/bold green]\n"
                f"[white]Session ID:[/white] {sm.session_id}\n"
                f"[white]Total Frames:[/white] {summary['total_frames_processed']}\n"
                f"[white]Students Present:[/white] {summary['active_present_attendees']}\n"
                f"[white]Report Saved:[/white] {report_path}",
                border_style="green",
            )
        )


@cli.command("demo")
@click.option("--duration", "-d", default=30, type=int, help="Number of simulated ticks to run.")
def demo(duration: int) -> None:
    """Run an offline demonstration using synthetic participant data (no hardware required)."""
    from run_demo import run_synthetic_session
    run_synthetic_session(ticks=duration)


@cli.command("report")
@click.option("--session-id", "-s", required=True, help="Session ID to compile report for.")
@click.option("--output", "-o", default=None, help="Custom output HTML or PDF file path.")
def report(session_id: str, output: Optional[str]) -> None:
    """Generate an analytical post-session HTML/PDF report."""
    rg = ReportGenerator()
    html_content = rg.generate_html_report(session_id)
    if "Error Generating Report" in html_content:
        console.print(f"[bold red]Session '{session_id}' not found in database.[/bold red]")
        return

    saved_path = rg.save_report(html_content, output_path=output)
    console.print(f"[bold green]Report successfully generated and saved to:[/bold green] {saved_path}")


@cli.command("list-sessions")
def list_sessions() -> None:
    """List all recorded classroom sessions in the SQLite database."""
    ds = DataStore()
    sessions = ds.list_sessions()

    if not sessions:
        console.print("[yellow]No classroom sessions recorded yet. Run `python main.py demo` or `start`.[/yellow]")
        return

    table = Table(title="Recorded Classroom Sessions", header_style="bold blue")
    table.add_column("Session ID", style="cyan")
    table.add_column("Subject", style="white")
    table.add_column("Instructor", style="green")
    table.add_column("Start Time", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Attendees", justify="right")

    for s in sessions:
        start_t = s.get("start_time", "")[:19].replace("T", " ")
        table.add_row(
            s.get("id"),
            s.get("subject"),
            s.get("instructor"),
            start_t,
            s.get("status"),
            str(s.get("attendee_count", 0)),
        )

    console.print(table)


if __name__ == "__main__":
    cli()
