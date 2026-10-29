"""Post-session automated reporting engine producing standalone HTML and PDF analytics.

This module generates professional pedagogical reports complete with embedded base64
visual charts, tabular attendance rosters, engagement timeline progressions, and actionable AI recommendations.
"""

import base64
from datetime import datetime
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from agents.analytics_agent import AnalyticsAgent
from config import get_config
from utils.data_store import DataStore
from utils.logger import get_logger

logger = get_logger("ReportGenerator")


class ReportGenerator:
    """Generates rich HTML and PDF reports summarizing classroom analytics."""

    def __init__(self, data_store: Optional[DataStore] = None) -> None:
        """Initialize ReportGenerator.

        Args:
            data_store: Optional DataStore instance.
        """
        self.data_store = data_store or DataStore()
        self.analytics_agent = AnalyticsAgent(self.data_store)

    def _generate_engagement_plot_base64(self, timeline: list) -> str:
        """Create a Matplotlib engagement trend chart and encode it as a Base64 image string.

        Args:
            timeline: Chronological engagement records.

        Returns:
            str: Base64 data URI string for embedding directly in HTML <img> tags.
        """
        if not timeline:
            return ""

        try:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
            timestamps = [datetime.fromisoformat(r["timestamp"]) for r in timeline]
            start_t = timestamps[0]
            minutes = [(t - start_t).total_seconds() / 60.0 for t in timestamps]
            scores = [float(r["engagement_score"]) * 100.0 for r in timeline]

            # Line plot
            ax.plot(minutes, scores, color="#2563eb", linewidth=2.5, label="Class Engagement (%)")
            ax.fill_between(minutes, scores, color="#93c5fd", alpha=0.35)

            # Target threshold lines
            ax.axhline(80.0, color="#16a34a", linestyle="--", alpha=0.7, label="High Focus (80%)")
            ax.axhline(40.0, color="#dc2626", linestyle="--", alpha=0.7, label="Attention Alert (40%)")

            ax.set_title("Classroom Engagement Progression Over Time", fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Elapsed Time (Minutes)", fontsize=10)
            ax.set_ylabel("Engagement Level (%)", fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="lower left", framealpha=0.9)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception as err:
            logger.error(f"Error generating engagement plot: {err}")
            return ""

    def generate_html_report(self, session_id: str) -> str:
        """Compile a complete standalone HTML pedagogical report.

        Args:
            session_id: Classroom session identifier.

        Returns:
            str: Self-contained HTML report string with embedded styling and charts.
        """
        analytics = self.analytics_agent.analyze_session(session_id)
        if "error" in analytics:
            return f"<html><body><h1>Error Generating Report</h1><p>{analytics['error']}</p></body></html>"

        session_info = analytics["session_info"]
        summary = analytics["summary_metrics"]
        insights = analytics["key_insights"]
        attendance = analytics["attendance_records"]
        suggestions = analytics["suggestions"]

        raw_timeline = self.data_store.get_engagement_history(session_id, participant_id="CLASS_AVERAGE")
        chart_base64 = self._generate_engagement_plot_base64(raw_timeline)

        # Build attendance rows
        attendance_rows_html = ""
        for att in attendance:
            status = att.get("status", "absent").lower()
            badge_class = "badge-green" if status == "present" else ("badge-yellow" if status == "late" else "badge-red")
            first_seen_str = att.get("first_seen", "")[:19].replace("T", " ") if att.get("first_seen") else "-"
            last_seen_str = att.get("last_seen", "")[:19].replace("T", " ") if att.get("last_seen") else "-"
            attendance_rows_html += f"""
            <tr>
                <td><strong>{att.get('participant_id')}</strong></td>
                <td>{att.get('name', 'Student')}</td>
                <td><span class="badge {badge_class}">{status.upper()}</span></td>
                <td>{first_seen_str}</td>
                <td>{last_seen_str}</td>
                <td>{att.get('total_frames', 0)}</td>
                <td>{att.get('confidence', 1.0):.1%}</td>
            </tr>
            """

        # Build suggestion cards
        suggestions_html = ""
        if suggestions:
            for s in suggestions:
                prio = s.get("priority", "MEDIUM")
                prio_class = "prio-urgent" if prio == "URGENT" else ("prio-high" if prio == "HIGH" else "prio-norm")
                ts_str = s.get("timestamp", "")[:19].replace("T", " ")
                suggestions_html += f"""
                <div class="card suggestion-card {prio_class}">
                    <div class="sugg-header">
                        <span class="badge">{s.get('category')}</span>
                        <span class="priority-label">{prio}</span>
                        <span class="timestamp">{ts_str}</span>
                    </div>
                    <p class="sugg-message">{s.get('message')}</p>
                    <p class="sugg-reason"><em>Trigger Context: {s.get('trigger_reason')}</em></p>
                </div>
                """
        else:
            suggestions_html = "<p class='text-muted'>No urgent teaching interventions were triggered during this session.</p>"

        # Build insights list
        insights_html = "".join([f"<li>{item}</li>" for item in insights])

        # Render complete template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classroom Analytics Report - {session_info.get('title')}</title>
    <style>
        :root {{
            --primary: #1e40af;
            --bg: #f8fafc;
            --surface: #ffffff;
            --text: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--surface);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border);
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: 700;
            color: var(--primary);
            margin: 6px 0;
        }}
        .metric-card .label {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            margin-top: 0;
            font-size: 18px;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 12px;
            color: #1e293b;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-green {{ background-color: #dcfce7; color: #166534; }}
        .badge-yellow {{ background-color: #fef9c3; color: #854d0e; }}
        .badge-red {{ background-color: #fee2e2; color: #991b1b; }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .suggestion-card {{
            background: #f8fafc;
            border-left: 5px solid #3b82f6;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 12px;
        }}
        .prio-urgent {{ border-left-color: #ef4444; background: #fff5f5; }}
        .prio-high {{ border-left-color: #f97316; background: #fffaf0; }}
        .sugg-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .sugg-message {{ font-size: 15px; font-weight: 500; margin: 4px 0; }}
        .sugg-reason {{ font-size: 13px; color: var(--text-muted); margin: 0; }}
        ul.insights-list {{
            padding-left: 20px;
            font-size: 14px;
        }}
        ul.insights-list li {{ margin-bottom: 8px; }}
        footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Classroom Analytics Report</h1>
            <p><strong>Subject:</strong> {session_info.get('subject')} | <strong>Instructor:</strong> {session_info.get('instructor')} | <strong>Session ID:</strong> {session_id}</p>
            <p><strong>Conducted:</strong> {session_info.get('start_time', '')[:19].replace('T', ' ')}</p>
        </header>

        <!-- KPI Metrics Grid -->
        <div class="grid">
            <div class="metric-card">
                <div class="label">Attendance Rate</div>
                <div class="value">{summary.get('attendance_rate', 0.0):.0%}</div>
                <div class="label">{summary.get('present_count', 0)} of {summary.get('total_students', 0)} Present</div>
            </div>
            <div class="metric-card">
                <div class="label">Average Engagement</div>
                <div class="value">{summary.get('average_engagement', 0.0):.0%}</div>
                <div class="label">Peak: {summary.get('peak_engagement_score', 0.0):.0%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Late Arrivals</div>
                <div class="value">{summary.get('late_count', 0)}</div>
                <div class="label">Students Delayed</div>
            </div>
            <div class="metric-card">
                <div class="label">AI Interventions</div>
                <div class="value">{summary.get('total_suggestions_given', 0)}</div>
                <div class="label">Guidance Events</div>
            </div>
        </div>

        <!-- Key Pedagogical Insights -->
        <div class="section">
            <h2>Executive Insights & Key Takeaways</h2>
            <ul class="insights-list">
                {insights_html}
            </ul>
        </div>

        <!-- Engagement Chart -->
        <div class="section">
            <h2>Session Engagement Timeline</h2>
            <div class="chart-container">
                {"<img src='" + chart_base64 + "' alt='Engagement Timeline Chart' />" if chart_base64 else "<p>No timeline data available for graphing.</p>"}
            </div>
        </div>

        <!-- Attendance Roster -->
        <div class="section">
            <h2>Verified Attendance Roster</h2>
            <table>
                <thead>
                    <tr>
                        <th>Participant ID</th>
                        <th>Name</th>
                        <th>Status</th>
                        <th>First Seen</th>
                        <th>Last Seen</th>
                        <th>Total Frames</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {attendance_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Teaching Suggestions Log -->
        <div class="section">
            <h2>AI Pedagogical Recommendations Log</h2>
            {suggestions_html}
        </div>

        <footer>
            <p>Multimodal Classroom Intelligence System • Privacy-Preserving Computer Vision & Acoustic Analysis</p>
        </footer>
    </div>
</body>
</html>
"""
        return html_content

    def save_report(self, html_content: str, output_path: Optional[str] = None) -> str:
        """Save HTML report string to disk.

        Args:
            html_content: Generated HTML string.
            output_path: Optional destination path.

        Returns:
            str: Resolved absolute file path of written report.
        """
        cfg = get_config()
        reports_dir = Path(cfg.storage.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        if not output_path:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            target_path = reports_dir / filename
        else:
            target_path = Path(output_path)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Report successfully saved to {target_path}")
        return str(target_path.resolve())

    def generate_pdf_report(self, session_id: str, output_path: Optional[str] = None) -> str:
        """Generate PDF report for session.

        Uses WeasyPrint or pdfkit if installed; otherwise saves pristine printable HTML report.

        Args:
            session_id: Classroom session ID.
            output_path: Optional target PDF path.

        Returns:
            str: File path of saved PDF or HTML printable report.
        """
        html_str = self.generate_html_report(session_id)
        cfg = get_config()
        reports_dir = Path(cfg.storage.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = output_path or str(reports_dir / f"session_report_{session_id}.pdf")

        # Attempt WeasyPrint conversion
        try:
            from weasyprint import HTML  # type: ignore
            HTML(string=html_str).write_pdf(pdf_path)
            logger.info(f"Generated PDF report at {pdf_path}")
            return pdf_path
        except Exception as err:
            logger.debug(f"WeasyPrint PDF engine unavailable ({err}). Saving printable HTML report instead.")
            fallback_html_path = pdf_path.replace(".pdf", ".html")
            return self.save_report(html_str, fallback_html_path)
