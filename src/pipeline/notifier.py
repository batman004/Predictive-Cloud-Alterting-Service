from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict


@dataclass
class Alert:
    timestamp: str
    source: str
    probability: float
    severity: str
    message: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class StdoutNotifier:
    """Writes structured JSON alerts to stdout, one per line."""

    def emit(self, alert: Alert) -> None:
        sys.stdout.write(alert.to_json() + "\n")
        sys.stdout.flush()


class SSENotifier:
    """Writes alerts as Server-Sent Events to stdout.

    Format:
        event: alert
        data: {"timestamp": "...", "source": "...", ...}

    Each event is followed by a blank line and flush so clients receive it immediately.
    """

    def emit(self, alert: Alert) -> None:
        sys.stdout.write("event: alert\n")
        sys.stdout.write(f"data: {alert.to_json()}\n\n")
        sys.stdout.flush()


class ArchivingNotifier:
    """Wraps another notifier and appends every alert to a log file (JSON lines)
    for archival. The inner notifier still runs (e.g. stdout or SSE)."""

    def __init__(self, inner: StdoutNotifier | SSENotifier, log_path: str | None):
        self.inner = inner
        self.log_path = log_path

    def emit(self, alert: Alert) -> None:
        self.inner.emit(alert)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(alert.to_json() + "\n")
                f.flush()
