from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime


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
