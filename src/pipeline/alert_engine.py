from __future__ import annotations

from collections import deque

import numpy as np
from loguru import logger

from src.config import Config
from src.ml.features import extract_features
from src.ml.predictor import Predictor
from src.pipeline.notifier import Alert, ArchivingNotifier, SSENotifier, StdoutNotifier


class AlertEngine:
    """Ingests metric data points one at a time, runs prediction on a rolling
    window, and fires alerts through the notifier when threshold + cooldown
    conditions are met."""

    def __init__(
        self,
        predictor: Predictor,
        cfg: Config,
        notifier: StdoutNotifier | SSENotifier | ArchivingNotifier | None = None,
        debug: bool = False,
    ):
        self.predictor = predictor
        self.cfg = cfg
        self.notifier = notifier or StdoutNotifier()
        self.debug = debug
        self.buffer: deque[float] = deque(maxlen=cfg.W)
        self.steps_since_alert: float = float("inf")
        self.alerts_fired: list[Alert] = []
        self.max_prob_seen: float = 0.0

    def reset(self) -> None:
        self.buffer.clear()
        self.steps_since_alert = float("inf")
        self.alerts_fired = []
        self.max_prob_seen = 0.0

    def ingest(self, timestamp: str, value: float, source: str) -> Alert | None:
        """Process a single data point. Returns an Alert if one was fired."""
        self.buffer.append(value)
        self.steps_since_alert += 1

        if len(self.buffer) < self.cfg.W:
            return None

        features = extract_features(np.array(self.buffer))
        prob, should_alert = self.predictor.predict(features)
        if self.debug:
            self.max_prob_seen = max(self.max_prob_seen, prob)
            if prob > 0.15:
                logger.debug("step prob={:.3f} (threshold={:.2f}) ts={}", prob, self.predictor.threshold, timestamp)

        if not should_alert:
            return None

        if self.steps_since_alert <= self.cfg.cooldown_steps:
            return None

        severity = "critical" if prob >= self.cfg.critical_threshold else "warning"
        alert = Alert(
            timestamp=timestamp,
            source=source,
            probability=round(prob, 4),
            severity=severity,
            message=f"Predicted incident within {self.cfg.H * 5} minutes",
        )

        self.steps_since_alert = 0
        self.alerts_fired.append(alert)
        self.notifier.emit(alert)
        logger.info(
            "[{}] {} alert for {} (p={:.3f})",
            timestamp, severity.upper(), source, prob,
        )
        return alert

    def run_on_series(self, timestamps, values, source: str) -> list[Alert]:
        """Stream an entire series through the engine. Used for backtesting."""
        self.reset()
        for ts, val in zip(timestamps, values):
            self.ingest(str(ts), float(val), source)
        return list(self.alerts_fired)
