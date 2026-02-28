from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.ml.features import FEATURE_NAMES


class Predictor:
    """Stateless wrapper: loads a trained model and returns predictions.
    If feature_stats.json exists next to the model, normalizes input features
    to match training distribution."""

    def __init__(self, model_path: Path, threshold_path: Path):
        self.model = joblib.load(model_path)
        self.threshold = json.loads(threshold_path.read_text())["threshold"]
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        stats_path = model_path.parent / "feature_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            self._mean = np.array(stats["mean"], dtype=np.float64)
            self._std = np.array(stats["std"], dtype=np.float64)
            self._std = np.where(self._std > 1e-9, self._std, 1.0)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            return X
        return (X - self._mean) / self._std

    def predict(self, features: dict[str, float]) -> tuple[float, bool]:
        """Return (probability_of_incident, should_alert)."""
        vec = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float64)
        vec = self._normalize(vec)
        prob = float(self.model.predict_proba(vec)[0, 1])
        return prob, prob >= self.threshold

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Normalize and return P(incident) for each row. Used by evaluator."""
        X = np.asarray(X, dtype=np.float64)
        X = self._normalize(X)
        return self.model.predict_proba(X)[:, 1]
