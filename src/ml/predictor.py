from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.ml.features import FEATURE_NAMES


class Predictor:
    """Stateless wrapper: loads a trained model and returns predictions."""

    def __init__(self, model_path: Path, threshold_path: Path):
        self.model = joblib.load(model_path)
        self.threshold = json.loads(threshold_path.read_text())["threshold"]

    def predict(self, features: dict[str, float]) -> tuple[float, bool]:
        """Return (probability_of_incident, should_alert)."""
        vec = np.array([[features[f] for f in FEATURE_NAMES]])
        prob = float(self.model.predict_proba(vec)[0, 1])
        return prob, prob >= self.threshold
