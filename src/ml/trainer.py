from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import Config
from src.ml.features import FEATURE_NAMES


def _find_best_threshold(
    model: XGBClassifier, X_val: np.ndarray, y_val: np.ndarray,
) -> float:
    """Sweep probability thresholds on the validation set to maximise F1."""
    probas = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probas >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), score
    logger.info("Best threshold={:.2f}  val-F1={:.4f}", best_t, best_f1)
    return best_t


def train(
    train_csv: Path,
    val_csv: Path,
    out_dir: Path,
    cfg: Config,
) -> Path:
    """Train XGBoost on prepared data, tune threshold, save artifacts.

    Returns the path to the saved model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    feature_cols = [c for c in FEATURE_NAMES if c in train_df.columns]
    X_train = train_df[feature_cols].values
    y_train = (train_df["label"] == 1).astype(int).values
    X_val = val_df[feature_cols].values
    y_val = (val_df["label"] == 1).astype(int).values

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos if pos > 0 else 1.0
    logger.info("Training samples={} (pos={}, neg={}, scale_pos_weight={:.1f})", len(y_train), pos, neg, spw)

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    threshold = _find_best_threshold(model, X_val, y_val)

    model_path = out_dir / "model.joblib"
    threshold_path = out_dir / "threshold.json"
    joblib.dump(model, model_path)
    threshold_path.write_text(json.dumps({"threshold": threshold}))

    logger.info("Model saved to {}", model_path)
    logger.info("Threshold saved to {}", threshold_path)
    return model_path
