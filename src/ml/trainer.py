from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

from src.config import Config
from src.ml.features import FEATURE_NAMES

TARGET_INCIDENT_RECALL = 0.80
MAX_FPR = 0.15


def _find_best_threshold(
    model: XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
) -> float:
    """Pick threshold that maximises incident recall on val while keeping FPR
    below MAX_FPR.  Falls back to best-F1 if no threshold meets the target."""
    probas = model.predict_proba(X_val)[:, 1]

    best_t, best_recall = 0.5, 0.0
    fallback_t, fallback_f1 = 0.5, 0.0

    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probas >= t).astype(int)

        fp = int(((preds == 1) & (y_val == 0)).sum())
        tn = int(((preds == 0) & (y_val == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        if f1 > fallback_f1:
            fallback_t, fallback_f1 = float(t), f1

        if fpr <= MAX_FPR and rec > best_recall:
            best_t, best_recall = float(t), rec

    if best_recall >= TARGET_INCIDENT_RECALL:
        logger.info("Threshold={:.2f}  val-recall={:.2%}  (target met)", best_t, best_recall)
        return best_t

    if best_recall > 0:
        logger.info("Threshold={:.2f}  val-recall={:.2%}  (best under FPR cap)", best_t, best_recall)
        return best_t

    logger.info("Threshold={:.2f}  val-F1={:.4f}  (fallback)", fallback_t, fallback_f1)
    return fallback_t


def train(
    train_csv: Path,
    val_csv: Path,
    out_dir: Path,
    cfg: Config,
) -> Path:
    """Train XGBoost on SMOTE-balanced data, tune threshold for incident recall,
    and save artifacts.  Returns the path to the saved model."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    feature_cols = [c for c in FEATURE_NAMES if c in train_df.columns]
    X_train = train_df[feature_cols].values
    y_train = (train_df["label"] == 1).astype(int).values
    X_val = val_df[feature_cols].values
    y_val = (val_df["label"] == 1).astype(int).values

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    logger.info("Raw training samples={} (pos={}, neg={})", len(y_train), pos, neg)

    if pos >= 2:
        k = min(5, pos - 1) if pos > 1 else 1
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        new_pos = int((y_train == 1).sum())
        logger.info("After SMOTE: {} samples (pos={}, neg={})", len(y_train), new_pos, len(y_train) - new_pos)
    else:
        logger.warning("Too few positives for SMOTE (pos={}), training without oversampling", pos)

    spw = 1.0

    model = XGBClassifier(
        n_estimators=800,
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

    threshold = _find_best_threshold(model, X_val, y_val, val_df)

    model_path = out_dir / "model.joblib"
    threshold_path = out_dir / "threshold.json"
    joblib.dump(model, model_path)
    threshold_path.write_text(json.dumps({"threshold": threshold}))

    logger.info("Model saved to {}", model_path)
    logger.info("Threshold saved to {}", threshold_path)
    return model_path
