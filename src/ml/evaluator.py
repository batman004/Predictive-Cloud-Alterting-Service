from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import Config
from src.ml.features import FEATURE_NAMES
from src.ml.predictor import Predictor


def point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Precision, recall, FPR, AUC-ROC, AUC-PR at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def incident_metrics(
    test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float, cfg: Config,
) -> dict:
    """Incident-level recall: fraction of real incidents that got at least one
    alert before onset.  Works with binary 0/1 labels (label=-1 rows are
    excluded during windowing, so incidents appear as contiguous label=1 groups)."""
    y_pred = (y_prob >= threshold).astype(int)

    sources = test_df["source"].unique()
    total_incidents = 0
    detected_incidents = 0
    lead_times: list[int] = []

    for src in sources:
        mask = test_df["source"] == src
        labels = test_df.loc[mask, "label"].values
        preds = y_pred[mask.values]

        in_warning = False
        warning_start = -1

        for i, lab in enumerate(labels):
            if lab == 1 and not in_warning:
                in_warning = True
                warning_start = i
            elif lab != 1 and in_warning:
                total_incidents += 1
                zone_preds = preds[warning_start:i]
                if zone_preds.any():
                    detected_incidents += 1
                    first_hit = int(np.argmax(zone_preds))
                    lead_times.append(i - (warning_start + first_hit))
                in_warning = False

        if in_warning:
            total_incidents += 1
            zone_preds = preds[warning_start:]
            if zone_preds.any():
                detected_incidents += 1
                first_hit = int(np.argmax(zone_preds))
                lead_times.append(len(labels) - (warning_start + first_hit))

    incident_recall = detected_incidents / total_incidents if total_incidents > 0 else 0.0
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "total_incidents": total_incidents,
        "detected_incidents": detected_incidents,
        "incident_recall": incident_recall,
        "avg_lead_time_steps": avg_lead,
    }


def evaluate(
    test_csv: Path,
    model_path: Path,
    threshold_path: Path,
    report_dir: Path,
    cfg: Config,
) -> dict:
    """Run full evaluation on test set and write report."""
    report_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(model_path, threshold_path)
    test_df = pd.read_csv(test_csv)

    feature_cols = [c for c in FEATURE_NAMES if c in test_df.columns]
    X_test = test_df[feature_cols].values
    y_true = (test_df["label"] == 1).astype(int).values
    y_prob = predictor.model.predict_proba(X_test)[:, 1]

    pm = point_metrics(y_true, y_prob, predictor.threshold)
    im = incident_metrics(test_df, y_prob, predictor.threshold, cfg)

    results = {**pm, **im}

    lines = [
        "=== Evaluation Report ===",
        f"Threshold:          {pm['threshold']:.2f}",
        f"Precision:          {pm['precision']:.4f}",
        f"Recall:             {pm['recall']:.4f}",
        f"FPR:                {pm['fpr']:.4f}",
        f"AUC-ROC:            {pm['auc_roc']:.4f}",
        f"AUC-PR:             {pm['auc_pr']:.4f}",
        f"TP={pm['tp']}  FP={pm['fp']}  FN={pm['fn']}  TN={pm['tn']}",
        "",
        "--- Incident-Level ---",
        f"Total incidents:    {im['total_incidents']}",
        f"Detected:           {im['detected_incidents']}",
        f"Incident recall:    {im['incident_recall']:.2%}",
        f"Avg lead time:      {im['avg_lead_time_steps']:.1f} steps",
    ]
    report_text = "\n".join(lines) + "\n"

    report_path = report_dir / "evaluation.txt"
    report_path.write_text(report_text)
    logger.info("Report saved to {}", report_path)

    for line in lines:
        logger.info(line)

    return results
