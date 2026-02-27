"""Unified CLI entry point for the predictive cloud alerting pipeline.

Usage:
    python cli.py train   [--artifacts ARTIFACTS_DIR]
    python cli.py evaluate [--artifacts ARTIFACTS_DIR]
    python cli.py predict  --source SOURCE_NAME [--artifacts ARTIFACTS_DIR]
    python cli.py stream   --input METRICS.csv [--source ID]  # SSE output when threshold crossed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.config import Config
from src.ml.features import (
    FEATURE_NAMES,
    build_window_df,
    label_series,
    temporal_split,
)
from src.ml.trainer import train as train_model
from src.ml.evaluator import evaluate as evaluate_model
from src.ml.predictor import Predictor
from src.pipeline.ingest import NABClient, read_metric_stream
from src.pipeline.alert_engine import AlertEngine
from src.pipeline.notifier import ArchivingNotifier, SSENotifier, StdoutNotifier


ARTIFACTS_DEFAULT = Path("artifacts")


# ── train ────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config()
    out = Path(args.artifacts)
    data_dir = out / "data"
    model_dir = out / "models"
    report_dir = out / "reports"

    client = NABClient(cfg, local_path=args.local_nab)
    windows_map = client.load_windows()

    import pandas as pd
    all_train, all_val, all_test = [], [], []

    for filename in cfg.aws_files:
        source_id = Path(filename).stem
        try:
            raw = client.load_series(filename)
        except Exception as exc:
            logger.error("{} load failed: {}", source_id, exc)
            continue

        labeled = label_series(raw, windows_map.get(filename, []), cfg.H)
        windowed = build_window_df(labeled, source_id, cfg)

        if len(windowed) < 10:
            logger.warning("{} skipped (too few windows)", source_id)
            continue

        tr, va, te = temporal_split(windowed, cfg)
        if min(len(tr), len(va), len(te)) == 0:
            logger.warning("{} skipped (empty split)", source_id)
            continue

        all_train.append(tr)
        all_val.append(va)
        all_test.append(te)
        pos = int((tr["label"] == 1).sum())
        logger.info("{} -> {} windows (pos_train={})", source_id, len(windowed), pos)

    if not all_train:
        logger.error("No series processed successfully.")
        sys.exit(1)

    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    logger.info(
        "Datasets saved: train={}, val={}, test={}",
        len(train_df), len(val_df), len(test_df),
    )

    train_model(data_dir / "train.csv", data_dir / "val.csv", model_dir, cfg)

    evaluate_model(
        data_dir / "test.csv",
        model_dir / "model.joblib",
        model_dir / "threshold.json",
        report_dir,
        cfg,
    )

    logger.info("Training pipeline complete. Artifacts in {}", out)


# ── evaluate ─────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = Config()
    out = Path(args.artifacts)

    evaluate_model(
        out / "data" / "test.csv",
        out / "models" / "model.joblib",
        out / "models" / "threshold.json",
        out / "reports",
        cfg,
    )


# ── predict ──────────────────────────────────────────────────────────────────

def cmd_predict(args: argparse.Namespace) -> None:
    cfg = Config()
    out = Path(args.artifacts)

    predictor = Predictor(
        out / "models" / "model.joblib",
        out / "models" / "threshold.json",
    )

    client = NABClient(cfg, local_path=args.local_nab)

    matching = [f for f in cfg.aws_files if args.source in f]
    if not matching:
        logger.error("No source matching '{}'. Available: {}", args.source, cfg.aws_files)
        sys.exit(1)

    filename = matching[0]
    source_id = Path(filename).stem
    logger.info("Running alerting pipeline on {}", source_id)

    raw = client.load_series(filename)
    notifier: StdoutNotifier | ArchivingNotifier = StdoutNotifier()
    if getattr(args, "alert_log", None):
        notifier = ArchivingNotifier(notifier, args.alert_log)
    engine = AlertEngine(predictor, cfg, notifier=notifier)
    alerts = engine.run_on_series(
        raw["timestamp"].values, raw["value"].values, source_id,
    )

    logger.info(
        "Finished. {} alerts fired across {} data points.",
        len(alerts), len(raw),
    )


# ── stream ───────────────────────────────────────────────────────────────────

def cmd_stream(args: argparse.Namespace) -> None:
    """Read metrics from a file (or stdin) in a stream; output SSE events when
    the model predicts an incident (threshold crossed)."""
    cfg = Config()
    out = Path(args.artifacts)
    source_id = args.source or "stream"

    predictor = Predictor(
        out / "models" / "model.joblib",
        out / "models" / "threshold.json",
    )
    notifier: SSENotifier | ArchivingNotifier = SSENotifier()
    if getattr(args, "alert_log", None):
        notifier = ArchivingNotifier(notifier, args.alert_log)
    engine = AlertEngine(predictor, cfg, notifier=notifier)

    # Optional: send SSE comment so clients know the stream started
    sys.stdout.write(": stream started\n\n")
    sys.stdout.flush()

    count = 0
    for ts, value in read_metric_stream(args.input):
        count += 1
        engine.ingest(ts, value, source_id)

    logger.info("Stream finished. {} points processed, {} alerts fired.", count, len(engine.alerts_fired))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictive Cloud Alerting Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Prepare data, train model, evaluate")
    p_train.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))
    p_train.add_argument("--local-nab", default=None, help="Path to local NAB clone")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Re-evaluate a saved model on test data")
    p_eval.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))

    # predict
    p_pred = sub.add_parser("predict", help="Run alerting pipeline on a metric source")
    p_pred.add_argument("--source", required=True, help="Source name (partial match)")
    p_pred.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))
    p_pred.add_argument("--local-nab", default=None, help="Path to local NAB clone")
    p_pred.add_argument("--alert-log", default=None, help="Append alerts to this file (JSON lines) for archival")

    # stream: read file in stream, output SSE when threshold crossed
    p_stream = sub.add_parser("stream", help="Stream metrics from file/stdin; output SSE events on threshold")
    p_stream.add_argument("--input", default="-", help="Path to CSV (timestamp,value) or '-' for stdin")
    p_stream.add_argument("--source", default="stream", help="Source id for alert payloads")
    p_stream.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))
    p_stream.add_argument("--alert-log", default=None, help="Append alerts to this file (JSON lines) for archival")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "stream":
        cmd_stream(args)


if __name__ == "__main__":
    main()
