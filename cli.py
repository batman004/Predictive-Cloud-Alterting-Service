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

from src.config import Config, get_metric_key, get_metric_key_from_source
from src.ml.features import (
    FEATURE_NAMES,
    build_window_df,
    label_series,
    temporal_split,
)
from src.ml.trainer import train as train_model
from src.ml.evaluator import evaluate as evaluate_model
from src.ml.predictor import Predictor
from src.ml.synthetic import generate_synthetic_data
from src.pipeline.ingest import NABClient, read_metric_stream
from src.pipeline.alert_engine import AlertEngine
from src.pipeline.notifier import ArchivingNotifier, SSENotifier, StdoutNotifier


ARTIFACTS_DEFAULT = Path("artifacts")

# Minimum rows per split to train a per-metric model
MIN_TRAIN_ROWS = 100


def _build_dataset(filenames, client, windows_map, cfg):
    """Build train/val/test DataFrames from a list of NAB filenames. Returns (train_df, val_df, test_df) or None."""
    import pandas as pd
    all_train, all_val, all_test = [], [], []
    for filename in filenames:
        source_id = Path(filename).stem
        try:
            raw = client.load_series(filename)
        except Exception as exc:
            logger.error("{} load failed: {}", source_id, exc)
            continue
        labeled = label_series(raw, windows_map.get(filename, []), cfg.H)
        windowed = build_window_df(labeled, source_id, cfg)
        if len(windowed) < 10:
            continue
        tr, va, te = temporal_split(windowed, cfg)
        if min(len(tr), len(va), len(te)) == 0:
            continue
        all_train.append(tr)
        all_val.append(va)
        all_test.append(te)
        pos = int((tr["label"] == 1).sum())
        logger.info("  {} -> {} windows (pos_train={})", source_id, len(windowed), pos)
    if not all_train:
        return None
    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)
    return train_df, val_df, test_df


def _build_synthetic_dataset(cfg):
    """Generate synthetic data and run it through the same feature/label/split pipeline."""
    import pandas as pd
    all_train, all_val, all_test = [], [], []
    for source_id, df, windows in generate_synthetic_data():
        labeled = label_series(df, windows, cfg.H)
        windowed = build_window_df(labeled, source_id, cfg)
        if len(windowed) < 10:
            continue
        tr, va, te = temporal_split(windowed, cfg)
        if min(len(tr), len(va), len(te)) == 0:
            continue
        all_train.append(tr)
        all_val.append(va)
        all_test.append(te)
        pos = int((tr["label"] == 1).sum())
        logger.info("  {} -> {} windows (pos_train={})", source_id, len(windowed), pos)
    if not all_train:
        return None
    return (
        pd.concat(all_train, ignore_index=True),
        pd.concat(all_val, ignore_index=True),
        pd.concat(all_test, ignore_index=True),
    )


def _get_model_dir(artifacts_root: Path, source_id: str | None, cfg: Config) -> Path:
    """Return the model directory to use: per-metric if available, else global. Supports legacy layout."""
    models_root = artifacts_root / "models"
    if source_id:
        key = get_metric_key_from_source(source_id, cfg.aws_files)
        if key and (models_root / key / "model.joblib").exists():
            return models_root / key
    if (models_root / "global" / "model.joblib").exists():
        return models_root / "global"
    return models_root


# ── train ────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config()
    dataset = getattr(args, "dataset", "aws")
    out = Path(args.artifacts)
    data_dir = out / "data"
    model_dir = out / "models"
    report_dir = out / "reports"

    import pandas as pd
    import json

    if dataset == "synthetic":
        logger.info("Building synthetic dataset")
        result = _build_synthetic_dataset(cfg)
    else:
        client = NABClient(cfg, local_path=args.local_nab)
        windows_map = client.load_windows()
        logger.info("Building global dataset (all series)")
        result = _build_dataset(list(cfg.aws_files), client, windows_map, cfg)

    if result is None:
        logger.error("No series processed successfully.")
        sys.exit(1)
    train_df, val_df, test_df = result
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    logger.info("Global: train={}, val={}, test={}", len(train_df), len(val_df), len(test_df))

    global_dir = model_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    train_model(data_dir / "train.csv", data_dir / "val.csv", global_dir, cfg)
    evaluate_model(
        data_dir / "test.csv",
        global_dir / "model.joblib",
        global_dir / "threshold.json",
        report_dir,
        cfg,
    )

    if dataset == "synthetic":
        (model_dir / "manifest.json").write_text(json.dumps({"models": ["global"]}, indent=2))
        logger.info("Training complete. Models: ['global']")
        return

    # ---- Per-metric models (AWS only) ----
    client = client  # already created above for aws path
    from collections import defaultdict
    by_key = defaultdict(list)
    for f in cfg.aws_files:
        by_key[get_metric_key(f)].append(f)

    manifest = ["global"]
    for key, filenames in sorted(by_key.items()):
        if key == "other":
            continue
        logger.info("Building dataset for metric key '{}' ({} series)", key, len(filenames))
        result = _build_dataset(filenames, client, windows_map, cfg)
        if result is None:
            logger.warning("  {}: skipped (no data)", key)
            continue
        train_df, val_df, test_df = result
        if len(train_df) < MIN_TRAIN_ROWS:
            logger.warning("  {}: skipped (too few rows: {})", key, len(train_df))
            continue
        key_dir = model_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        key_data = data_dir / key
        key_data.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(key_data / "train.csv", index=False)
        val_df.to_csv(key_data / "val.csv", index=False)
        test_df.to_csv(key_data / "test.csv", index=False)
        train_model(key_data / "train.csv", key_data / "val.csv", key_dir, cfg)
        key_report = report_dir / key
        evaluate_model(
            key_data / "test.csv",
            key_dir / "model.joblib",
            key_dir / "threshold.json",
            key_report,
            cfg,
        )
        manifest.append(key)

    (model_dir / "manifest.json").write_text(json.dumps({"models": manifest}, indent=2))
    logger.info("Training complete. Models: {}", manifest)


# ── evaluate ─────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = Config()
    out = Path(args.artifacts)
    model_dir = _get_model_dir(out, None, cfg)

    evaluate_model(
        out / "data" / "test.csv",
        model_dir / "model.joblib",
        model_dir / "threshold.json",
        out / "reports",
        cfg,
    )


# ── predict ──────────────────────────────────────────────────────────────────

def cmd_predict(args: argparse.Namespace) -> None:
    cfg = Config()
    out = Path(args.artifacts)
    dataset = getattr(args, "dataset", "aws")

    if dataset == "synthetic":
        # Load synthetic series; match --source to a synthetic service name
        from src.ml.synthetic import SERVICE_NAMES
        source_id = args.source
        matching = [s for s in SERVICE_NAMES if source_id.lower() in s]
        if not matching:
            logger.error(
                "No synthetic source matching '{}'. Available: {}",
                args.source, ", ".join(SERVICE_NAMES),
            )
            sys.exit(1)
        source_id = matching[0]
        # Generate all synthetic data and take the series we need
        raw_dfs = {sid: df for sid, df, _ in generate_synthetic_data()}
        raw = raw_dfs[source_id]
        raw = raw[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    else:
        client = NABClient(cfg, local_path=args.local_nab)
        matching = [f for f in cfg.aws_files if args.source in f]
        if not matching:
            logger.error("No source matching '{}'. Available: {}", args.source, cfg.aws_files)
            sys.exit(1)
        filename = matching[0]
        source_id = Path(filename).stem
        raw = client.load_series(filename)

    model_dir = _get_model_dir(out, source_id, cfg)
    logger.info("Using model {} for source {}", model_dir.name, source_id)

    predictor = Predictor(
        model_dir / "model.joblib",
        model_dir / "threshold.json",
    )

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
    if args.input != "-" and not Path(args.input).exists():
        logger.error("Input file not found: {}", args.input)
        sys.exit(1)
    cfg = Config()
    out = Path(args.artifacts)
    source_id = args.source or "stream"
    model_dir = _get_model_dir(out, source_id, cfg)
    logger.info("Using model {} for source {}", model_dir.name, source_id)

    predictor = Predictor(
        model_dir / "model.joblib",
        model_dir / "threshold.json",
    )
    notifier: SSENotifier | ArchivingNotifier = SSENotifier()
    if getattr(args, "alert_log", None):
        notifier = ArchivingNotifier(notifier, args.alert_log)
    debug = getattr(args, "debug", False)
    engine = AlertEngine(predictor, cfg, notifier=notifier, debug=debug)

    # Optional: send SSE comment so clients know the stream started
    sys.stdout.write(": stream started\n\n")
    sys.stdout.flush()

    count = 0
    for ts, value in read_metric_stream(args.input):
        count += 1
        engine.ingest(ts, value, source_id)

    if debug:
        logger.info("Debug: max probability = {:.4f} (alert threshold = {:.2f})", engine.max_prob_seen, predictor.threshold)
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
    p_train.add_argument("--dataset", choices=["aws", "synthetic"], default="aws", help="Data source: NAB AWS data or generated synthetic data")
    p_train.add_argument("--local-nab", default=None, help="Path to local NAB clone")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Re-evaluate a saved model on test data")
    p_eval.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))

    # predict
    p_pred = sub.add_parser("predict", help="Run alerting pipeline on a metric source")
    p_pred.add_argument("--source", required=True, help="Source name (partial match)")
    p_pred.add_argument("--dataset", choices=["aws", "synthetic"], default="aws", help="Data source: NAB AWS or synthetic (use with --source for synthetic service name)")
    p_pred.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))
    p_pred.add_argument("--local-nab", default=None, help="Path to local NAB clone")
    p_pred.add_argument("--alert-log", default=None, help="Append alerts to this file (JSON lines) for archival")

    # stream: read file in stream, output SSE when threshold crossed
    p_stream = sub.add_parser("stream", help="Stream metrics from file/stdin; output SSE events on threshold")
    p_stream.add_argument("--input", default="-", help="Path to CSV (timestamp,value) or '-' for stdin")
    p_stream.add_argument("--source", default="stream", help="Source id for alert payloads")
    p_stream.add_argument("--artifacts", default=str(ARTIFACTS_DEFAULT))
    p_stream.add_argument("--alert-log", default=None, help="Append alerts to this file (JSON lines) for archival")
    p_stream.add_argument("--debug", action="store_true", help="Log probability when > 0.15 and max prob at end")

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
