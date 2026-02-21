import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
WINDOWS_URL = f"{NAB_BASE}/labels/combined_windows.json"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
META_COLS = {"label", "timestamp", "source"}

AWS_FILES = [
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
    "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
    "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
    "realAWSCloudwatch/grok_asg_anomaly.csv",
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
]


def load_windows(local_nab: str | None) -> dict:
    if local_nab:
        raw = (Path(local_nab) / "labels" / "combined_windows.json").read_text()
    else:
        raw = requests.get(WINDOWS_URL, timeout=30).text
    all_windows = json.loads(raw)
    return {k: v for k, v in all_windows.items() if k in AWS_FILES}


def load_series(filename: str, local_nab: str | None) -> pd.DataFrame:
    if local_nab:
        path = Path(local_nab) / "data" / filename
        df = pd.read_csv(path, parse_dates=["timestamp"])
    else:
        url = f"{NAB_BASE}/data/{filename}"
        df = pd.read_csv(
            io.StringIO(requests.get(url, timeout=30).text), parse_dates=["timestamp"]
        )
    df = df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
    df["value"] = df["value"].ffill()
    return df.dropna(subset=["value"]).reset_index(drop=True)


def merge_windows(windows: list, freq_min: float) -> list:
    if not windows:
        return []
    step_td = pd.Timedelta(minutes=freq_min)
    parsed = sorted((pd.Timestamp(ws), pd.Timestamp(we)) for ws, we in windows)
    merged = []
    cur_start, cur_end = parsed[0]
    for nxt_start, nxt_end in parsed[1:]:
        if nxt_start <= cur_end + step_td:
            cur_end = max(cur_end, nxt_end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = nxt_start, nxt_end
    merged.append((cur_start, cur_end))
    return merged


def label_series(df: pd.DataFrame, windows: list, h: int) -> pd.DataFrame:
    out = df.copy()
    out["label"] = 0
    freq_min = out["timestamp"].diff().dropna().dt.total_seconds().median() / 60
    for ws, we in merge_windows(windows, freq_min):
        warning_start = ws - pd.Timedelta(minutes=freq_min * h)
        out.loc[
            (out["timestamp"] >= warning_start) & (out["timestamp"] < ws), "label"
        ] = 1
        out.loc[(out["timestamp"] >= ws) & (out["timestamp"] <= we), "label"] = -1
    return out


def extract_features(series: np.ndarray) -> dict:
    diffs = np.diff(series)
    x = np.arange(len(series), dtype=float)
    slope = np.polyfit(x, series, deg=1)[0] if len(series) > 1 else 0.0
    if len(series) > 1 and np.std(series[:-1]) > 0 and np.std(series[1:]) > 0:
        c = np.corrcoef(series[:-1], series[1:])[0, 1]
        autocorr = float(c) if np.isfinite(c) else 0.0
    else:
        autocorr = 0.0
    feats = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "trend_slope": float(slope),
        "mean_change": float(diffs.mean()) if len(diffs) else 0.0,
        "max_abs_change": float(np.abs(diffs).max()) if len(diffs) else 0.0,
        "autocorr_lag1": autocorr,
    }
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


def build_window_df(
    df: pd.DataFrame, source_id: str, w: int, h: int, step: int
) -> pd.DataFrame:
    values = df["value"].values
    labels = df["label"].values
    timestamps = df["timestamp"].values
    records = []
    for i in range(w, len(df) - h, step):
        if labels[i] == -1:
            continue
        row = extract_features(values[i - w : i])
        row["label"] = int(labels[i])
        row["timestamp"] = timestamps[i]
        row["source"] = source_id
        records.append(row)
    return pd.DataFrame(records)


def temporal_split(
    df: pd.DataFrame, w: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train = df.iloc[: max(0, train_end - w)].copy()
    val = df.iloc[train_end : max(train_end, val_end - w)].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def process_one(
    filename: str, windows: list, local_nab: str | None, w: int, h: int, step: int
):
    source_id = Path(filename).stem
    raw = load_series(filename, local_nab)
    labeled = label_series(raw, windows, h)
    windowed = build_window_df(labeled, source_id, w, h, step)
    if len(windowed) < 10:
        return None
    train, val, test = temporal_split(windowed, w)
    if min(len(train), len(val), len(test)) == 0:
        return None
    meta = {
        "source": source_id,
        "n_windows": len(windowed),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "pos_train": int((train["label"] == 1).sum()),
        "pos_val": int((val["label"] == 1).sum()),
        "pos_test": int((test["label"] == 1).sum()),
    }
    return train, val, test, meta


def run(local_nab: str | None, out_dir: Path, w: int, h: int, step: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_map = load_windows(local_nab)
    all_train, all_val, all_test, all_meta = [], [], [], []

    for filename in AWS_FILES:
        try:
            result = process_one(
                filename, windows_map.get(filename, []), local_nab, w, h, step
            )
        except Exception as exc:
            logger.error("{} failed: {}", Path(filename).name, exc)
            continue
        if result is None:
            logger.warning("{} skipped", Path(filename).name)
            continue
        train, val, test, meta = result
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
        all_meta.append(meta)
        logger.info(
            "{} -> {} windows (pos_train={})",
            meta["source"],
            meta["n_windows"],
            meta["pos_train"],
        )

    if not all_train:
        raise RuntimeError("No series processed.")

    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)
    meta_df = pd.DataFrame(all_meta)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    meta_df.to_csv(out_dir / "label_distribution.csv", index=False)

    summary = [
        f"W={w}, H={h}, STEP={step}",
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
        f"train_pos={int((train_df['label'] == 1).sum())}",
        f"val_pos={int((val_df['label'] == 1).sum())}",
        f"test_pos={int((test_df['label'] == 1).sum())}",
    ]
    (out_dir / "dataset_summary.txt").write_text("\n".join(summary) + "\n")

    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    nan_count = int(train_df[feature_cols].isnull().sum().sum())
    inf_count = int(
        np.isinf(train_df[feature_cols].select_dtypes("number").values).sum()
    )
    if nan_count or inf_count:
        raise RuntimeError(f"Invalid values found. NaN={nan_count}, Inf={inf_count}")

    logger.info(
        "Done. train={}, val={}, test={}", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df, meta_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NAB realAWSCloudwatch training data"
    )
    parser.add_argument("--local-nab", default=None, help="Path to local NAB clone")
    parser.add_argument("--W", type=int, default=24, help="Lookback window steps")
    parser.add_argument("--H", type=int, default=3, help="Prediction horizon steps")
    parser.add_argument("--step", type=int, default=1, help="Window stride")
    parser.add_argument("--out", default="./output", help="Output directory")
    args = parser.parse_args()
    run(args.local_nab, Path(args.out), args.W, args.H, args.step)


if __name__ == "__main__":
    main()
