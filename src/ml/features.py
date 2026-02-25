from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

FEATURE_NAMES = (
    "mean", "std", "min", "max",
    "trend_slope", "mean_change", "max_abs_change", "autocorr_lag1",
)


def merge_windows(windows: list, freq_min: float) -> list[tuple]:
    """Merge overlapping or adjacent anomaly windows into consolidated intervals."""
    if not windows:
        return []
    step_td = pd.Timedelta(minutes=freq_min)
    parsed = sorted((pd.Timestamp(s), pd.Timestamp(e)) for s, e in windows)
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
    """Assign labels: 1 = onset warning (within H steps before incident),
    -1 = incident in progress, 0 = normal."""
    out = df.copy()
    out["label"] = 0
    freq_min = out["timestamp"].diff().dropna().dt.total_seconds().median() / 60
    for ws, we in merge_windows(windows, freq_min):
        warning_start = ws - pd.Timedelta(minutes=freq_min * h)
        out.loc[
            (out["timestamp"] >= warning_start) & (out["timestamp"] < ws), "label"
        ] = 1
        out.loc[
            (out["timestamp"] >= ws) & (out["timestamp"] <= we), "label"
        ] = -1
    return out


def extract_features(series: np.ndarray) -> dict[str, float]:
    """Compute 8 statistical features from a 1-D numpy array (one sliding window)."""
    diffs = np.diff(series)

    slope = np.polyfit(np.arange(len(series), dtype=float), series, deg=1)[0] if len(series) > 1 else 0.0

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
    df: pd.DataFrame, source_id: str, cfg: Config,
) -> pd.DataFrame:
    """Slide a window of size W across the labeled series and produce a feature DataFrame.
    Skips windows where an incident is already in progress (label=-1)."""
    values = df["value"].values
    labels = df["label"].values
    timestamps = df["timestamp"].values
    records = []
    for i in range(cfg.W, len(df) - cfg.H, cfg.step):
        if labels[i] == -1:
            continue
        row = extract_features(values[i - cfg.W : i])
        row["label"] = int(labels[i])
        row["timestamp"] = timestamps[i]
        row["source"] = source_id
        records.append(row)
    return pd.DataFrame(records)


def temporal_split(
    df: pd.DataFrame, cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 split with a W-step gap to prevent leakage."""
    n = len(df)
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.val_frac))
    train = df.iloc[: max(0, train_end - cfg.W)].copy()
    val = df.iloc[train_end : max(train_end, val_end - cfg.W)].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test
