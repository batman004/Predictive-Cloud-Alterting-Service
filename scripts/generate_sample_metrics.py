#!/usr/bin/env python3
"""Generate a sample metrics CSV for testing the stream/predict pipeline.

Output: timestamp, value (5-min intervals). Includes a normal baseline, then
a pre-incident ramp and spike so a model trained on synthetic data may fire alerts.

Usage:
    python scripts/generate_sample_metrics.py
    python scripts/generate_sample_metrics.py -o my_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Match pipeline: 5-min steps
STEP_MINUTES = 5
W = 24  # need at least W points before model can predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample metrics CSV")
    parser.add_argument("-o", "--output", default="sample_metrics.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = 400  # ~33 hours at 5-min steps
    timestamps = pd.date_range("2024-01-15T00:00:00", periods=n, freq=f"{STEP_MINUTES}min")

    # Match synthetic: baseline + light noise, then linear ramp (12 steps) then incident
    baseline = 35.0
    ramp_len = 12
    ramp_height = 40.0
    incident_height = 55.0  # baseline + this = spike level
    values = baseline + rng.normal(0, 2.5, n)

    # First incident: exact synthetic pattern (linear ramp then spike)
    ramp_start = 80
    values[ramp_start : ramp_start + ramp_len] += np.linspace(0, ramp_height, ramp_len) + rng.normal(0, 1.0, ramp_len)
    inc_start = ramp_start + ramp_len
    inc_end = min(inc_start + 20, n - 5)
    values[inc_start:inc_end] = baseline + incident_height + rng.normal(0, 1.0, inc_end - inc_start)
    # Recovery
    rec_len = min(20, n - inc_end)
    if rec_len > 0:
        decay = incident_height * np.exp(-np.linspace(0, 3, rec_len))
        values[inc_end : inc_end + rec_len] = baseline + decay + rng.normal(0, 2, rec_len)

    # Second incident (so we have two clear alerts)
    ramp_start2 = 280
    values[ramp_start2 : ramp_start2 + ramp_len] += np.linspace(0, 38.0, ramp_len) + rng.normal(0, 1.0, ramp_len)
    inc_start2 = ramp_start2 + ramp_len
    inc_end2 = min(inc_start2 + 15, n - 2)
    values[inc_start2:inc_end2] = baseline + 50.0 + rng.normal(0, 1.0, inc_end2 - inc_start2)

    values = np.clip(values, 0, None)

    out = Path(args.output)
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}", file=sys.stderr)
    print("Run: python cli.py stream --input", out, "--source sample-service", file=sys.stderr)


if __name__ == "__main__":
    main()
