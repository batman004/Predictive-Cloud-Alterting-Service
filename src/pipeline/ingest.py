from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from loguru import logger

from src.config import Config


class NABClient:
    """Loads raw time-series data and anomaly windows from the NAB dataset."""

    def __init__(self, cfg: Config, local_path: str | None = None):
        self.cfg = cfg
        self.local_path = Path(local_path) if local_path else None

    def load_windows(self) -> dict[str, list]:
        if self.local_path:
            raw = (self.local_path / "labels" / "combined_windows.json").read_text()
        else:
            raw = requests.get(self.cfg.windows_url, timeout=30).text

        all_windows = json.loads(raw)
        return {k: v for k, v in all_windows.items() if k in self.cfg.aws_files}

    def load_series(self, filename: str) -> pd.DataFrame:
        if self.local_path:
            path = self.local_path / "data" / filename
            df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            url = self.cfg.data_url(filename)
            logger.debug("Fetching {}", url)
            df = pd.read_csv(
                io.StringIO(requests.get(url, timeout=30).text),
                parse_dates=["timestamp"],
            )

        df = df[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
        df["value"] = df["value"].ffill()
        return df.dropna(subset=["value"]).reset_index(drop=True)


def read_metric_stream(
    input_path: str,
) -> Iterator[tuple[str, float]]:
    """Read a CSV stream (file or stdin) line-by-line; yield (timestamp, value).

    Expects columns: timestamp, value. First row is skipped if it looks like a header
    (e.g. contains 'timestamp'). Use input_path='-' for stdin.
    """
    if input_path == "-":
        f = sys.stdin
    else:
        f = open(input_path, newline="", encoding="utf-8")

    try:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            ts = str(row[0]).strip()
            if ts.lower() in ("timestamp", "time", ""):
                continue
            try:
                val = float(row[1])
            except (TypeError, ValueError):
                continue
            yield ts, val
    finally:
        if input_path != "-":
            f.close()
