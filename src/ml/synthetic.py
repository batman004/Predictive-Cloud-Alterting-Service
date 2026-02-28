"""Generate synthetic cloud metric time series with realistic pre-incident patterns.

Each service produces a long time series (5-min intervals) with periodic incidents.
Unlike NAB data, incidents here have clear *precursors* -- gradual ramp-ups in mean,
variance, or trend -- so an ML model can learn to predict them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
STEP_MINUTES = 5
STEPS_PER_DAY = 288


def _daily_pattern(n: int, amplitude: float, phase: float) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return amplitude * np.sin(2 * np.pi * t / STEPS_PER_DAY + phase)


def _generate_service(
    name: str,
    n_steps: int,
    baseline: float,
    noise_std: float,
    daily_amp: float,
    ramp_height: float,
    incident_height: float,
    incident_len_range: tuple[int, int],
    n_incidents: int,
    rng: np.random.Generator,
    ramp_style: str = "linear",
) -> tuple[str, pd.DataFrame, list[list[str]]]:
    """Build one service's time series and anomaly windows."""
    phase = rng.uniform(0, 2 * np.pi)
    values = baseline + _daily_pattern(n_steps, daily_amp, phase) + rng.normal(0, noise_std, n_steps)

    ramp_len = 12
    windows: list[list[str]] = []

    avg_inc_len = (incident_len_range[0] + incident_len_range[1]) // 2
    segment = (n_steps - 100) // n_incidents
    for k in range(n_incidents):
        seg_start = 50 + k * segment
        latest_start = seg_start + segment - ramp_len - avg_inc_len - 25
        pos = int(rng.integers(seg_start + 10, max(seg_start + 11, latest_start)))

        inc_len = int(rng.integers(incident_len_range[0], incident_len_range[1] + 1))
        ramp_start = pos
        ramp_end = pos + ramp_len
        inc_start = ramp_end
        inc_end = min(inc_start + inc_len, n_steps - 5)
        if inc_end <= inc_start:
            continue

        if ramp_style == "linear":
            values[ramp_start:ramp_end] += np.linspace(0, ramp_height, ramp_len)
            values[ramp_start:ramp_end] += rng.normal(0, noise_std * 1.5, ramp_len)
        elif ramp_style == "exponential":
            values[ramp_start:ramp_end] += ramp_height * (np.exp(np.linspace(0, 2, ramp_len)) - 1) / (np.e**2 - 1)
            values[ramp_start:ramp_end] += rng.normal(0, noise_std * 2, ramp_len)
        elif ramp_style == "step":
            n_steps_up = ramp_len // 3
            for j in range(3):
                s = ramp_start + j * n_steps_up
                e = min(s + n_steps_up, ramp_end)
                values[s:e] += ramp_height * (j + 1) / 3
            values[ramp_start:ramp_end] += rng.normal(0, noise_std * 1.5, ramp_len)

        actual_inc_len = inc_end - inc_start
        values[inc_start:inc_end] = baseline + incident_height + rng.normal(0, noise_std * 0.5, actual_inc_len)

        recovery_len = min(20, n_steps - inc_end)
        if recovery_len > 0:
            decay = incident_height * np.exp(-np.linspace(0, 3, recovery_len))
            values[inc_end:inc_end + recovery_len] = baseline + decay + rng.normal(0, noise_std, recovery_len)

        windows.append([
            _step_to_ts(inc_start),
            _step_to_ts(inc_end - 1),
        ])

    values = np.clip(values, 0, None)

    timestamps = pd.date_range("2024-01-01", periods=n_steps, freq=f"{STEP_MINUTES}min")
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    return name, df, windows


_EPOCH = pd.Timestamp("2024-01-01")


def _step_to_ts(step: int) -> str:
    return str(_EPOCH + pd.Timedelta(minutes=step * STEP_MINUTES))


SERVICE_SPECS = [
    dict(baseline=30, noise_std=3, daily_amp=5, ramp_height=40, incident_height=65, incident_len_range=(10, 25), ramp_style="linear"),
    dict(baseline=40, noise_std=4, daily_amp=3, ramp_height=35, incident_height=55, incident_len_range=(15, 30), ramp_style="linear"),
    dict(baseline=50, noise_std=8, daily_amp=10, ramp_height=80, incident_height=150, incident_len_range=(8, 20), ramp_style="exponential"),
    dict(baseline=0.5, noise_std=0.2, daily_amp=0.1, ramp_height=3, incident_height=8, incident_len_range=(10, 20), ramp_style="step"),
    dict(baseline=25, noise_std=2, daily_amp=4, ramp_height=50, incident_height=70, incident_len_range=(12, 22), ramp_style="linear"),
    dict(baseline=60, noise_std=5, daily_amp=8, ramp_height=30, incident_height=40, incident_len_range=(10, 18), ramp_style="exponential"),
    dict(baseline=35, noise_std=3, daily_amp=6, ramp_height=45, incident_height=60, incident_len_range=(10, 25), ramp_style="linear"),
    dict(baseline=100, noise_std=15, daily_amp=20, ramp_height=150, incident_height=300, incident_len_range=(8, 15), ramp_style="exponential"),
    dict(baseline=10, noise_std=1.5, daily_amp=2, ramp_height=15, incident_height=30, incident_len_range=(15, 30), ramp_style="step"),
    dict(baseline=45, noise_std=4, daily_amp=7, ramp_height=35, incident_height=50, incident_len_range=(10, 20), ramp_style="linear"),
]

SERVICE_NAMES = [
    "cpu_web_server", "cpu_api_server", "latency_gateway",
    "error_rate_auth", "cpu_worker", "mem_cache_server",
    "cpu_db_replica", "network_ingress", "error_rate_payment",
    "cpu_scheduler",
]


def generate_synthetic_data(
    n_steps: int = 4000,
    n_incidents: int = 12,
    seed: int = SEED,
) -> list[tuple[str, pd.DataFrame, list[list[str]]]]:
    """Return a list of (source_id, dataframe, anomaly_windows) tuples."""
    rng = np.random.default_rng(seed)
    results = []
    for name, spec in zip(SERVICE_NAMES, SERVICE_SPECS):
        source_id, df, windows = _generate_service(
            name=name,
            n_steps=n_steps,
            n_incidents=n_incidents,
            rng=rng,
            **spec,
        )
        results.append((source_id, df, windows))
    return results
