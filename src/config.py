from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def get_metric_key(rel_path: str) -> str:
    """Map a NAB file path to a metric type key for per-metric models."""
    name = Path(rel_path).stem.lower()
    if "ec2_cpu_utilization" in name or name.startswith("ec2_cpu"):
        return "ec2_cpu"
    if "ec2_disk_write" in name:
        return "ec2_disk"
    if "ec2_network_in" in name or "networkin" in name:
        return "ec2_network"
    if "elb_request" in name:
        return "elb"
    if "grok_asg" in name or "asg_anomaly" in name:
        return "asg"
    if "rds_cpu_utilization" in name:
        return "rds_cpu"
    return "other"


def get_metric_key_from_source(source_id: str, aws_files: tuple[str, ...]) -> str | None:
    """Resolve source_id (e.g. from --source) to a metric key if it matches a known series."""
    source_lower = source_id.lower()
    for rel_path in aws_files:
        stem = Path(rel_path).stem.lower()
        if source_lower in stem or stem in source_lower:
            return get_metric_key(rel_path)
    return None


@dataclass(frozen=True)
class Config:
    """Single source of truth for all pipeline parameters."""

    # sliding-window parameters
    W: int = 24
    H: int = 12
    step: int = 1

    # train / val / test split fractions
    train_frac: float = 0.70
    val_frac: float = 0.15

    # alerting parameters
    cooldown_steps: int = 10
    critical_threshold: float = 0.80

    # NAB dataset
    nab_base_url: str = "https://raw.githubusercontent.com/numenta/NAB/master"
    aws_files: tuple[str, ...] = field(default_factory=lambda: (
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
    ))

    # columns that are metadata, not features
    meta_cols: frozenset[str] = frozenset({"label", "timestamp", "source"})

    @property
    def windows_url(self) -> str:
        return f"{self.nab_base_url}/labels/combined_windows.json"

    def data_url(self, rel_path: str) -> str:
        return f"{self.nab_base_url}/data/{rel_path}"
