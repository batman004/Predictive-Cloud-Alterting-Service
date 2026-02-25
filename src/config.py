from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Single source of truth for all pipeline parameters."""

    # sliding-window parameters
    W: int = 24
    H: int = 3
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
