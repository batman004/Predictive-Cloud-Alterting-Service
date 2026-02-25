# Predictive Cloud Alerting Service

A production-ready pipeline that predicts incidents in cloud services before they occur, using historical metric data from AWS CloudWatch.

## Architecture

The system is split into two layers:

- **ML layer** (`src/ml/`) -- stateless, pure functions: feature extraction, model training, evaluation, inference. Swap the model without touching the pipeline.
- **Pipeline layer** (`src/pipeline/`) -- owns state and I/O: data ingestion, rolling-window alert engine with cooldown logic, structured JSON notification output.

```
src/
├── config.py              # Frozen dataclass -- single source of truth for all parameters
├── ml/
│   ├── features.py        # 8 statistical features, onset labeling, sliding window, temporal split
│   ├── trainer.py         # XGBoost training with class-imbalance handling + threshold tuning
│   ├── evaluator.py       # Point-level (precision/recall/AUC) + incident-level metrics
│   └── predictor.py       # Loads trained model, returns (probability, should_alert)
└── pipeline/
    ├── ingest.py          # NAB dataset client (swap for CloudWatch/Prometheus later)
    ├── alert_engine.py    # Rolling buffer, cooldown, severity bands
    └── notifier.py        # Structured JSON alerts to stdout
```

## Quick Start

### Prerequisites

- Python 3.12+
- Dependencies: `pip install -r requirements.txt`

### Train the model

```bash
python cli.py train
```

This downloads 17 AWS CloudWatch time series from the NAB dataset, extracts features, trains an XGBoost classifier, tunes the alert threshold, and evaluates on a held-out test set. Artifacts are saved to `artifacts/`.

### Evaluate a saved model

```bash
python cli.py evaluate
```

### Run the alerting pipeline

```bash
python cli.py predict --source ec2_cpu_utilization_fe7f93
```

Streams through the time series chronologically, simulating real-time data arrival. Fires structured JSON alerts to stdout when the model predicts an incident:

```json
{"timestamp": "2014-02-17T00:17:00", "source": "ec2_cpu_utilization_fe7f93", "probability": 0.9653, "severity": "critical", "message": "Predicted incident within 15 minutes"}
```

### Docker

```bash
docker build -t cloud-alerting .
docker run -v $(pwd)/artifacts:/app/artifacts cloud-alerting train
docker run -v $(pwd)/artifacts:/app/artifacts cloud-alerting predict --source ec2_cpu_utilization_fe7f93
```

## How It Works

### Problem Formulation

Binary time-series classification: given the last **W=24** observations (2 hours at 5-minute intervals), predict whether an incident will start within the next **H=3** steps (15 minutes).

### Feature Set (8 features)

| Feature | Description |
|---------|-------------|
| `mean` | Mean value over the window |
| `std` | Standard deviation |
| `min` / `max` | Range boundaries |
| `trend_slope` | Linear regression slope (is the metric rising or falling?) |
| `mean_change` | Average step-to-step change |
| `max_abs_change` | Largest single-step jump |
| `autocorr_lag1` | Lag-1 autocorrelation (is the signal smooth or noisy?) |

### Labeling

- **label=1** (onset): timestamp is within H steps before an incident window
- **label=-1** (incident in progress): excluded from training to avoid data leakage
- **label=0** (normal): everything else

### Alert Engine

The alert engine (`src/pipeline/alert_engine.py`) processes data points one at a time:

1. Appends value to a rolling buffer of size W
2. Extracts features from the buffer
3. Runs model inference
4. Checks threshold + cooldown (suppresses duplicate alerts for 10 steps after firing)
5. Classifies severity: `critical` (p >= 0.80) or `warning` (p >= threshold)

### Dataset

[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) -- `realAWSCloudwatch` subset (17 time series covering EC2 CPU, disk, network, ELB, and RDS metrics).

## Design Decisions

- **XGBoost over deep learning**: fast to train, interpretable feature importances, works well with small datasets and tabular features. No GPU required.
- **Univariate approach**: each metric stream is modeled independently. Keeps the pipeline simple and allows per-metric alerting without cross-metric dependencies.
- **Temporal split (70/15/15)**: chronological ordering preserved with a W-step gap between splits to prevent data leakage.
- **Cooldown logic**: prevents alert storms. After firing, the engine suppresses for N steps before allowing another alert on the same source.
- **Structured JSON stdout**: composable with any log aggregator (CloudWatch Logs, ELK, Datadog). No vendor lock-in.

## Limitations and Future Work

- **Class imbalance**: only ~0.15% of training samples are positive. Current `scale_pos_weight` helps but more sophisticated sampling (SMOTE, undersampling) or loss functions could improve recall.
- **Univariate only**: cross-metric correlation (e.g., CPU spike + network drop) could improve prediction accuracy.
- **Static threshold**: a per-source adaptive threshold based on historical alert patterns could reduce false positives.
- **NAB dataset only**: integrating real CloudWatch data requires swapping `src/pipeline/ingest.py` -- the pipeline architecture supports this without other changes.
