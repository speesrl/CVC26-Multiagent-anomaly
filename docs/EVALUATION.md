# Evaluation Methodology

This document describes the evaluation methodology used for the cascaded anomaly detection system.

## Metrics Overview

### 1. Detection Metrics

We report standard anomaly detection metrics on the UCF-Crime dataset:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Frame-level AUC** | Area under ROC curve for per-frame predictions | `sklearn.metrics.roc_auc_score` |
| **Average Precision** | Area under Precision-Recall curve | `sklearn.metrics.average_precision_score` |
| **Accuracy** | Correct predictions at optimal threshold | `(TP + TN) / (TP + TN + FP + FN)` |
| **F1-Score** | Harmonic mean of precision and recall | `2 * P * R / (P + R)` |

### 2. Reconstruction Metrics

For autoencoder quality assessment:

| Metric | Range | Better |
|--------|-------|--------|
| **PSNR** | 0-50 dB | Higher |
| **SSIM** | 0-1 | Higher |
| **MSE** | 0-1 | Lower |

### 3. Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **Latency (ms/frame)** | Per-stage processing time |
| **FPS** | Frames per second throughput |
| **Early Exit Rate** | Percentage of frames skipping Stage II |

## Datasets

### UCF-Crime

The primary benchmark (1900 videos, 13 anomaly categories, ~128 hours).
We follow the **standard full-video evaluation protocol**: all 290 test
videos are processed frame-by-frame from untrimmed sequences, and
frame-level AUC is computed using temporal annotations as ground truth.

### ShanghaiTech Campus

Secondary benchmark (437 videos, 13 scenes, 130 anomalous events).
Frame-level pixel masks are binarised to per-frame labels.

### XD-Violence

Tertiary benchmark (4754 videos, 6 anomaly categories, audio+visual).
Temporal segment annotations are converted to per-frame binary labels.

### Evaluation Protocol

1. **Training**: Train autoencoder only on normal videos
2. **Testing**: Evaluate on all test videos using full untrimmed sequences
3. **Frame-level**: Compute AUC over all frames with temporal annotations
4. **Video-level**: Max anomaly score per video for video-level AUC

### UCF-Crime Categories

| Category | Type | Description |
|----------|------|-------------|
| Abuse, Arrest, Arson, Assault, Burglary | Anomaly | Various |
| Explosion, Fighting, RoadAccidents | Anomaly | Various |
| Robbery, Shooting, Shoplifting, Stealing, Vandalism | Anomaly | Various |
| Normal | Normal | Regular surveillance footage |

## Baseline Comparison

### Methods Compared

| Method | Year | Approach | AUC |
|--------|------|----------|-----|
| **C3D + MIL** | 2018 | 3D CNN with multiple instance learning | 75.41% |
| **RTFM** | 2021 | Robust temporal feature magnitude | 84.30% |
| **MGFN** | 2023 | Multi-granularity feature network | 86.67% |
| **VadCLIP** | 2024 | CLIP-based video anomaly detection | 88.02% |
| **Ours** | 2026 | Cascaded AE + YOLO | 72.37% |

### Fair Comparison Notes

1. **Different objectives**: Our method optimizes for efficiency, not maximum accuracy
2. **Real-time constraint**: We target <10ms per frame
3. **Early exit**: ~72% of frames never reach expensive Stage II

## Ablation Study

### Configurations

| Config | Stage I | Stage II | Stage III |
|--------|---------|----------|-----------|
| **YOLO Only** | ✗ | ✓ | ✗ |
| **AE Only** | ✓ | ✗ | ✗ |
| **AE + YOLO** | ✓ | ✓ | ✗ |
| **Full** | ✓ | ✓ | ✓ |

### Insights

1. **AE Only**: Fast but lower accuracy on person-specific anomalies
2. **YOLO Only**: Better at person detection, misses texture anomalies
3. **Cascade**: Combines strengths with early-exit efficiency

## Latency Measurement

### Methodology

```python
# GPU-synchronized timing
torch.cuda.synchronize()
start = time.perf_counter()
# ... operation ...
torch.cuda.synchronize()
end = time.perf_counter()
```

### Warmup Protocol

- 10 warmup iterations before measurement
- 100 timed iterations for statistics
- Report mean, std, min, max

### Hardware Configuration

Results were measured on:
- GPU: NVIDIA RTX series (varies)
- CPU: Intel/AMD multi-core
- RAM: 16+ GB
- CUDA: 11.8+

## Detection vs Identification

### Clarification

Our system performs two distinct tasks that should not be conflated:

#### 1. Anomaly Detection (Quantitative)

- **Task**: Binary classification (normal vs anomaly)
- **Metric**: Frame-level AUC on UCF-Crime
- **Method**: Reconstruction error thresholding
- **Evaluation**: Standard ROC/PR curves

#### 2. Semantic Identification (Qualitative)

- **Task**: Categorize detected anomaly
- **Categories**: camera_blur, person_intrusion, suspicious_object, etc.
- **Method**: VLM-based scene description
- **Purpose**: Human operator assistance

### Why Different?

- Detection is the core ML task being evaluated
- Identification is a downstream UI feature
- AUC measures detection, not identification accuracy

## Multi-Agent System Evaluation

The multi-agent claims are validated with concrete system-level metrics:

### Scaling Benchmark

Tests throughput at 1, 2, 4, 8, and 16 concurrent camera streams:

| Metric | Description |
|--------|-------------|
| **FPS** | Frames processed per second across all streams |
| **Queue Depth** | Mean and max pending frames in the message queue |
| **Frames Dropped** | Frames lost when queue saturates |
| **Scaling Efficiency** | Actual FPS / (N x single-stream FPS) |

### Agent Mode Comparison

Compares event-driven (alarm-triggered) vs. cyclical (polling) agents on
identical workloads, measuring latency, throughput, and frame utilisation.

### Resource Utilisation

Continuous monitoring of CPU %, RAM (MB), and GPU memory during
multi-stream processing, reported as mean/max/min summaries.

See `multi_agent_benchmark.py` for implementation.

## Reproducibility

### Random Seeds

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Code Availability

All evaluation code is provided:
- `metrics.py`: AUC/ROC computation
- `full_video_evaluation.py`: Standard full-video UCF-Crime protocol
- `cross_dataset_evaluation.py`: Multi-dataset evaluation (UCF-Crime, ShanghaiTech, XD-Violence)
- `ablation_study.py`: Ablation experiments
- `latency_benchmark.py`: Timing measurements
- `multi_agent_benchmark.py`: Scaling, queueing, resource utilisation
- `run_evaluation.py`: Complete pipeline orchestrator

### Data Availability

- UCF-Crime: Publicly available from UCF
- Pre-extracted frames: Available upon request
- Trained weights: `autoencoder_model.pth` in repository
