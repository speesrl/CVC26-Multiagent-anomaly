# CVC26-Multiagent-anomaly

# Cascaded Surveillance Anomaly Detection with Vision–Language Foundation Model Reasoning and Semantic Label Stabilization

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-11.8+-76B900.svg" alt="CUDA"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [System Architecture](#-system-architecture)
- [Model Details](#-model-details)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🎯 Overview

This repository contains the official implementation of **"Cascaded Surveillance Anomaly Detection with Vision–Language Foundation Model Reasoning and Semantic Label Stabilization"**.

We propose a **three-stage cascaded pipeline** for real-time surveillance anomaly detection that combines:

| Stage | Component | Purpose | Latency |
|-------|-----------|---------|---------|
| **I** | Convolutional Autoencoder | Fast anomaly gating via reconstruction error | 6.5 ms |
| **II** | YOLOv8 Object Detection | Person/object semantic classification | 8.5 ms |
| **III** | Vision-Language Model | Human-interpretable explanations | ~2.3 s |

### Why Cascaded?

Traditional approaches process every frame through expensive models. Our cascade **exits early** for normal frames:

```
📊 Efficiency Gains:
├── 72% of frames exit at Stage I (never reach YOLO)
├── 95% of frames exit at Stage II (never reach VLM)
└── Only true anomalies trigger full pipeline
```

---

## 🌟 Key Contributions

1. **Cascaded Architecture**: Multi-stage pipeline with early-exit mechanism for computational efficiency
2. **Reconstruction-based Gating**: Lightweight autoencoder filters normal frames before expensive detection
3. **Semantic Label Stabilization**: VLM reasoning provides consistent, human-interpretable anomaly categories
4. **Real-time Performance**: ~152 FPS for Stage I, enabling deployment on edge devices

---

## 🏗️ System Architecture

### Dual-Stage Perception Pipeline

<p align="center">
  <img src="dual_stage_perception.png" alt="Dual Stage Perception Architecture" width="800"/>
</p>

### High-Level Pipeline

```
                              ┌─────────────────────────────────────────┐
                              │         SURVEILLANCE CAMERA             │
                              │            (Video Stream)               │
                              └─────────────────────────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    PREPROCESSING                                        │
│                                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│   │ Frame Grab   │ → │  Resize to   │ → │  Normalize   │ → │  To Tensor   │        │
│   │              │    │   128×128    │    │   [0, 1]     │    │  (B,3,H,W)   │        │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE I: AUTOENCODER GATE                                     │
│                                  (6.5 ms/frame)                                         │
│                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              ENCODER                                             │  │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────────────────┐   │  │
│   │  │  Input  │    │ Conv2D  │    │ Conv2D  │    │        Conv2D               │   │  │
│   │  │ 128×128 │ → │ 3→32 ch │ → │ 32→64ch │ → │       64→128 ch             │   │  │
│   │  │  ×3 ch  │    │  k=3×3  │    │  k=3×3  │    │        k=3×3               │   │  │
│   │  └─────────┘    │  s=2    │    │  s=2    │    │         s=2                │   │  │
│   │                 │ ReLU    │    │ ReLU    │    │        ReLU                │   │  │
│   │                 │ 64×64   │    │ 32×32   │    │       16×16                │   │  │
│   │                 └─────────┘    └─────────┘    └─────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                                │
│                            Bottleneck: 16×16×128 = 32,768 dims                         │
│                                        │                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              DECODER                                             │  │
│   │  ┌─────────────────────────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │  │
│   │  │      ConvTranspose2D        │    │ConvT2D  │    │ConvT2D  │    │ Output  │   │  │
│   │  │        128→64 ch            │ → │ 64→32ch │ → │ 32→3 ch │ → │ 128×128 │   │  │
│   │  │          k=3×3              │    │  k=3×3  │    │  k=3×3  │    │  ×3 ch  │   │  │
│   │  │           s=2               │    │  s=2    │    │  s=2    │    │         │   │  │
│   │  │          ReLU               │    │ ReLU    │    │ Sigmoid │    │         │   │  │
│   │  │         32×32               │    │ 64×64   │    │ 128×128 │    │         │   │  │
│   │  └─────────────────────────────┘    └─────────┘    └─────────┘    └─────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                         │
│                    Anomaly Score = MSE(Input, Reconstruction)                          │
│                                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         │                                               │
                   score ≤ τ                                       score > τ
                   (Normal)                                        (Anomaly)
                         │                                               │
                         ▼                                               ▼
              ┌──────────────────┐                    ┌──────────────────────────────────┐
              │   ✅ NORMAL      │                    │      STAGE II: YOLO DETECTION    │
              │   Early Exit     │                    │           (8.5 ms/frame)         │
              │                  │                    │                                   │
              │  ~72% of frames  │                    │  ┌───────────────────────────┐   │
              │  exit here       │                    │  │      YOLOv8-nano          │   │
              └──────────────────┘                    │  │                           │   │
                                                      │  │  • Person Detection       │   │
                                                      │  │  • Object Classification  │   │
                                                      │  │  • Bounding Boxes         │   │
                                                      │  │  • Confidence Scores      │   │
                                                      │  └───────────────────────────┘   │
                                                      └──────────────────────────────────┘
                                                                       │
                                              ┌────────────────────────┴────────────────────────┐
                                              │                                                  │
                                        No Person                                          Person Found
                                              │                                                  │
                                              ▼                                                  ▼
                               ┌──────────────────────┐                       ┌─────────────────────────────────┐
                               │  ⚠️ Generic Anomaly  │                       │    STAGE III: VLM REASONING     │
                               │                      │                       │          (~2.3 s/event)         │
                               │  Motion/Texture      │                       │                                  │
                               │  Anomaly             │                       │  ┌───────────────────────────┐  │
                               └──────────────────────┘                       │  │   Vision-Language Model   │  │
                                                                              │  │                           │  │
                                                                              │  │  "Describe this scene     │  │
                                                                              │  │   and identify the        │  │
                                                                              │  │   anomalous activity"     │  │
                                                                              │  │                           │  │
                                                                              │  └───────────────────────────┘  │
                                                                              │                                  │
                                                                              │  Output: Semantic Label          │
                                                                              │  • person_intrusion              │
                                                                              │  • suspicious_behavior           │
                                                                              │  • violent_activity              │
                                                                              │  • theft_attempt                 │
                                                                              └─────────────────────────────────┘
```

### Semantic Label Categories

| Category | Description | Trigger Condition |
|----------|-------------|-------------------|
| `camera_blur` | Camera lens obstruction or defocus | High reconstruction error, no person |
| `person_intrusion` | Unauthorized person detected | Person in restricted zone |
| `suspicious_behavior` | Unusual movement patterns | Person + abnormal pose/motion |
| `violent_activity` | Physical altercation | Multiple persons + rapid motion |
| `theft_attempt` | Suspicious object interaction | Person + reaching/grabbing motion |
| `environmental` | Lighting/weather anomaly | High error, scene-wide change |

---

## 🔧 Model Details

### Autoencoder Architecture

```python
ConvAutoencoder(
  (encoder): Sequential(
    (0): Conv2d(3, 32, kernel_size=3, stride=2, padding=1)    # 128→64
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 64→32
    (3): ReLU(inplace=True)
    (4): Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 32→16
    (5): ReLU(inplace=True)
  )
  (decoder): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
    (1): ReLU(inplace=True)
    (2): ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
    (3): ReLU(inplace=True)
    (4): ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
    (5): Sigmoid()
  )
)
```

| Property | Value |
|----------|-------|
| Parameters | ~115K |
| Model Size | ~460 KB |
| Input Shape | (B, 3, 128, 128) |
| Bottleneck | (B, 128, 16, 16) |
| Compression Ratio | 1.5:1 |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Epochs | 100 |
| Loss Function | MSE |
| Training Data | Normal frames only |

---

## 📊 Results

### Detection Performance (UCF-Crime Dataset)

| Metric | Value | Description |
|--------|-------|-------------|
| **Frame-level AUC** | 72.37% | Area under ROC curve |
| **Average Precision** | 91.67% | Area under PR curve |
| **Accuracy @ Optimal τ** | 75.21% | At threshold = 0.001104 |
| **F1-Score @ Optimal τ** | 81.04% | Harmonic mean of P/R |
| **Optimal Threshold** | 0.001104 | MSE threshold value |

### Reconstruction Quality

| Metric | Normal Frames | Anomaly Frames |
|--------|---------------|----------------|
| PSNR | 28-32 dB | 18-24 dB |
| SSIM | 0.90-0.95 | 0.70-0.85 |
| MSE | 0.001-0.002 | 0.004-0.010 |

### Efficiency Benchmarks

| Stage | Mean Latency | Std Dev | FPS | GPU Memory |
|-------|--------------|---------|-----|------------|
| Preprocessing | 0.5 ms | 0.1 ms | 2000 | - |
| Stage I (AE) | 6.55 ms | 0.8 ms | 152.7 | 500 MB |
| Stage II (YOLO) | 8.45 ms | 1.2 ms | 118.3 | 1 GB |
| Stage III (VLM) | 2300 ms | 200 ms | 0.4 | 4 GB |
| **Total (early exit)** | **7.05 ms** | - | **141.8** | - |
| **Total (full pipeline)** | **2315 ms** | - | **0.4** | - |

### Early Exit Statistics

| Dataset | Exit @ Stage I (AE gate) |
|---------|--------------------------|
| UCF-Crime | 72.1% |

Additional datasets (ShanghaiTech, XD-Violence) can be evaluated using `cross_dataset_evaluation.py`.

### Baseline Comparison

| Method | Year | AUC (%) | Notes |
|--------|------|---------|-------|
| C3D + MIL | 2018 | 75.41 | 3D CNN |
| RTFM | 2021 | 84.30 | Temporal features |
| MGFN | 2023 | 86.67 | Multi-granularity |
| ProDisc-VAD | 2025 | 87.31 | Prototype + discriminative |
| Ex-VAD | 2024 | 86.92 | Explainable VLM |
| VadCLIP | 2024 | 88.02 | CLIP-based |
| **Ours (Cascade)** | 2026 | **74.47** | Efficiency + interpretability |

> The cascade trades some detection accuracy for a threefold reduction in latency and the ability to provide real-time semantic explanations via selective VLM invocation.

### Ablation Study

| Configuration | AUC (%) | Exit Rate (%) |
|---------------|---------|---------------|
| YOLO Only | 72.15 | 0.0 |
| AE Only (Ours) | 72.37 | 0.0 |
| AE + YOLO Cascade | 73.87 | 68.3 |
| Full Cascade (AE+YOLO+VLM) | 74.47 | 72.1 |

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended for GPU acceleration)
- 8 GB RAM minimum
- NVIDIA GPU with 4+ GB VRAM (optional)

### Quick Start

```bash
# Clone repository
git clone https://github.com/speesrl/CVC26-Multiagent-anomaly.git
cd CVC26-Multiagent-anomaly

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
matplotlib>=3.7.0
Pillow>=10.0.0
tqdm>=4.65.0
```

### Dataset Setup

1. **Download UCF-Crime Dataset**:
   - Official: [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/)
   - Extract frames to `Test/` folder

2. **Expected Structure**:
```
Test/
├── Arrest/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── Arson/
├── Assault/
├── Burglary/
├── Explosion/
├── Fighting/
└── NormalVideos/
```

---

## 📖 Usage

### Run Complete Evaluation

```bash
python run_evaluation.py --data-dir ./Test --model ./autoencoder_model.pth
```

Options:
- `--data-dir`: Path to UCF-Crime test frames
- `--model`: Path to autoencoder weights
- `--output-dir`: Results directory (default: `./evaluation_results`)
- `--skip-ablation`: Skip ablation study
- `--skip-latency`: Skip latency benchmark
- `--demo`: Run with synthetic data

### Run Interactive Dashboard

```bash
python anomaly_dashboard.py
```

Features:
- Real-time frame processing
- Reconstruction visualization
- Anomaly score display
- Semantic label output

### Python API

```python
import torch
from anomaly_dashboard import ConvAutoencoder
from metrics import evaluate_anomaly_detection

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder()
model.load_state_dict(torch.load("autoencoder_model.pth", weights_only=True))
model.to(device).eval()

# Process frame
frame = preprocess(image)  # (1, 3, 128, 128)
reconstruction = model(frame)
anomaly_score = torch.mean((frame - reconstruction) ** 2).item()

# Threshold decision
threshold = 0.0035
is_anomaly = anomaly_score > threshold
```

### Standard Full-Video UCF-Crime Evaluation

```bash
python full_video_evaluation.py --ucf-crime /path/to/UCF-Crime \
    --ae-model autoencoder_model.pth --sample-rate 5
```

### Cross-Dataset Evaluation (UCF-Crime + ShanghaiTech + XD-Violence)

```bash
python cross_dataset_evaluation.py \
    --ucf-crime /path/to/UCF-Crime \
    --shanghaitech /path/to/ShanghaiTech \
    --xd-violence /path/to/XD-Violence \
    --ae-model autoencoder_model.pth
```

### Multi-Agent System Benchmark

```bash
python multi_agent_benchmark.py --ae-model autoencoder_model.pth \
    --streams 1,2,4,8,16 --duration 5.0
```

### Run Ablation Study

```bash
python ablation_study.py --ae-model autoencoder_model.pth --demo
```

### Run Latency Benchmark

```bash
python latency_benchmark.py --ae-model autoencoder_model.pth --n-frames 100
```

---

## 📁 Project Structure

```
CVC26-Multiagent-anomaly/
│
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
│
├── 🧠 autoencoder_model.pth        # Trained model weights (240 KB)
│
├── 📓 AAMS.ipynb                   # Training notebook
│
├── 🖥️ anomaly_dashboard.py         # GUI dashboard application
│
├── 🔬 Evaluation Scripts
│   ├── run_evaluation.py           # Main evaluation pipeline
│   ├── full_video_evaluation.py    # Standard full-video UCF-Crime protocol
│   ├── cross_dataset_evaluation.py # Multi-dataset eval (UCF/SHT/XD)
│   ├── multi_agent_benchmark.py    # Scaling, queueing, resource metrics
│   ├── metrics.py                  # AUC/ROC/PR computation
│   ├── ablation_study.py           # Ablation experiments
│   ├── latency_benchmark.py        # Timing measurements
│   ├── ucf_crime_loader.py         # UCF-Crime dataset loader
│   ├── shanghaitech_loader.py      # ShanghaiTech Campus loader
│   └── xd_violence_loader.py       # XD-Violence dataset loader
│
├── 📊 Results Generation
│   ├── generate_results.py         # Generate metric tables
│   └── final_metrics.py            # Final paper metrics
│
├── 📚 docs/
│   ├── EVALUATION.md               # Evaluation methodology
│   └── ARCHITECTURE.md             # Detailed architecture specs
│
└── 🗂️ Test/                        # UCF-Crime frames (not in repo)
    ├── Arrest/
    ├── Arson/
    └── ...
```

---

## 🔬 Methodology

### Problem Formulation

Given surveillance video V = {f₁, f₂, ..., fₜ}, predict frame-level anomaly labels Y = {y₁, y₂, ..., yₜ} where yₜ ∈ {0, 1}.

### Stage I: Reconstruction Gate

The autoencoder is trained on normal frames only:

```
L_AE = (1/N) Σ ||fᵢ - f̂ᵢ||²
```

Anomaly score for frame f:

```
s(f) = (1/HW) Σ (f_hw - f̂_hw)²
```

Decision rule:

```
ŷ = 1  if s(f) > τ
    0  otherwise
```

### Stage II: Semantic Classification

For frames with s(f) > τ, apply YOLOv8:

```
boxes, classes, conf = YOLO(f)
person_detected = ∃ c ∈ classes : c = "person"
```

### Stage III: VLM Reasoning

For person-containing anomalies:

```
label = VLM(prompt, f)
```

Where prompt = "Describe the anomalous activity in this surveillance frame."

### Detection vs Identification

| Task | Metric | Method |
|------|--------|--------|
| **Detection** | AUC-ROC | Reconstruction error thresholding |
| **Identification** | Human evaluation | VLM semantic labeling |

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{rehman2026cascaded,
  title={Cascaded Surveillance Anomaly Detection with Vision--Language Foundation Model Reasoning and Semantic Label Stabilization},
  author={Rehman, Tayyab and De Gasperis, Giovanni and Shmahell, Aly},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Spee s.r.l** — Industry partner and sponsor
- **University of L'Aquila** — Academic support
- [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/) — Real-world Anomaly Detection in Surveillance Videos
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- Baseline methods: C3D-MIL, RTFM, MGFN, VadCLIP

---

## 📧 Contact

For questions, issues, or collaboration:

- **Email**: [tayyab.rehman@graduate.univaq.it](mailto:tayyab.rehman@graduate.univaq.it)
- **GitHub Issues**: [Open an issue](https://github.com/speesrl/CVC26-Multiagent-anomaly/issues)
- **Organization**: [Spee s.r.l](https://github.com/speesrl)

---

<p align="center">
  Tayyab Rehman — University of L'Aquila / SPEE S.R.L.
</p>
