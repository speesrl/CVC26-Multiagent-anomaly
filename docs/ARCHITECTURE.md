# Model Architecture

This document provides detailed specifications of the model architecture used in the cascaded anomaly detection system.

## System Overview

```
                    ┌─────────────────────────────────────┐
                    │           INPUT FRAME               │
                    │         (H×W×3 RGB)                 │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │          PREPROCESSING              │
                    │    Resize to 128×128, Normalize     │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                     STAGE I: AUTOENCODER                          │
    │                                                                   │
    │  ┌──────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
    │  │   ENCODER    │ → │  BOTTLENECK  │ → │      DECODER        │  │
    │  │ 3→32→64→128  │    │   128 ch    │    │   128→64→32→3      │  │
    │  └──────────────┘    └─────────────┘    └─────────────────────┘  │
    │                                                                   │
    │                  Anomaly Score = MSE(input, output)               │
    └───────────────────────────────────────────────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                score ≤ threshold            score > threshold
                         │                         │
                         ▼                         ▼
                  ┌─────────────┐    ┌─────────────────────────────────┐
                  │   NORMAL    │    │        STAGE II: YOLO           │
                  │   (Exit)    │    │                                 │
                  └─────────────┘    │   YOLOv8-nano person detection  │
                                     │   Output: person_detected bool  │
                                     └─────────────────────────────────┘
                                                   │
                                                   ▼
                                     ┌─────────────────────────────────┐
                                     │        STAGE III: VLM           │
                                     │   (Vision-Language Model)       │
                                     │                                 │
                                     │   Scene description + semantic  │
                                     │   category for operators        │
                                     └─────────────────────────────────┘
```

## Stage I: Convolutional Autoencoder

### Architecture Details

```python
class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for frame reconstruction.
    
    Input:  (B, 3, 128, 128)   - RGB frames
    Output: (B, 3, 128, 128)   - Reconstructed frames
    Latent: (B, 128, 16, 16)   - 32,768 dimensional bottleneck
    """
```

### Encoder

| Layer | Type | In Channels | Out Channels | Kernel | Stride | Padding | Output Size |
|-------|------|-------------|--------------|--------|--------|---------|-------------|
| 1 | Conv2d | 3 | 32 | 3×3 | 2 | 1 | 64×64×32 |
| - | ReLU | - | - | - | - | - | 64×64×32 |
| 2 | Conv2d | 32 | 64 | 3×3 | 2 | 1 | 32×32×64 |
| - | ReLU | - | - | - | - | - | 32×32×64 |
| 3 | Conv2d | 64 | 128 | 3×3 | 2 | 1 | 16×16×128 |
| - | ReLU | - | - | - | - | - | 16×16×128 |

**Total Encoder Parameters**: ~57K

### Decoder

| Layer | Type | In Channels | Out Channels | Kernel | Stride | Padding | Output Padding | Output Size |
|-------|------|-------------|--------------|--------|--------|---------|----------------|-------------|
| 1 | ConvTranspose2d | 128 | 64 | 3×3 | 2 | 1 | 1 | 32×32×64 |
| - | ReLU | - | - | - | - | - | - | 32×32×64 |
| 2 | ConvTranspose2d | 64 | 32 | 3×3 | 2 | 1 | 1 | 64×64×32 |
| - | ReLU | - | - | - | - | - | - | 64×64×32 |
| 3 | ConvTranspose2d | 32 | 3 | 3×3 | 2 | 1 | 1 | 128×128×3 |
| - | Sigmoid | - | - | - | - | - | - | 128×128×3 |

**Total Decoder Parameters**: ~57K

### Complete Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | ~115K |
| Model Size | ~460 KB |
| FLOPs (forward) | ~100M |
| Latency (GPU) | ~6.5 ms |
| Latency (CPU) | ~25 ms |

### PyTorch Implementation

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: 128x128 -> 16x16 with 128 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 128->64
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64->32
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32->16
            nn.ReLU(True)
        )
        
        # Decoder: 16x16 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

## Stage II: YOLOv8 Person Detection

### Model Selection

We use **YOLOv8-nano** for efficiency:

| Model | Parameters | mAP@0.5 | Latency |
|-------|------------|---------|---------|
| YOLOv8n | 3.2M | 37.3 | ~8 ms |
| YOLOv8s | 11.2M | 44.9 | ~15 ms |
| YOLOv8m | 25.9M | 50.2 | ~35 ms |

### Detection Logic

```python
def detect_person(frame):
    """
    Detect persons in frame using YOLOv8.
    
    Args:
        frame: RGB image (H, W, 3)
    
    Returns:
        person_detected: bool
        confidence: float (0-1)
        bounding_boxes: List[Tuple[x1, y1, x2, y2]]
    """
    results = yolo_model(frame, classes=[0])  # class 0 = person
    
    person_detected = len(results[0].boxes) > 0
    confidence = max([box.conf for box in results[0].boxes], default=0)
    
    return person_detected, confidence
```

## Stage III: Vision-Language Model

### Model Options

| Model | Size | Latency | Use Case |
|-------|------|---------|----------|
| BLIP-2 | 3B | ~2s | High quality descriptions |
| LLaVA | 7B | ~3s | Detailed scene understanding |
| GPT-4V | API | ~1s | Best quality (cloud) |

### Semantic Categories

The VLM maps detected anomalies to human-interpretable categories:

```python
SEMANTIC_CATEGORIES = {
    "camera_blur": "Camera lens obstruction or defocus",
    "person_intrusion": "Unauthorized person in frame",
    "suspicious_object": "Unattended or unusual object",
    "unusual_motion": "Abnormal movement pattern",
    "crowd_formation": "Unusual gathering of people",
    "vehicle_incident": "Vehicle-related anomaly",
    "environmental": "Lighting, weather, or scene change",
}
```

## Loss Functions

### Training Loss

```python
def reconstruction_loss(input, output):
    """MSE loss for autoencoder training."""
    return F.mse_loss(output, input)
```

### Anomaly Score

```python
def compute_anomaly_score(input, output):
    """Per-pixel MSE averaged over image."""
    return torch.mean((input - output) ** 2, dim=[1, 2, 3])
```

## Threshold Calibration

### Optimal Threshold Selection

We use Youden's J statistic to find optimal threshold:

```python
def find_optimal_threshold(scores, labels):
    """
    Find threshold that maximizes TPR - FPR.
    
    This balances sensitivity and specificity.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]
```

### Threshold Values

| Dataset | Optimal Threshold | TPR | FPR |
|---------|-------------------|-----|-----|
| UCF-Crime | 0.0035 | 85% | 32% |
| CUHK Avenue | 0.0028 | 82% | 28% |
| ShanghaiTech | 0.0031 | 80% | 25% |

## Computational Requirements

### Training

| Resource | Requirement |
|----------|-------------|
| GPU Memory | 4 GB minimum |
| Training Time | ~2 hours (100 epochs) |
| Dataset Size | ~10 GB (UCF-Crime) |

### Inference

| Resource | Stage I | Stage II | Stage III |
|----------|---------|----------|-----------|
| GPU Memory | 500 MB | 1 GB | 4 GB |
| Latency | 6.5 ms | 8.5 ms | 2000 ms |
| FPS | 153 | 118 | 0.5 |

## Model Checkpoints

### Saved Weights

The `autoencoder_model.pth` file contains:

```python
{
    'encoder.0.weight': ...,  # Conv2d(3, 32, 3, 2, 1)
    'encoder.0.bias': ...,
    'encoder.2.weight': ...,  # Conv2d(32, 64, 3, 2, 1)
    'encoder.2.bias': ...,
    'encoder.4.weight': ...,  # Conv2d(64, 128, 3, 2, 1)
    'encoder.4.bias': ...,
    'decoder.0.weight': ...,  # ConvTranspose2d(128, 64, 3, 2, 1, 1)
    'decoder.0.bias': ...,
    'decoder.2.weight': ...,  # ConvTranspose2d(64, 32, 3, 2, 1, 1)
    'decoder.2.bias': ...,
    'decoder.4.weight': ...,  # ConvTranspose2d(32, 3, 3, 2, 1, 1)
    'decoder.4.bias': ...,
}
```

### Loading Weights

```python
model = ConvAutoencoder()
state_dict = torch.load('autoencoder_model.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()
```
