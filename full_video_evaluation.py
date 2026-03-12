"""
Standard Full-Video Evaluation Protocol for UCF-Crime

Implements the standard UCF-Crime evaluation protocol used by prior work
(VadCLIP, ProDisc-VAD, RTFM, etc.) where all 290 test videos are processed
frame-by-frame through the full untrimmed sequence, and frame-level AUC is
computed over all frames with temporal annotations as ground truth.

This addresses Reviewer 5A/5B/5C concern about fair evaluation protocol.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import cv2
from tqdm import tqdm
from torchvision import transforms

from metrics import (
    evaluate_anomaly_detection,
    EvaluationResult,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from ucf_crime_loader import UCFCrimeDataset


class ConvAutoencoder(nn.Module):
    """Autoencoder matching training notebook: 3->32->64->128 channels."""
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class VideoResult:
    """Per-video evaluation result."""
    video_name: str
    category: str
    n_frames: int
    n_anomaly_frames: int
    max_score: float
    mean_score: float
    video_label: int  # 1 if any anomaly frame, 0 otherwise


@dataclass
class FullVideoEvalResult:
    """Complete evaluation under full-video protocol."""
    frame_auc: float
    video_auc: float
    average_precision: float
    accuracy_at_optimal: float
    f1_at_optimal: float
    optimal_threshold: float
    total_frames: int
    total_videos: int
    anomaly_videos: int
    normal_videos: int
    per_video: List[VideoResult]
    eval_result: EvaluationResult


def load_ae_model(model_path: str, device: torch.device) -> nn.Module:
    """Load autoencoder with version-compatible weight loading."""
    model = ConvAutoencoder().to(device)
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_video_scores(
    model: nn.Module,
    video_path: str,
    device: torch.device,
    image_size: Tuple[int, int] = (128, 128),
    batch_size: int = 64,
    sample_rate: int = 1,
) -> np.ndarray:
    """
    Compute per-frame reconstruction error for an entire video.

    Processes the video in batches for GPU efficiency.  Returns an array
    of MSE scores with one entry per sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    scores = []
    frame_buffer = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb)
            frame_buffer.append(tensor)

            if len(frame_buffer) >= batch_size:
                batch = torch.stack(frame_buffer).to(device, non_blocking=True)
                with torch.inference_mode():
                    recon = model(batch)
                    mse = F.mse_loss(recon, batch, reduction='none')
                    mse = mse.mean(dim=(1, 2, 3))
                scores.extend(mse.cpu().numpy().tolist())
                frame_buffer = []

        frame_idx += 1

    # flush remaining frames
    if frame_buffer:
        batch = torch.stack(frame_buffer).to(device, non_blocking=True)
        with torch.inference_mode():
            recon = model(batch)
            mse = F.mse_loss(recon, batch, reduction='none')
            mse = mse.mean(dim=(1, 2, 3))
        scores.extend(mse.cpu().numpy().tolist())

    cap.release()
    return np.array(scores)


def run_full_video_evaluation(
    ucf_crime_root: str,
    ae_model_path: str,
    device: torch.device = None,
    sample_rate: int = 1,
    batch_size: int = 64,
    max_videos: Optional[int] = None,
    output_dir: str = "evaluation_results",
) -> FullVideoEvalResult:
    """
    Run the standard full-video UCF-Crime evaluation protocol.

    1. Load all 290 test videos (anomaly + normal).
    2. For each video, compute per-frame reconstruction scores.
    3. Generate frame-level labels from temporal annotations.
    4. Compute frame-level AUC, video-level AUC, AP, F1.

    Args:
        ucf_crime_root: Root of the UCF-Crime dataset
        ae_model_path: Path to trained autoencoder weights
        device: Torch device (auto-detected if None)
        sample_rate: Process every Nth frame (1 = all frames)
        batch_size: GPU batch size
        max_videos: Cap on number of test videos (None = all)
        output_dir: Directory for saving plots and JSON
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Sample rate: every {sample_rate} frame(s)")

    # Load model
    model = load_ae_model(ae_model_path, device)
    # Warmup
    _ = model(torch.zeros(1, 3, 128, 128, device=device))
    print("Autoencoder loaded and warmed up.")

    # Load dataset annotations
    dataset = UCFCrimeDataset(ucf_crime_root)
    test_videos = dataset.get_test_videos()
    normal_test = dataset.get_normal_test_videos()

    all_videos = [(v, True) for v in test_videos]  # (name, is_annotated)
    all_videos += [(v, False) for v in normal_test]

    if max_videos:
        all_videos = all_videos[:max_videos]

    print(f"Total test videos to process: {len(all_videos)}")

    all_scores = []
    all_labels = []
    all_video_ids = []
    per_video_results = []
    video_id = 0

    for video_name, is_annotated in tqdm(all_videos, desc="Processing videos"):
        video_path = dataset.get_video_path(video_name)
        if video_path is None:
            print(f"  Skipping (not found): {video_name}")
            continue

        try:
            scores = compute_video_scores(
                model, str(video_path), device,
                batch_size=batch_size, sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            continue

        n_frames = len(scores)
        if n_frames == 0:
            continue

        total_video_frames = n_frames * sample_rate
        labels = dataset.get_frame_level_labels(video_name, total_video_frames)
        # subsample labels to match sampled frames
        labels = labels[::sample_rate][:n_frames]

        n_anomaly = int(labels.sum())
        category = dataset._extract_category(video_name)
        video_label = 1 if n_anomaly > 0 else 0

        per_video_results.append(VideoResult(
            video_name=video_name,
            category=category,
            n_frames=n_frames,
            n_anomaly_frames=n_anomaly,
            max_score=float(scores.max()),
            mean_score=float(scores.mean()),
            video_label=video_label,
        ))

        all_scores.append(scores)
        all_labels.append(labels)
        all_video_ids.extend([video_id] * n_frames)
        video_id += 1

    if not all_scores:
        raise RuntimeError("No videos could be processed. Check dataset path.")

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_video_ids = np.array(all_video_ids)

    n_anomaly_vids = sum(1 for v in per_video_results if v.video_label == 1)
    n_normal_vids = sum(1 for v in per_video_results if v.video_label == 0)

    print(f"\nTotal frames evaluated: {len(all_scores):,}")
    print(f"Anomaly frames: {int(all_labels.sum()):,} "
          f"({all_labels.mean()*100:.1f}%)")
    print(f"Videos: {len(per_video_results)} "
          f"({n_anomaly_vids} anomaly, {n_normal_vids} normal)")

    # Evaluate
    eval_result = evaluate_anomaly_detection(
        all_scores, all_labels, all_video_ids
    )

    result = FullVideoEvalResult(
        frame_auc=eval_result.frame_auc,
        video_auc=eval_result.video_auc,
        average_precision=eval_result.average_precision,
        accuracy_at_optimal=eval_result.accuracy_at_optimal,
        f1_at_optimal=eval_result.f1_at_optimal,
        optimal_threshold=eval_result.optimal_threshold,
        total_frames=len(all_scores),
        total_videos=len(per_video_results),
        anomaly_videos=n_anomaly_vids,
        normal_videos=n_normal_vids,
        per_video=per_video_results,
        eval_result=eval_result,
    )

    # Print results
    print("\n" + "=" * 65)
    print("  FULL-VIDEO EVALUATION RESULTS (Standard UCF-Crime Protocol)")
    print("=" * 65)
    print(f"  Frame-level AUC:     {result.frame_auc*100:.2f}%")
    print(f"  Video-level AUC:     {result.video_auc*100:.2f}%")
    print(f"  Average Precision:   {result.average_precision*100:.2f}%")
    print(f"  Accuracy @ Optimal:  {result.accuracy_at_optimal*100:.2f}%")
    print(f"  F1 @ Optimal:        {result.f1_at_optimal*100:.2f}%")
    print(f"  Optimal Threshold:   {result.optimal_threshold:.6f}")
    print(f"  Total Frames:        {result.total_frames:,}")
    print(f"  Total Videos:        {result.total_videos}")
    print("=" * 65)

    # Save
    os.makedirs(output_dir, exist_ok=True)

    plot_roc_curve(
        eval_result,
        title="ROC Curve - Full-Video UCF-Crime Protocol",
        save_path=os.path.join(output_dir, "roc_full_video.png"),
    )
    plot_precision_recall_curve(
        eval_result,
        title="PR Curve - Full-Video UCF-Crime Protocol",
        save_path=os.path.join(output_dir, "pr_full_video.png"),
    )

    summary = {
        "protocol": "full-video (standard UCF-Crime)",
        "frame_auc": result.frame_auc,
        "video_auc": result.video_auc,
        "average_precision": result.average_precision,
        "accuracy_at_optimal": result.accuracy_at_optimal,
        "f1_at_optimal": result.f1_at_optimal,
        "optimal_threshold": result.optimal_threshold,
        "total_frames": result.total_frames,
        "total_videos": result.total_videos,
        "anomaly_videos": result.anomaly_videos,
        "normal_videos": result.normal_videos,
        "sample_rate": sample_rate,
    }
    with open(os.path.join(output_dir, "full_video_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standard full-video UCF-Crime evaluation"
    )
    parser.add_argument("--ucf-crime", type=str, required=True,
                        help="Root directory of UCF-Crime dataset")
    parser.add_argument("--ae-model", type=str,
                        default="autoencoder_model.pth")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Sample every Nth frame (1=all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos (for quick test)")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation_results")
    args = parser.parse_args()

    run_full_video_evaluation(
        ucf_crime_root=args.ucf_crime,
        ae_model_path=args.ae_model,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        max_videos=args.max_videos,
        output_dir=args.output_dir,
    )
