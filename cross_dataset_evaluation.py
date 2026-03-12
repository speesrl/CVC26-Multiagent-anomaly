"""
Cross-Dataset Evaluation for Cascaded Anomaly Detection

Runs the standard evaluation protocol on multiple benchmarks:
  - UCF-Crime       (1900 videos, 13 anomaly categories)
  - ShanghaiTech    (437 videos, 13 scenes)
  - XD-Violence     (4754 videos, 6 anomaly categories)

Addresses Reviewer 5A: "evaluated only on a single dataset -- broader
validation on ShanghaiTech or XD-Violence is needed."
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from metrics import evaluate_anomaly_detection, EvaluationResult, plot_roc_curve


class ConvAutoencoder(nn.Module):
    """Autoencoder matching training notebook: 3->32->64->128 channels."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class DatasetResult:
    dataset_name: str
    frame_auc: float
    video_auc: float
    average_precision: float
    f1_at_optimal: float
    accuracy_at_optimal: float
    optimal_threshold: float
    total_frames: int
    total_videos: int
    exit_rate: float
    mean_latency_ms: float


def load_model(path: str, device: torch.device) -> nn.Module:
    model = ConvAutoencoder().to(device)
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def score_frames(
    model: nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, float]:
    """Compute reconstruction error and average latency for a batch of frames."""
    tensors = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    scores = []
    times = []

    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            recon = model(batch)
            mse = F.mse_loss(recon, batch, reduction='none').mean(dim=(1, 2, 3))
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000 / len(batch))
            scores.extend(mse.cpu().numpy().tolist())

    avg_lat = float(np.mean(times)) if times else 0.0
    return np.array(scores), avg_lat


def evaluate_dataset(
    dataset_iter,
    model: nn.Module,
    device: torch.device,
    dataset_name: str,
    ae_threshold: float = 0.0015,
    max_videos: Optional[int] = None,
) -> DatasetResult:
    """
    Generic evaluation loop: iterate over (video_id, frames, labels),
    accumulate scores, compute metrics.
    """
    all_scores, all_labels, all_vids = [], [], []
    latencies = []
    vid_id = 0
    n_processed = 0

    for video_name, frames, labels in tqdm(dataset_iter,
                                           desc=f"Eval {dataset_name}"):
        if max_videos and n_processed >= max_videos:
            break
        scores, lat = score_frames(model, frames, device)
        latencies.append(lat)
        all_scores.append(scores)
        all_labels.append(labels[:len(scores)])
        all_vids.extend([vid_id] * len(scores))
        vid_id += 1
        n_processed += 1

    if not all_scores:
        print(f"  No data for {dataset_name}")
        return DatasetResult(dataset_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_vids = np.array(all_vids)

    result = evaluate_anomaly_detection(all_scores, all_labels, all_vids)
    exit_rate = (all_scores <= ae_threshold).sum() / len(all_scores)

    return DatasetResult(
        dataset_name=dataset_name,
        frame_auc=result.frame_auc,
        video_auc=result.video_auc,
        average_precision=result.average_precision,
        f1_at_optimal=result.f1_at_optimal,
        accuracy_at_optimal=result.accuracy_at_optimal,
        optimal_threshold=result.optimal_threshold,
        total_frames=len(all_scores),
        total_videos=vid_id,
        exit_rate=float(exit_rate),
        mean_latency_ms=float(np.mean(latencies)),
    )


def run_cross_dataset(
    ae_model_path: str,
    ucf_crime_root: Optional[str] = None,
    shanghaitech_root: Optional[str] = None,
    xd_violence_root: Optional[str] = None,
    sample_rate: int = 5,
    max_videos: Optional[int] = None,
    output_dir: str = "evaluation_results",
) -> List[DatasetResult]:
    """Run evaluation across all available datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ae_model_path, device)
    _ = model(torch.zeros(1, 3, 128, 128, device=device))

    results: List[DatasetResult] = []

    # UCF-Crime
    if ucf_crime_root and Path(ucf_crime_root).exists():
        from ucf_crime_loader import UCFCrimeDataset
        ds = UCFCrimeDataset(ucf_crime_root)
        r = evaluate_dataset(
            ds.iterate_test_set(sample_rate=sample_rate),
            model, device, "UCF-Crime", max_videos=max_videos,
        )
        results.append(r)
    else:
        print("UCF-Crime root not provided or not found, skipping.")

    # ShanghaiTech
    if shanghaitech_root and Path(shanghaitech_root).exists():
        from shanghaitech_loader import ShanghaiTechDataset
        ds = ShanghaiTechDataset(shanghaitech_root)
        r = evaluate_dataset(
            ds.iterate_test_set(sample_rate=sample_rate),
            model, device, "ShanghaiTech", max_videos=max_videos,
        )
        results.append(r)
    else:
        print("ShanghaiTech root not provided or not found, skipping.")

    # XD-Violence
    if xd_violence_root and Path(xd_violence_root).exists():
        from xd_violence_loader import XDViolenceDataset
        ds = XDViolenceDataset(xd_violence_root)
        r = evaluate_dataset(
            ds.iterate_test_set(sample_rate=sample_rate),
            model, device, "XD-Violence", max_videos=max_videos,
        )
        results.append(r)
    else:
        print("XD-Violence root not provided or not found, skipping.")

    # Print comparison table
    print("\n" + "=" * 90)
    print("  CROSS-DATASET EVALUATION RESULTS")
    print("=" * 90)
    hdr = (f"{'Dataset':<16} {'Frame AUC':>10} {'Video AUC':>10} "
           f"{'AP':>8} {'F1':>8} {'Exit Rate':>10} {'Lat (ms)':>10} "
           f"{'Frames':>10}")
    print(hdr)
    print("-" * 90)
    for r in results:
        row = (f"{r.dataset_name:<16} {r.frame_auc*100:>9.2f}% "
               f"{r.video_auc*100:>9.2f}% {r.average_precision*100:>7.2f}% "
               f"{r.f1_at_optimal*100:>7.2f}% {r.exit_rate*100:>9.1f}% "
               f"{r.mean_latency_ms:>9.2f} {r.total_frames:>10,}")
        print(row)
    print("=" * 90)

    # LaTeX
    latex = r"""\begin{table}[t]
\centering
\caption{Cross-dataset evaluation of the cascaded framework}
\label{tab:cross_dataset}
\begin{tabular}{lccccc}
\toprule
\textbf{Dataset} & \textbf{Frame AUC (\%)} & \textbf{Video AUC (\%)} & \textbf{AP (\%)} & \textbf{F1 (\%)} & \textbf{Exit Rate (\%)} \\
\midrule
"""
    for r in results:
        latex += (f"{r.dataset_name} & {r.frame_auc*100:.2f} & "
                  f"{r.video_auc*100:.2f} & {r.average_precision*100:.2f} & "
                  f"{r.f1_at_optimal*100:.2f} & {r.exit_rate*100:.1f} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    print("\nLaTeX table:")
    print(latex)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    summary = [{
        "dataset": r.dataset_name,
        "frame_auc": r.frame_auc,
        "video_auc": r.video_auc,
        "average_precision": r.average_precision,
        "f1": r.f1_at_optimal,
        "accuracy": r.accuracy_at_optimal,
        "exit_rate": r.exit_rate,
        "total_frames": r.total_frames,
        "total_videos": r.total_videos,
    } for r in results]
    with open(os.path.join(output_dir, "cross_dataset_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-dataset anomaly detection evaluation"
    )
    parser.add_argument("--ae-model", type=str,
                        default="autoencoder_model.pth")
    parser.add_argument("--ucf-crime", type=str, default=None)
    parser.add_argument("--shanghaitech", type=str, default=None)
    parser.add_argument("--xd-violence", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output-dir", type=str,
                        default="evaluation_results")
    args = parser.parse_args()

    run_cross_dataset(
        ae_model_path=args.ae_model,
        ucf_crime_root=args.ucf_crime,
        shanghaitech_root=args.shanghaitech,
        xd_violence_root=args.xd_violence,
        sample_rate=args.sample_rate,
        max_videos=args.max_videos,
        output_dir=args.output_dir,
    )
