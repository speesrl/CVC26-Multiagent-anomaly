"""
Multi-Agent System Benchmark

Measures scaling, queueing, throughput under load, and resource
utilization for the cascaded multi-agent anomaly detection framework.

Addresses Reviewer 5C: "the multi-agent framing is interesting, but
the results do not support the claims -- scaling, queueing, throughput
under load, resource utilization etc. are missing."

Benchmarks:
  1. Throughput vs. concurrent camera streams (1 to 16)
  2. Queue depth and latency under varying load
  3. Resource utilization (CPU, GPU memory, system RAM)
  4. Scaling efficiency factor
  5. Event-driven vs. cyclical agent performance comparison
"""

import os
import sys
import time
import json
import threading
import queue
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =========================================================================
# Model (matches training notebook)
# =========================================================================

class ConvAutoencoder(nn.Module):
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


# =========================================================================
# Data classes for results
# =========================================================================

@dataclass
class ResourceSnapshot:
    timestamp_s: float
    cpu_percent: float
    ram_used_mb: float
    ram_percent: float
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_util_percent: float = 0.0


@dataclass
class ThroughputResult:
    n_streams: int
    total_frames: int
    elapsed_s: float
    fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    queue_depth_mean: float
    queue_depth_max: int
    frames_dropped: int


@dataclass
class ScalingResult:
    """Complete scaling benchmark output."""
    throughput_by_streams: List[ThroughputResult]
    resource_snapshots: List[ResourceSnapshot]
    scaling_efficiency: Dict[int, float]
    device: str
    ae_threshold: float


# =========================================================================
# Resource monitor
# =========================================================================

class ResourceMonitor:
    """Background thread that periodically snapshots system resources."""

    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self._snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._snapshots = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[ResourceSnapshot]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return list(self._snapshots)

    def _loop(self):
        t0 = time.monotonic()
        while self._running:
            snap = ResourceSnapshot(
                timestamp_s=time.monotonic() - t0,
                cpu_percent=psutil.cpu_percent() if HAS_PSUTIL else 0.0,
                ram_used_mb=(psutil.virtual_memory().used / 1e6
                             if HAS_PSUTIL else 0.0),
                ram_percent=(psutil.virtual_memory().percent
                             if HAS_PSUTIL else 0.0),
            )
            if torch.cuda.is_available():
                snap.gpu_mem_used_mb = torch.cuda.memory_allocated() / 1e6
                snap.gpu_mem_total_mb = torch.cuda.get_device_properties(0).total_mem / 1e6
                try:
                    snap.gpu_util_percent = torch.cuda.utilization()
                except Exception:
                    pass
            self._snapshots.append(snap)
            time.sleep(self.interval)


# =========================================================================
# Simulated camera stream
# =========================================================================

class SimulatedCameraStream:
    """
    Simulates a surveillance camera sending frames at a target FPS.

    Frames are synthetic random images; the purpose is to measure
    pipeline throughput, not detection accuracy.
    """

    def __init__(
        self,
        stream_id: int,
        target_fps: float = 30.0,
        image_size: Tuple[int, int] = (128, 128),
        duration_s: float = 5.0,
    ):
        self.stream_id = stream_id
        self.target_fps = target_fps
        self.image_size = image_size
        self.n_frames = int(target_fps * duration_s)
        self._idx = 0

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self._idx >= self.n_frames:
            raise StopIteration
        frame = torch.rand(1, 3, *self.image_size)
        self._idx += 1
        return frame


# =========================================================================
# Main benchmark
# =========================================================================

class MultiAgentBenchmark:
    """
    Benchmarks the cascaded pipeline under multi-stream load.

    For each concurrency level (1, 2, 4, 8, 16 streams), we:
      - create N simulated camera streams
      - dispatch frames through a shared processing queue
      - measure per-frame latency, throughput, queue depth, drops
      - record CPU/GPU/RAM usage
    """

    def __init__(
        self,
        ae_model_path: Optional[str] = None,
        yolo_model_path: str = "yolov8n.pt",
        device: str = None,
        ae_threshold: float = 0.0015,
        image_size: Tuple[int, int] = (128, 128),
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ae_threshold = ae_threshold
        self.image_size = image_size

        # Load AE
        self.ae_model = ConvAutoencoder().to(self.device)
        if ae_model_path and Path(ae_model_path).exists():
            try:
                sd = torch.load(ae_model_path, map_location=self.device,
                                weights_only=True)
            except TypeError:
                sd = torch.load(ae_model_path, map_location=self.device)
            self.ae_model.load_state_dict(sd)
        self.ae_model.eval()

        # YOLO (lazy)
        self._yolo = None
        self._yolo_path = yolo_model_path

    @property
    def yolo_model(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self._yolo_path)
            except Exception:
                pass
        return self._yolo

    # ------------------------------------------------------------------
    # Single-frame pipeline (AE gate -> optional YOLO)
    # ------------------------------------------------------------------

    def _process_frame(self, frame_tensor: torch.Tensor) -> Tuple[float, bool]:
        """Returns (latency_ms, exited_early)."""
        frame_tensor = frame_tensor.to(self.device, non_blocking=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            recon = self.ae_model(frame_tensor)
            err = F.mse_loss(recon, frame_tensor).item()

        exited = err <= self.ae_threshold
        if not exited and self.yolo_model is not None:
            dummy = np.random.randint(0, 255, (*self.image_size, 3),
                                      dtype=np.uint8)
            self.yolo_model(dummy, verbose=False)

        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        return latency_ms, exited

    # ------------------------------------------------------------------
    # Queue-based multi-stream benchmark
    # ------------------------------------------------------------------

    def benchmark_streams(
        self,
        n_streams: int,
        stream_fps: float = 30.0,
        duration_s: float = 5.0,
        max_queue_depth: int = 256,
    ) -> ThroughputResult:
        """
        Benchmark with `n_streams` concurrent camera streams.
        """
        q: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        latencies: List[float] = []
        n_early_exit = 0
        n_dropped = 0
        queue_depths: List[int] = []
        done_producing = threading.Event()

        # Producer: pushes frames from all streams
        def producer():
            streams = [
                SimulatedCameraStream(i, stream_fps, self.image_size, duration_s)
                for i in range(n_streams)
            ]
            for stream in streams:
                for frame in stream:
                    try:
                        q.put_nowait((stream.stream_id, frame))
                    except queue.Full:
                        nonlocal n_dropped
                        n_dropped += 1
                    # pace to approximate real-time
                    time.sleep(1.0 / (stream_fps * n_streams + 1e-9))
            done_producing.set()

        # Consumer: processes frames from the queue
        def consumer():
            nonlocal n_early_exit
            while not (done_producing.is_set() and q.empty()):
                try:
                    _sid, frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                queue_depths.append(q.qsize())
                lat, exited = self._process_frame(frame)
                latencies.append(lat)
                if exited:
                    n_early_exit += 1

        prod_thread = threading.Thread(target=producer, daemon=True)
        cons_thread = threading.Thread(target=consumer, daemon=True)

        start = time.perf_counter()
        prod_thread.start()
        cons_thread.start()
        prod_thread.join()
        cons_thread.join()
        elapsed = time.perf_counter() - start

        total_processed = len(latencies)
        fps = total_processed / elapsed if elapsed > 0 else 0

        return ThroughputResult(
            n_streams=n_streams,
            total_frames=total_processed,
            elapsed_s=elapsed,
            fps=fps,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=(sorted(latencies)[int(0.95 * len(latencies))]
                            if latencies else 0),
            queue_depth_mean=(statistics.mean(queue_depths)
                              if queue_depths else 0),
            queue_depth_max=max(queue_depths) if queue_depths else 0,
            frames_dropped=n_dropped,
        )

    # ------------------------------------------------------------------
    # Full scaling study
    # ------------------------------------------------------------------

    def run_scaling_study(
        self,
        stream_counts: List[int] = None,
        stream_fps: float = 30.0,
        duration_s: float = 5.0,
    ) -> ScalingResult:
        """
        Run the full multi-agent scaling study.

        Tests throughput at 1, 2, 4, 8, 16 concurrent streams and records
        resource utilization throughout.
        """
        if stream_counts is None:
            stream_counts = [1, 2, 4, 8, 16]

        # Warmup
        dummy = torch.rand(1, 3, *self.image_size, device=self.device)
        with torch.inference_mode():
            _ = self.ae_model(dummy)

        monitor = ResourceMonitor(interval_s=0.5)
        monitor.start()

        results: List[ThroughputResult] = []
        for n in stream_counts:
            print(f"  Benchmarking {n} stream(s)...")
            r = self.benchmark_streams(n, stream_fps, duration_s)
            results.append(r)
            print(f"    FPS: {r.fps:.1f} | Latency: {r.mean_latency_ms:.1f} ms | "
                  f"Queue max: {r.queue_depth_max} | Dropped: {r.frames_dropped}")

        snapshots = monitor.stop()

        # Compute scaling efficiency: ideal = n * single_stream_fps
        base_fps = results[0].fps if results else 1.0
        efficiency = {}
        for r in results:
            ideal = base_fps * r.n_streams
            efficiency[r.n_streams] = r.fps / ideal if ideal > 0 else 0.0

        return ScalingResult(
            throughput_by_streams=results,
            resource_snapshots=snapshots,
            scaling_efficiency=efficiency,
            device=self.device,
            ae_threshold=self.ae_threshold,
        )

    # ------------------------------------------------------------------
    # Event-driven vs. cyclical agent comparison
    # ------------------------------------------------------------------

    def benchmark_agent_modes(
        self, n_frames: int = 500, anomaly_rate: float = 0.1
    ) -> Dict[str, Dict]:
        """
        Compare event-driven (alarm triggered) vs. cyclical (polling) agents.

        Returns dict with latency & throughput for each mode.
        """
        frames = [torch.rand(1, 3, *self.image_size) for _ in range(n_frames)]
        is_alarm = np.random.random(n_frames) < anomaly_rate

        # Event-driven: only process frames with alarms
        ed_latencies = []
        ed_processed = 0
        t0 = time.perf_counter()
        for i, frame in enumerate(frames):
            if is_alarm[i]:
                lat, _ = self._process_frame(frame)
                ed_latencies.append(lat)
                ed_processed += 1
        ed_elapsed = time.perf_counter() - t0

        # Cyclical: process every frame at a fixed poll interval
        cy_latencies = []
        cy_processed = 0
        t0 = time.perf_counter()
        for frame in frames:
            lat, _ = self._process_frame(frame)
            cy_latencies.append(lat)
            cy_processed += 1
        cy_elapsed = time.perf_counter() - t0

        return {
            "event_driven": {
                "frames_processed": ed_processed,
                "elapsed_s": ed_elapsed,
                "fps": ed_processed / ed_elapsed if ed_elapsed > 0 else 0,
                "mean_latency_ms": (statistics.mean(ed_latencies)
                                    if ed_latencies else 0),
                "p95_latency_ms": (sorted(ed_latencies)[int(0.95 * len(ed_latencies))]
                                   if ed_latencies else 0),
            },
            "cyclical": {
                "frames_processed": cy_processed,
                "elapsed_s": cy_elapsed,
                "fps": cy_processed / cy_elapsed if cy_elapsed > 0 else 0,
                "mean_latency_ms": (statistics.mean(cy_latencies)
                                    if cy_latencies else 0),
                "p95_latency_ms": (sorted(cy_latencies)[int(0.95 * len(cy_latencies))]
                                   if cy_latencies else 0),
            },
        }

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_scaling_table(result: ScalingResult) -> Tuple[str, str]:
        """Return (markdown, latex) tables for the scaling study."""
        md = ("| Streams | Total Frames | FPS | Mean Lat (ms) | P95 Lat (ms) "
              "| Queue Max | Dropped | Scaling Eff. |\n")
        md += ("|---------|-------------|-----|---------------|-------------- "
               "|-----------|---------|--------------|\n")
        for r in result.throughput_by_streams:
            eff = result.scaling_efficiency.get(r.n_streams, 0)
            md += (f"| {r.n_streams} | {r.total_frames} | {r.fps:.1f} | "
                   f"{r.mean_latency_ms:.1f} | {r.p95_latency_ms:.1f} | "
                   f"{r.queue_depth_max} | {r.frames_dropped} | "
                   f"{eff*100:.1f}% |\n")

        latex = r"""\begin{table}[t]
\centering
\caption{Multi-agent throughput scaling with concurrent camera streams}
\label{tab:scaling}
\begin{tabular}{rrrrrrrr}
\toprule
\textbf{Streams} & \textbf{Frames} & \textbf{FPS} & \textbf{Lat (ms)} & \textbf{P95 (ms)} & \textbf{Q Max} & \textbf{Drop} & \textbf{Eff.} \\
\midrule
"""
        for r in result.throughput_by_streams:
            eff = result.scaling_efficiency.get(r.n_streams, 0)
            latex += (f"{r.n_streams} & {r.total_frames} & {r.fps:.1f} & "
                      f"{r.mean_latency_ms:.1f} & {r.p95_latency_ms:.1f} & "
                      f"{r.queue_depth_max} & {r.frames_dropped} & "
                      f"{eff*100:.1f}\\% \\\\\n")
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return md, latex

    @staticmethod
    def generate_agent_table(agent_results: Dict) -> Tuple[str, str]:
        """Return (markdown, latex) for event-driven vs cyclical comparison."""
        ed = agent_results["event_driven"]
        cy = agent_results["cyclical"]

        md = "| Agent Mode | Frames | FPS | Mean Lat (ms) | P95 Lat (ms) |\n"
        md += "|------------|--------|-----|---------------|---------------|\n"
        md += (f"| Event-Driven | {ed['frames_processed']} | {ed['fps']:.1f} | "
               f"{ed['mean_latency_ms']:.1f} | {ed['p95_latency_ms']:.1f} |\n")
        md += (f"| Cyclical | {cy['frames_processed']} | {cy['fps']:.1f} | "
               f"{cy['mean_latency_ms']:.1f} | {cy['p95_latency_ms']:.1f} |\n")

        latex = r"""\begin{table}[t]
\centering
\caption{Event-driven vs.\ cyclical agent performance}
\label{tab:agent_modes}
\begin{tabular}{lrrrr}
\toprule
\textbf{Agent Mode} & \textbf{Frames} & \textbf{FPS} & \textbf{Mean Lat (ms)} & \textbf{P95 (ms)} \\
\midrule
"""
        latex += (f"Event-Driven & {ed['frames_processed']} & {ed['fps']:.1f} & "
                  f"{ed['mean_latency_ms']:.1f} & {ed['p95_latency_ms']:.1f} \\\\\n")
        latex += (f"Cyclical & {cy['frames_processed']} & {cy['fps']:.1f} & "
                  f"{cy['mean_latency_ms']:.1f} & {cy['p95_latency_ms']:.1f} \\\\\n")
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return md, latex

    @staticmethod
    def generate_resource_table(snapshots: List[ResourceSnapshot]) -> str:
        """Summarise resource utilization as a markdown table."""
        if not snapshots:
            return "No resource data collected."

        cpus = [s.cpu_percent for s in snapshots]
        rams = [s.ram_used_mb for s in snapshots]
        gpus = [s.gpu_mem_used_mb for s in snapshots]

        md = "| Metric | Mean | Max | Min |\n"
        md += "|--------|------|-----|-----|\n"
        md += (f"| CPU (%) | {statistics.mean(cpus):.1f} | "
               f"{max(cpus):.1f} | {min(cpus):.1f} |\n")
        md += (f"| RAM (MB) | {statistics.mean(rams):.0f} | "
               f"{max(rams):.0f} | {min(rams):.0f} |\n")
        if any(g > 0 for g in gpus):
            md += (f"| GPU Mem (MB) | {statistics.mean(gpus):.0f} | "
                   f"{max(gpus):.0f} | {min(gpus):.0f} |\n")
        return md


# =========================================================================
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent System Benchmark"
    )
    parser.add_argument("--ae-model", type=str,
                        default="autoencoder_model.pth")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration per stream (seconds)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Target FPS per stream")
    parser.add_argument("--streams", type=str, default="1,2,4,8,16",
                        help="Comma-separated stream counts")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation_results")
    args = parser.parse_args()

    stream_counts = [int(s) for s in args.streams.split(",")]

    print("=" * 65)
    print("  MULTI-AGENT SYSTEM BENCHMARK")
    print("=" * 65)

    bench = MultiAgentBenchmark(ae_model_path=args.ae_model)

    # 1. Scaling study
    print("\n--- Scaling Study ---")
    scaling = bench.run_scaling_study(
        stream_counts=stream_counts,
        stream_fps=args.fps,
        duration_s=args.duration,
    )
    md_scale, latex_scale = MultiAgentBenchmark.generate_scaling_table(scaling)
    print("\n" + md_scale)

    # 2. Resource utilization
    print("--- Resource Utilization ---")
    res_md = MultiAgentBenchmark.generate_resource_table(
        scaling.resource_snapshots
    )
    print(res_md)

    # 3. Agent mode comparison
    print("--- Agent Mode Comparison ---")
    agent_results = bench.benchmark_agent_modes()
    md_agent, latex_agent = MultiAgentBenchmark.generate_agent_table(
        agent_results
    )
    print(md_agent)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "scaling": [asdict(r) for r in scaling.throughput_by_streams],
        "scaling_efficiency": scaling.scaling_efficiency,
        "agent_modes": agent_results,
        "device": scaling.device,
    }
    with open(os.path.join(args.output_dir, "multi_agent_benchmark.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}/multi_agent_benchmark.json")


if __name__ == "__main__":
    main()
