"""
ShanghaiTech Campus Dataset Loader for Anomaly Detection Evaluation

ShanghaiTech Campus is a medium-scale benchmark with 437 videos across
13 scenes, containing 130 anomalous events in 107 videos.  Ground truth
is provided as pixel-level masks per frame.

Reference:
    Liu, W., Luo, W., Lian, D. and Gao, S., 2018.  Future frame prediction
    for anomaly detection -- A new baseline.  CVPR 2018.

Dataset layout expected:
    shanghaitech/
    ├── training/
    │   └── videos/          # normal training videos
    ├── testing/
    │   ├── frames/          # pre-extracted frames per video
    │   │   ├── 01_0014/
    │   │   │   ├── 000.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── test_frame_mask/ # ground-truth masks (0/255)
    │       ├── 01_0014.npy
    │       └── ...
    └── test_videos/         # optional raw .avi files

Addresses Reviewer 5A: multi-dataset evaluation.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass
class SHTVideoInfo:
    video_id: str
    scene_id: str
    n_frames: int
    has_anomaly: bool


class ShanghaiTechDataset:
    """
    Loader for the ShanghaiTech Campus anomaly detection dataset.

    Supports two modes:
      1. Pre-extracted frames  (testing/frames/<video_id>/*.jpg)
      2. Raw video files       (test_videos/*.avi)
    """

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.frames_dir = self.root / "testing" / "frames"
        self.masks_dir = self.root / "testing" / "test_frame_mask"
        self.videos_dir = self.root / "test_videos"

        self._video_ids: List[str] = []
        self._labels_cache: Dict[str, np.ndarray] = {}
        self._discover_videos()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_videos(self):
        if self.frames_dir.exists():
            self._video_ids = sorted(
                d.name for d in self.frames_dir.iterdir() if d.is_dir()
            )
        elif self.videos_dir.exists():
            self._video_ids = sorted(
                p.stem for p in self.videos_dir.glob("*.avi")
            )
        else:
            print(f"Warning: no frames or videos found under {self.root}")

    @property
    def video_ids(self) -> List[str]:
        return list(self._video_ids)

    def __len__(self) -> int:
        return len(self._video_ids)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def get_frame_labels(self, video_id: str) -> Optional[np.ndarray]:
        """
        Return per-frame binary labels (1 = anomaly).

        Labels are derived from the `.npy` mask files shipped with
        ShanghaiTech.  Each file contains an array of shape (N,) where
        1 indicates an anomalous frame.
        """
        if video_id in self._labels_cache:
            return self._labels_cache[video_id]

        npy_path = self.masks_dir / f"{video_id}.npy"
        if npy_path.exists():
            labels = np.load(str(npy_path))
            if labels.ndim > 1:
                labels = labels.max(axis=tuple(range(1, labels.ndim)))
            labels = (labels > 0).astype(np.int32)
            self._labels_cache[video_id] = labels
            return labels

        return None

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        video_id: str,
        resize: Tuple[int, int] = (128, 128),
        sample_rate: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (frames [N, H, W, 3], labels [N]) for a test video.
        """
        if self.frames_dir.exists():
            return self._load_from_frames_dir(video_id, resize, sample_rate)
        return self._load_from_video(video_id, resize, sample_rate)

    def _load_from_frames_dir(
        self, video_id, resize, sample_rate
    ) -> Tuple[np.ndarray, np.ndarray]:
        frame_dir = self.frames_dir / video_id
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame dir not found: {frame_dir}")

        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        labels_full = self.get_frame_labels(video_id)

        frames, labels = [], []
        for idx, p in enumerate(paths):
            if idx % sample_rate != 0:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, resize)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            if labels_full is not None and idx < len(labels_full):
                labels.append(labels_full[idx])
            else:
                labels.append(0)

        return np.array(frames), np.array(labels, dtype=np.int32)

    def _load_from_video(
        self, video_id, resize, sample_rate
    ) -> Tuple[np.ndarray, np.ndarray]:
        video_path = self.videos_dir / f"{video_id}.avi"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        labels_full = self.get_frame_labels(video_id)
        cap = cv2.VideoCapture(str(video_path))

        frames, labels = [], []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                frame = cv2.resize(frame, resize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if labels_full is not None and idx < len(labels_full):
                    labels.append(labels_full[idx])
                else:
                    labels.append(0)
            idx += 1

        cap.release()
        return np.array(frames), np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iterate_test_set(
        self,
        resize: Tuple[int, int] = (128, 128),
        sample_rate: int = 1,
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Yield (video_id, frames, labels) for every test video."""
        for vid in self._video_ids:
            try:
                frames, labels = self.extract_frames(vid, resize, sample_rate)
                if len(frames) > 0:
                    yield vid, frames, labels
            except Exception as e:
                print(f"Error loading {vid}: {e}")

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def get_video_info(self) -> List[SHTVideoInfo]:
        infos = []
        for vid in self._video_ids:
            labels = self.get_frame_labels(vid)
            n_frames = len(labels) if labels is not None else 0
            has_anomaly = bool(labels is not None and labels.sum() > 0)
            scene = vid.split("_")[0] if "_" in vid else "unknown"
            infos.append(SHTVideoInfo(vid, scene, n_frames, has_anomaly))
        return infos


def load_shanghaitech(root_dir: str) -> ShanghaiTechDataset:
    return ShanghaiTechDataset(root_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ShanghaiTech Loader")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    ds = ShanghaiTechDataset(args.root)
    print(f"Found {len(ds)} test videos")
    for info in ds.get_video_info()[:5]:
        print(f"  {info.video_id}: scene={info.scene_id}, "
              f"frames={info.n_frames}, anomaly={info.has_anomaly}")
