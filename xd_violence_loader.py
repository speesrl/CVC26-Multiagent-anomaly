"""
XD-Violence Dataset Loader for Anomaly Detection Evaluation

XD-Violence is a large-scale dataset containing 4754 untrimmed videos with
both audio and visual modalities, covering six anomaly categories:
Abuse, Car Accident, Explosion, Fighting, Riot, Shooting.

Reference:
    Wu, P., Liu, J., Shi, Y., Sun, Y., Shao, F., Wu, Z. and Yang, Z.,
    2020.  Not only look, but also listen: Learning multimodal violence
    detection under weak supervision.  ECCV 2020.

Expected layout:
    xd_violence/
    ├── videos/             # .mp4 video files
    │   ├── A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4
    │   └── ...
    ├── annotations/
    │   ├── test_list.txt   # test split video list
    │   └── test_annotations.txt  # temporal segment labels
    └── rgb_features/       # optional pre-extracted features (I3D)

Addresses Reviewer 5A: multi-dataset evaluation.
"""

import os
import re
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass


ANOMALY_CATEGORIES = [
    "Abuse", "Car accident", "Explosion", "Fighting", "Riot", "Shooting"
]


@dataclass
class XDVideoAnnotation:
    video_name: str
    label: int  # 1 = anomaly video, 0 = normal
    category: str
    segments: List[Tuple[float, float]]  # (start_sec, end_sec)


class XDViolenceDataset:
    """
    Loader for the XD-Violence dataset.

    The test split typically contains ~800 videos.  Annotations provide
    video-level labels; temporal segments are available for anomalous
    videos only.
    """

    def __init__(
        self,
        root_dir: str,
        test_list: Optional[str] = None,
        annotation_file: Optional[str] = None,
    ):
        self.root = Path(root_dir)
        self.videos_dir = self.root / "videos"

        self._test_list: List[str] = []
        self._annotations: Dict[str, XDVideoAnnotation] = {}

        tl = test_list or str(self.root / "annotations" / "test_list.txt")
        af = annotation_file or str(self.root / "annotations" / "test_annotations.txt")

        self._load_test_list(tl)
        self._load_annotations(af)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _load_test_list(self, path: str):
        p = Path(path)
        if not p.exists():
            if self.videos_dir.exists():
                self._test_list = sorted(
                    f.name for f in self.videos_dir.glob("*.mp4")
                )
            return
        with open(p) as f:
            self._test_list = [
                line.strip() for line in f if line.strip()
            ]

    def _load_annotations(self, path: str):
        p = Path(path)
        if not p.exists():
            return

        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                video_name = parts[0]
                label = int(parts[1])

                segments = []
                category = "Normal"
                if label == 1 and len(parts) >= 4:
                    category = self._extract_category(video_name)
                    i = 2
                    while i + 1 < len(parts):
                        try:
                            s, e = float(parts[i]), float(parts[i + 1])
                            segments.append((s, e))
                            i += 2
                        except ValueError:
                            break

                self._annotations[video_name] = XDVideoAnnotation(
                    video_name=video_name,
                    label=label,
                    category=category,
                    segments=segments,
                )

    @staticmethod
    def _extract_category(video_name: str) -> str:
        lower = video_name.lower()
        for cat in ANOMALY_CATEGORIES:
            if cat.lower().replace(" ", "") in lower.replace(" ", ""):
                return cat
        if "label_a" in lower:
            return "Anomaly"
        return "Unknown"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def test_videos(self) -> List[str]:
        return list(self._test_list)

    def __len__(self) -> int:
        return len(self._test_list)

    def get_video_path(self, video_name: str) -> Optional[Path]:
        p = self.videos_dir / video_name
        if p.exists():
            return p
        for ext in [".mp4", ".avi"]:
            p2 = self.videos_dir / (Path(video_name).stem + ext)
            if p2.exists():
                return p2
        return None

    def get_frame_labels(
        self, video_name: str, total_frames: int, fps: float = 30.0
    ) -> np.ndarray:
        """
        Build per-frame binary labels from temporal segment annotations.
        """
        labels = np.zeros(total_frames, dtype=np.int32)
        ann = self._annotations.get(video_name)
        if ann is None or ann.label == 0:
            return labels

        for start_sec, end_sec in ann.segments:
            s_frame = max(0, int(start_sec * fps))
            e_frame = min(total_frames, int(end_sec * fps))
            labels[s_frame:e_frame] = 1

        return labels

    def extract_frames(
        self,
        video_name: str,
        resize: Tuple[int, int] = (128, 128),
        sample_rate: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        vpath = self.get_video_path(video_name)
        if vpath is None:
            raise FileNotFoundError(f"Video not found: {video_name}")

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_labels = self.get_frame_labels(video_name, total, fps)

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
                labels.append(all_labels[idx] if idx < len(all_labels) else 0)
            idx += 1

        cap.release()
        return np.array(frames), np.array(labels, dtype=np.int32)

    def iterate_test_set(
        self,
        resize: Tuple[int, int] = (128, 128),
        sample_rate: int = 1,
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        for vname in self._test_list:
            try:
                frames, labels = self.extract_frames(
                    vname, resize, sample_rate
                )
                if len(frames) > 0:
                    yield vname, frames, labels
            except Exception as e:
                print(f"Error loading {vname}: {e}")


def load_xd_violence(root_dir: str, **kwargs) -> XDViolenceDataset:
    return XDViolenceDataset(root_dir, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XD-Violence Loader")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    ds = XDViolenceDataset(args.root)
    print(f"Test videos: {len(ds)}")
    for v in ds.test_videos[:5]:
        ann = ds._annotations.get(v)
        if ann:
            print(f"  {v}: label={ann.label}, cat={ann.category}, "
                  f"segments={ann.segments}")
