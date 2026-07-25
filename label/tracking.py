from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .storage import find_frame_path, find_optional_prediction_frame_path
except Exception:
    from storage import find_frame_path, find_optional_prediction_frame_path


Point = Tuple[float, float]
HandPoints = List[List[Point]]
HandVisible = List[List[bool]]

_HAND_COUNT = 2
_JOINT_COUNT = 21
_MODEL_REPO = "/home/ubuntu/.cache/torch/hub/facebookresearch_co-tracker_main"


class CoTrackerRuntime:
    def __init__(self) -> None:
        self._torch = None
        self._model = None
        self._device: Optional[str] = None

    def track_camera(
        self,
        *,
        episode_dir: Path,
        cam_id: str,
        prev_frame_idx: int,
        frame_idx: int,
    ) -> Tuple[HandPoints, HandVisible]:
        prev_ann = self._load_previous_annotation(episode_dir, cam_id, prev_frame_idx)
        queries_np, indices = self._queries_from_annotation(prev_ann)
        if not indices:
            return self._hidden_points(), self._none_visible()

        clip = self._load_clip(episode_dir, cam_id, prev_frame_idx, frame_idx)
        torch = self._ensure_torch()
        model = self._ensure_model()

        videos = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().to(self._device)
        queries = torch.from_numpy(queries_np)[None].float().to(self._device)

        with torch.no_grad():
            pred_tracks, pred_visibility = model(videos, queries=queries, backward_tracking=False)

        tracks = pred_tracks.detach().cpu().numpy()[0, -1]
        visibility = pred_visibility.detach().cpu().numpy()[0, -1].astype(bool)
        return self._state_from_tracks(tracks, visibility, indices)

    def _ensure_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Tracking mode requires PyTorch. Install a compatible `torch` package first.") from exc
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        torch = self._ensure_torch()
        try:
            model = torch.hub.load(_MODEL_REPO, "cotracker3_offline", source="local")
        except Exception as exc:
            raise RuntimeError(f"Failed to load CoTracker model from {_MODEL_REPO}.") from exc
        self._model = model.to(self._device)
        self._model.eval()
        return self._model

    def _load_previous_annotation(self, episode_dir: Path, cam_id: str, prev_frame_idx: int) -> np.ndarray:
        path = find_optional_prediction_frame_path(episode_dir / "corrected_2d", cam_id, prev_frame_idx)
        if path is None:
            raise FileNotFoundError(f"Missing previous corrected_2d for camera {cam_id}, frame {prev_frame_idx}.")
        try:
            ann = np.load(path).astype(np.float32)
        except Exception as exc:
            raise ValueError(f"Failed to load previous corrected_2d: {path}") from exc
        if ann.shape != (_HAND_COUNT, _JOINT_COUNT, 2):
            raise ValueError(f"Previous corrected_2d must have shape (2,21,2), got {ann.shape}: {path}")
        return ann

    def _load_clip(self, episode_dir: Path, cam_id: str, prev_frame_idx: int, frame_idx: int) -> np.ndarray:
        frame_indices = self._continuous_frame_indices(episode_dir, cam_id, prev_frame_idx, frame_idx)
        frames = [self._load_rgb_frame(episode_dir, cam_id, idx) for idx in frame_indices]
        try:
            return np.stack(frames, axis=0)
        except ValueError as exc:
            raise ValueError(f"Tracking frames for camera {cam_id} do not share the same image size.") from exc

    def _continuous_frame_indices(
        self,
        episode_dir: Path,
        cam_id: str,
        prev_frame_idx: int,
        frame_idx: int,
    ) -> List[int]:
        if frame_idx > prev_frame_idx:
            candidate = list(range(int(prev_frame_idx), int(frame_idx) + 1))
            if all(find_frame_path(episode_dir, cam_id, idx) is not None for idx in candidate):
                return candidate
        return [int(prev_frame_idx), int(frame_idx)]

    def _load_rgb_frame(self, episode_dir: Path, cam_id: str, frame_idx: int) -> np.ndarray:
        path = find_frame_path(episode_dir, cam_id, frame_idx)
        if path is None:
            raise FileNotFoundError(f"Missing RGB frame for camera {cam_id}, frame {frame_idx}.")
        try:
            with Image.open(path) as im:
                return np.asarray(im.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Failed to load RGB frame: {path}") from exc

    def _queries_from_annotation(self, ann: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        visible = ~np.all(ann == -1, axis=-1)
        queries = []
        indices: List[Tuple[int, int]] = []
        for hand_idx in range(_HAND_COUNT):
            for joint_idx in range(_JOINT_COUNT):
                if visible[hand_idx, joint_idx]:
                    x, y = ann[hand_idx, joint_idx]
                    queries.append([0.0, float(x), float(y)])
                    indices.append((hand_idx, joint_idx))
        return np.asarray(queries, dtype=np.float32), indices

    def _state_from_tracks(
        self,
        tracks: np.ndarray,
        visibility: np.ndarray,
        indices: List[Tuple[int, int]],
    ) -> Tuple[HandPoints, HandVisible]:
        points = self._hidden_points()
        visible = self._none_visible()
        for n, (hand_idx, joint_idx) in enumerate(indices):
            x, y = tracks[n]
            points[hand_idx][joint_idx] = (float(x), float(y))
            visible[hand_idx][joint_idx] = bool(visibility[n])
        return points, visible

    @staticmethod
    def _hidden_points() -> HandPoints:
        return [[(-1.0, -1.0) for _ in range(_JOINT_COUNT)] for _ in range(_HAND_COUNT)]

    @staticmethod
    def _none_visible() -> HandVisible:
        return [[False for _ in range(_JOINT_COUNT)] for _ in range(_HAND_COUNT)]
