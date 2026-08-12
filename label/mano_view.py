from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


Point = Tuple[float, float]
HandPoints = List[List[Point]]
HandVisible = List[List[bool]]
MeshLine = Tuple[Point, Point, str]

_HAND_COUNT = 2
_JOINT_COUNT = 21
_MANO_MODEL_DIR = "/home/ubuntu/orbbec/mano"
_HAND_COLORS = ("#37c7ff", "#ff8a3d")
_SMPLX_MANO_TO_APP_ORDER = (
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
)


@dataclass(frozen=True)
class CameraParams:
    k: np.ndarray
    dist: np.ndarray
    r: np.ndarray
    t: np.ndarray

    @property
    def rt(self) -> np.ndarray:
        return np.concatenate([self.r, self.t.reshape(3, 1)], axis=1)


@dataclass
class ManoMeshResult:
    vertices: List[np.ndarray]
    faces: List[np.ndarray]


class ManoViewRuntime:
    def __init__(self) -> None:
        self._torch = None
        self._smplx = None
        self._models: Dict[int, object] = {}
        self._joints_cache: Dict[Tuple[str, str, int], Optional[np.ndarray]] = {}

    def project_mano_frame(
        self,
        *,
        episode_dir: Path,
        mano_dir: Path,
        cam_id: str,
        frame_idx: int,
    ) -> Optional[Tuple[HandPoints, HandVisible]]:
        joints_3d = self._load_mano_frame_joints(mano_dir, int(frame_idx))
        if joints_3d is None:
            return None
        return self.project_skeleton(episode_dir=episode_dir, cam_id=cam_id, joints_3d=joints_3d)

    def has_mano_frame(
        self,
        *,
        mano_dir: Path,
        frame_idx: int,
    ) -> bool:
        return self._load_mano_frame_joints(mano_dir, int(frame_idx)) is not None

    def _load_mano_frame_joints(self, mano_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
        meta_path = _mano_metadata_path(mano_dir)
        if meta_path is None:
            return None
        cache_key = (str(meta_path.resolve()), _mano_input_signature(mano_dir, meta_path), int(frame_idx))
        if cache_key in self._joints_cache:
            cached = self._joints_cache[cache_key]
            return None if cached is None else cached.copy()
        try:
            joints = load_mano_frame_joints(mano_dir, int(frame_idx), meta_path=meta_path)
        except Exception:
            joints = None
        self._joints_cache[cache_key] = None if joints is None else joints.copy()
        return joints

    def build_mesh(
        self,
        *,
        episode_dir: Path,
        camera_ids: List[str],
        view_states: Dict[str, Tuple[HandPoints, HandVisible]],
    ) -> ManoMeshResult:
        cameras = load_episode_cameras(episode_dir, camera_ids)
        joints_3d = triangulate_hands(cameras, view_states)
        observations = collect_normalized_observations(cameras, view_states)
        torch = self._ensure_torch()
        vertices = [
            self._fit_hand(
                target_joints=torch.as_tensor(joints_3d[hand], dtype=torch.float32),
                hand=hand,
                cameras=cameras,
                observations=observations[hand],
            )
            for hand in range(_HAND_COUNT)
        ]
        faces = [
            np.asarray(getattr(self._model_for_hand(hand), "faces"), dtype=np.int32)
            for hand in range(_HAND_COUNT)
        ]
        return ManoMeshResult(vertices=vertices, faces=faces)

    def build_skeleton(
        self,
        *,
        episode_dir: Path,
        camera_ids: List[str],
        view_states: Dict[str, Tuple[HandPoints, HandVisible]],
    ) -> np.ndarray:
        cameras = load_episode_cameras(episode_dir, camera_ids)
        return triangulate_hands(cameras, view_states)

    def project_skeleton(
        self,
        *,
        episode_dir: Path,
        cam_id: str,
        joints_3d: np.ndarray,
    ) -> Tuple[HandPoints, HandVisible]:
        cameras = load_episode_cameras(episode_dir, [cam_id])
        cam = cameras[cam_id]
        points: HandPoints = []
        visible: HandVisible = []
        for hand in range(_HAND_COUNT):
            hand_joints = np.asarray(joints_3d[hand], dtype=np.float32)
            finite_3d = np.all(np.isfinite(hand_joints), axis=1)
            pts_2d = np.full((_JOINT_COUNT, 2), np.nan, dtype=np.float32)
            if np.any(finite_3d):
                pts_2d[finite_3d] = project_points(cam, hand_joints[finite_3d]).astype(np.float32)
            points.append([(float(x), float(y)) for x, y in pts_2d])
            visible.append([bool(finite_3d[idx] and np.all(np.isfinite(pt))) for idx, pt in enumerate(pts_2d)])
        return points, visible

    def project_mesh(
        self,
        *,
        episode_dir: Path,
        cam_id: str,
        mesh: ManoMeshResult,
    ) -> List[MeshLine]:
        cameras = load_episode_cameras(episode_dir, [cam_id])
        cam = cameras[cam_id]
        out: List[MeshLine] = []
        for hand, vertices in enumerate(mesh.vertices):
            edges = mesh_edges(mesh.faces[hand])
            pts_2d = project_points(cam, vertices)
            color = _HAND_COLORS[hand]
            for a, b in edges:
                if a >= len(pts_2d) or b >= len(pts_2d):
                    continue
                pa = pts_2d[a]
                pb = pts_2d[b]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    out.append(((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1])), color))
        return out

    def _ensure_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Show MANO requires PyTorch.") from exc
        self._torch = torch
        return torch

    def _ensure_smplx(self):
        if self._smplx is not None:
            return self._smplx
        _ensure_mano_pickle_compat()
        try:
            import smplx
        except Exception as exc:
            raise RuntimeError("Show MANO requires `smplx`. Install it with `pip install smplx`.") from exc
        self._smplx = smplx
        return smplx

    def _model_for_hand(self, hand: int):
        if hand in self._models:
            return self._models[hand]
        smplx = self._ensure_smplx()
        errors = []
        for model_path in _mano_model_candidates(hand):
            try:
                model = smplx.create(
                    str(model_path),
                    model_type="mano",
                    is_rhand=bool(hand == 1),
                    use_pca=False,
                    batch_size=1,
                )
                break
            except Exception as exc:
                errors.append(f"{model_path}: {type(exc).__name__}: {exc}")
        else:
            details = "\n".join(errors) if errors else "No MANO model path candidates exist."
            raise RuntimeError(f"Failed to load MANO model for hand {hand}.\n{details}")
        model.eval()
        try:
            model.requires_grad_(False)
        except Exception:
            pass
        self._models[hand] = model
        return model

    def _fit_hand(
        self,
        *,
        target_joints,
        hand: int,
        cameras: Dict[str, CameraParams],
        observations: List[Tuple[str, int, np.ndarray]],
    ) -> np.ndarray:
        torch = self._ensure_torch()
        model = self._model_for_hand(hand)
        global_orient = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        hand_pose = torch.zeros((1, 45), dtype=torch.float32, requires_grad=True)
        betas = torch.zeros((1, 10), dtype=torch.float32, requires_grad=True)
        transl = target_joints[:1].detach().clone().requires_grad_(True)
        target = target_joints[None]
        camera_tensors = _camera_tensors(torch, cameras)
        obs_tensors = _observation_tensors(torch, observations)

        stages = (
            ([global_orient, transl], 250, 0.02, 0.0),
            ([global_orient, hand_pose, transl], 600, 0.01, 0.0),
            ([global_orient, hand_pose, betas, transl], 500, 0.003, 1.0),
        )
        for params, steps, lr, beta_weight in stages:
            optimizer = torch.optim.Adam(params, lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
                joints = self._mano_joints_21(out)
                loss = self._fit_loss(joints, target, hand_pose, betas, camera_tensors, obs_tensors, beta_weight)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            out = model(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl)
        return out.vertices.detach().cpu().numpy()[0]

    def _fit_loss(
        self,
        joints,
        target,
        hand_pose,
        betas,
        camera_tensors,
        obs_tensors,
        beta_weight: float,
    ):
        torch = self._ensure_torch()
        loss_3d = torch.nn.functional.smooth_l1_loss(joints, target, beta=0.01)
        loss_2d = _normalized_reprojection_loss(torch, joints[0], camera_tensors, obs_tensors)
        pose_prior = (hand_pose ** 2).mean()
        beta_prior = (betas ** 2).mean()
        return loss_3d + 0.5 * loss_2d + 1e-3 * pose_prior + float(beta_weight) * 1e-2 * beta_prior

    def _mano_joints_21(self, out):
        joints = out.joints
        if joints.shape[1] < _JOINT_COUNT:
            raise RuntimeError(
                f"MANO model must output at least 21 joints, got {int(joints.shape[1])}. "
                "Please update the local smplx MANO implementation to return 21 joints."
            )
        return joints[:, list(_SMPLX_MANO_TO_APP_ORDER)]


def _ensure_mano_pickle_compat() -> None:
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _mano_model_candidates(hand: int) -> List[Path]:
    filename = "MANO_RIGHT.pkl" if hand == 1 else "MANO_LEFT.pkl"
    base = Path(_MANO_MODEL_DIR)
    candidates = [base, base / "mano", base / filename, base / "mano" / filename]
    return [path for path in candidates if path.exists()]


def _mano_metadata_path(mano_dir: Path) -> Optional[Path]:
    for name in ("mano_episode.json", "mano_patch.json", "joints_3d.json"):
        path = mano_dir / name
        if path.exists() and path.is_file():
            return path
    return None


def _file_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return f"{path.resolve()}:missing"
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def _mano_input_signature(mano_dir: Path, meta_path: Path) -> str:
    parts = [_file_signature(meta_path)]
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        parts.append(f"metadata_error:{type(exc).__name__}")
        return "|".join(parts)
    if isinstance(meta, Mapping):
        raw_file = str(meta.get("joints_3d_file") or meta.get("joints_file") or "").strip()
        if raw_file:
            npy_path = Path(raw_file)
            if not npy_path.is_absolute():
                npy_path = mano_dir / npy_path
            parts.append(_file_signature(npy_path))
    return "|".join(parts)


def describe_mano_projection_issue(episode_dir: Path, mano_dir: Path, cam_id: str, frame_idx: int) -> str:
    meta_path = _mano_metadata_path(mano_dir)
    if meta_path is None:
        return f"MANO 3D metadata not found at {mano_dir}"
    try:
        joints = load_mano_frame_joints(mano_dir, int(frame_idx), meta_path=meta_path)
    except Exception as exc:
        return f"Failed to load MANO 3D from {mano_dir}: {exc}"
    if joints is None:
        return f"No MANO 3D joints for frame {int(frame_idx)} in {mano_dir}"
    try:
        load_episode_cameras(episode_dir, [str(cam_id)])
    except Exception as exc:
        return str(exc)
    return "No finite MANO projection for this camera/frame."


def episode_calibration_paths(episode_dir: Path) -> Tuple[Path, Path]:
    root = Path(episode_dir).expanduser().resolve()
    return root / "camera_params.json", root / "extrinsics.json"


def require_episode_calibration(episode_dir: Path) -> Tuple[Path, Path]:
    cam_path, ext_path = episode_calibration_paths(episode_dir)
    if not cam_path.is_file() or not ext_path.is_file():
        raise FileNotFoundError(
            "Collection calibration missing at episode root "
            f"{Path(episode_dir).expanduser().resolve()}: expected {cam_path} and {ext_path}"
        )
    return cam_path, ext_path


def require_mano_3d_artifact(mano_dir: Path) -> Path:
    root = Path(mano_dir).expanduser().resolve()
    meta_path = _mano_metadata_path(root)
    if meta_path is None:
        raise FileNotFoundError(
            f"MANO 3D output missing at {root}: expected mano_episode.json, mano_patch.json, or joints_3d.json"
        )
    return meta_path


def require_mano_episode_artifact(mano_dir: Path) -> Path:
    root = Path(mano_dir).expanduser().resolve()
    meta_path = root / "mano_episode.json"
    joints_path = root / "joints_3d.npy"
    if not meta_path.is_file() or not joints_path.is_file():
        raise FileNotFoundError(
            f"MANO episode output missing at {root}: expected {meta_path} and {joints_path}"
        )
    return meta_path


def load_mano_frame_joints(mano_dir: Path, frame_idx: int, *, meta_path: Optional[Path] = None) -> Optional[np.ndarray]:
    meta_path = meta_path or _mano_metadata_path(mano_dir)
    if meta_path is None:
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, Mapping):
        return None
    joints = _load_mano_frame_joints_from_npy(mano_dir, meta, int(frame_idx))
    if joints is None:
        joints = _load_mano_frame_joints_from_json(meta, int(frame_idx))
    if joints is None:
        return None
    arr = np.asarray(joints, dtype=np.float32)
    if arr.shape != (_HAND_COUNT, _JOINT_COUNT, 3):
        return None
    return arr


def _load_mano_frame_joints_from_npy(mano_dir: Path, meta: Mapping[str, Any], frame_idx: int) -> Optional[np.ndarray]:
    raw_file = str(meta.get("joints_3d_file") or meta.get("joints_file") or "joints_3d.npy").strip()
    if not raw_file:
        return None
    npy_path = Path(raw_file)
    if not npy_path.is_absolute():
        npy_path = mano_dir / npy_path
    if not npy_path.exists() or not npy_path.is_file():
        return None
    frames = _frames_from_mano_meta(meta)
    try:
        arr = np.load(npy_path)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == (_HAND_COUNT, _JOINT_COUNT, 3):
        return arr if not frames or int(frames[0]) == int(frame_idx) else None
    if arr.ndim != 4 or arr.shape[1:] != (_HAND_COUNT, _JOINT_COUNT, 3):
        return None
    if frames:
        try:
            pos = frames.index(int(frame_idx))
        except ValueError:
            return None
    else:
        pos = int(frame_idx)
    if pos < 0 or pos >= arr.shape[0]:
        return None
    return arr[pos]


def _load_mano_frame_joints_from_json(meta: Mapping[str, Any], frame_idx: int) -> Optional[Any]:
    joints_obj = meta.get("joints_3d")
    if isinstance(joints_obj, Mapping):
        for key in (f"{int(frame_idx):05d}", str(int(frame_idx))):
            if key in joints_obj:
                return joints_obj[key]
        return None
    if isinstance(joints_obj, list):
        frames = _frames_from_mano_meta(meta)
        if frames:
            try:
                return joints_obj[frames.index(int(frame_idx))]
            except (ValueError, IndexError):
                return None
        if 0 <= int(frame_idx) < len(joints_obj):
            return joints_obj[int(frame_idx)]
    return None


def _frames_from_mano_meta(meta: Mapping[str, Any]) -> List[int]:
    value = meta.get("frames")
    if not isinstance(value, list):
        return []
    frames: List[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            frames.append(int(item))
        except (TypeError, ValueError):
            pass
    return frames


def load_episode_cameras(episode_dir: Path, camera_ids: List[str]) -> Dict[str, CameraParams]:
    cam_path, ext_path = require_episode_calibration(episode_dir)

    with cam_path.open("r", encoding="utf-8") as f:
        cam_obj = json.load(f)
    with ext_path.open("r", encoding="utf-8") as f:
        ext_obj = json.load(f)

    out: Dict[str, CameraParams] = {}
    for cam_id in camera_ids:
        if cam_id not in cam_obj or cam_id not in ext_obj:
            raise KeyError(f"Missing camera parameters for camera {cam_id}.")
        intr = cam_obj[cam_id]["RGB"]["intrinsic"]
        dist = cam_obj[cam_id]["RGB"].get("distortion", {})
        k = np.asarray(
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.asarray(
            [
                float(dist.get("k1", 0.0)),
                float(dist.get("k2", 0.0)),
                float(dist.get("p1", 0.0)),
                float(dist.get("p2", 0.0)),
                float(dist.get("k3", 0.0)),
            ],
            dtype=np.float64,
        )
        r = np.asarray(ext_obj[cam_id]["rotation"], dtype=np.float64)
        t = np.asarray(ext_obj[cam_id]["translation"], dtype=np.float64)
        out[cam_id] = CameraParams(k=k, dist=dist_coeffs, r=r, t=t)
    return out


def triangulate_hands(
    cameras: Dict[str, CameraParams],
    view_states: Dict[str, Tuple[HandPoints, HandVisible]],
) -> np.ndarray:
    import cv2

    out = np.zeros((_HAND_COUNT, _JOINT_COUNT, 3), dtype=np.float32)
    missing = 0
    for hand in range(_HAND_COUNT):
        for joint in range(_JOINT_COUNT):
            observations = []
            for cam_id, cam in cameras.items():
                state = view_states.get(cam_id)
                if state is None:
                    continue
                points, visible = state
                if not visible[hand][joint]:
                    continue
                pt = np.asarray(points[hand][joint], dtype=np.float64)
                if not np.all(np.isfinite(pt)):
                    continue
                undist = cv2.undistortPoints(pt.reshape(1, 1, 2), cam.k, cam.dist).reshape(2)
                observations.append((cam.rt, undist))
            if len(observations) < 2:
                missing += 1
                continue
            point = _triangulate_joint(observations)
            out[hand, joint] = point.astype(np.float32)
    if missing:
        raise ValueError(f"还有 {missing} 个关节点的可见视角数少于 2，无法计算完整 3D。")
    return out


def collect_normalized_observations(
    cameras: Dict[str, CameraParams],
    view_states: Dict[str, Tuple[HandPoints, HandVisible]],
) -> List[List[Tuple[str, int, np.ndarray]]]:
    import cv2

    observations: List[List[Tuple[str, int, np.ndarray]]] = [[] for _ in range(_HAND_COUNT)]
    for cam_id, cam in cameras.items():
        state = view_states.get(cam_id)
        if state is None:
            continue
        points, visible = state
        for hand in range(_HAND_COUNT):
            for joint in range(_JOINT_COUNT):
                if not visible[hand][joint]:
                    continue
                pt = np.asarray(points[hand][joint], dtype=np.float64)
                if not np.all(np.isfinite(pt)):
                    continue
                undist = cv2.undistortPoints(pt.reshape(1, 1, 2), cam.k, cam.dist).reshape(2)
                observations[hand].append((cam_id, joint, undist.astype(np.float32)))
    return observations


def _triangulate_joint(observations) -> np.ndarray:
    import cv2

    p0, x0 = observations[0]
    points_4d = []
    for p1, x1 in observations[1:]:
        pt4 = cv2.triangulatePoints(p0, p1, x0.reshape(2, 1), x1.reshape(2, 1)).reshape(4)
        if abs(float(pt4[3])) > 1e-8:
            points_4d.append(pt4[:3] / pt4[3])
    if not points_4d:
        raise ValueError("Triangulation failed for a joint.")
    return np.mean(np.asarray(points_4d, dtype=np.float64), axis=0)


def _camera_tensors(torch, cameras: Dict[str, CameraParams]):
    return {
        cam_id: (
            torch.as_tensor(cam.r, dtype=torch.float32),
            torch.as_tensor(cam.t, dtype=torch.float32),
        )
        for cam_id, cam in cameras.items()
    }


def _observation_tensors(torch, observations: List[Tuple[str, int, np.ndarray]]):
    return [
        (cam_id, int(joint), torch.as_tensor(xy, dtype=torch.float32))
        for cam_id, joint, xy in observations
    ]


def _normalized_reprojection_loss(torch, joints, camera_tensors, obs_tensors):
    if not obs_tensors:
        return joints.sum() * 0.0
    losses = []
    for cam_id, joint, target_xy in obs_tensors:
        r, t = camera_tensors[cam_id]
        point_cam = r.matmul(joints[joint]) + t
        z = point_cam[2]
        z = torch.where(torch.abs(z) < 1e-4, torch.full_like(z, 1e-4), z)
        pred_xy = point_cam[:2] / z
        losses.append(torch.nn.functional.smooth_l1_loss(pred_xy, target_xy, beta=0.01))
    return torch.stack(losses).mean()


def project_points(cam: CameraParams, points_3d: np.ndarray) -> np.ndarray:
    import cv2

    rvec, _ = cv2.Rodrigues(cam.r)
    pts, _ = cv2.projectPoints(points_3d.astype(np.float64), rvec, cam.t.reshape(3, 1), cam.k, cam.dist)
    return pts.reshape(-1, 2)


def mesh_edges(faces: np.ndarray) -> List[Tuple[int, int]]:
    edges = set()
    for tri in np.asarray(faces, dtype=np.int32):
        a, b, c = [int(x) for x in tri[:3]]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return sorted(edges)
