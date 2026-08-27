from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from optimizer.mano_wrapper import build_mano_aa, mano_forward
from optimizer.rotation import pose_rot6d_to_axis_angle


SMPLX_MANO_PARENTS = np.asarray((-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9), dtype=np.int64)


def build_mano_layers(mano_dir: str | Path) -> dict[int, torch.nn.Module]:
    """Build local MANO layers from opt_toolkits/ckpt/mano; keys are 0=left, 1=right."""
    mano_path = Path(mano_dir)
    return {
        0: build_mano_aa(str(mano_path), is_rhand=False),
        1: build_mano_aa(str(mano_path), is_rhand=True),
    }


def mano_bone_lengths(layers: dict[int, torch.nn.Module], shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Compute shaped and scaled smplx MANO parent-child bone lengths."""
    shape = np.asarray(shape, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if shape.ndim != 2 or shape.shape[1] != 10:
        raise ValueError(f"shape must have shape [B,10], got {shape.shape}")
    if scale.shape != (shape.shape[0], 1):
        raise ValueError(f"scale must have shape [{shape.shape[0]},1], got {scale.shape}")
    if np.any(scale <= 0.0):
        raise ValueError("scale must be positive when computing MANO bone lengths")

    joints_by_hand = []
    for hand in (0, 1):
        if hand not in layers:
            raise KeyError(f"MANO layer is missing hand index {hand}")
        layer = layers[hand]
        device = layer.shapedirs.device
        dtype = layer.shapedirs.dtype
        shape_tensor = torch.as_tensor(shape, device=device, dtype=dtype)
        zeros3 = torch.zeros((shape.shape[0], 3), device=device, dtype=dtype)
        zeros45 = torch.zeros((shape.shape[0], 45), device=device, dtype=dtype)
        with torch.no_grad():
            output = layer(global_orient=zeros3, hand_pose=zeros45, betas=shape_tensor, transl=None)
        if output.joints is None or output.joints.shape[1:] != (21, 3):
            raise ValueError(f"MANO layer must return [B,21,3] joints, got {None if output.joints is None else tuple(output.joints.shape)}")
        joints = output.joints * torch.as_tensor(scale, device=device, dtype=dtype)[:, None]
        joints_by_hand.append(joints.detach().cpu().numpy().astype(np.float32, copy=False))

    joints = np.stack(joints_by_hand, axis=1)
    parent_indices = SMPLX_MANO_PARENTS.copy()
    parent_indices[0] = 0
    lengths = np.linalg.norm(joints - joints[:, :, parent_indices], axis=-1).astype(np.float32, copy=False)
    lengths[:, :, 0] = 0.0
    return lengths


def mano_faces(layer: torch.nn.Module) -> np.ndarray:
    """Return MANO triangle indices from an smplx MANO layer."""
    if hasattr(layer, "faces"):
        return np.asarray(layer.faces, dtype=np.int32)
    if hasattr(layer, "faces_tensor"):
        return layer.faces_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
    raise AttributeError("MANO layer does not expose faces or faces_tensor")


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous 4x4 transform to [..., 3] points."""
    transform = transform.to(device=points.device, dtype=points.dtype)
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=-1)
    return torch.matmul(points_h, transform.transpose(-1, -2))[..., :3]


def project_points(points_camera: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space [N, 3] points to image uv and return positive-depth mask."""
    intrinsics = intrinsics.to(device=points_camera.device, dtype=points_camera.dtype)
    z = points_camera[:, 2]
    valid = z > 1e-6
    uv = torch.full((points_camera.shape[0], 2), -1.0, dtype=points_camera.dtype, device=points_camera.device)
    uv[valid, 0] = points_camera[valid, 0] / z[valid] * intrinsics[0, 0] + intrinsics[0, 2]
    uv[valid, 1] = points_camera[valid, 1] / z[valid] * intrinsics[1, 1] + intrinsics[1, 2]
    return uv, valid


def load_pose(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"pose file not found: {path}")
    pose = np.load(path)
    if pose.ndim == 1 and pose.size >= 198:
        pose = pose[:198].reshape(2, 99)
    if pose.ndim != 2 or pose.shape[0] != 2 or pose.shape[1] < 99:
        raise ValueError(f"pose must have shape [2, >=99], got {pose.shape}: {path}")
    return pose[:, :99].astype(np.float32, copy=False)


def mano_outputs_from_pose(
    pose: np.ndarray,
    betas: np.ndarray,
    scales: np.ndarray,
    layers: dict[int, torch.nn.Module],
) -> dict[int, dict[str, torch.Tensor]]:
    """Forward both hands from saved pose rows [rot6d*16, xyz] using local MANO layers."""
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    pose_tensor = torch.from_numpy(pose.astype(np.float32, copy=False)).unsqueeze(0)
    hand_pose_aa, global_orient_aa, transl = pose_rot6d_to_axis_angle(pose_tensor)
    for hand_side in (0, 1):
        beta = torch.from_numpy(betas[hand_side]).view(1, -1).float()
        layer = layers[hand_side]
        scale = torch.as_tensor(scales[hand_side], dtype=hand_pose_aa.dtype).reshape(1, 1)
        with torch.no_grad():
            vertices, joints = mano_forward(
                layer,
                hand_pose_aa[:, hand_side],
                beta,
                global_orient_aa[:, hand_side],
                transl[:, hand_side],
                scale,
            )
        outputs[hand_side] = {
            "vertices": vertices,
            "joints": joints,
        }
    return outputs
