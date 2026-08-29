from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np


HAND_COLORS_RGB = {0: (0.85, 0.45, 0.45), 1: (0.65, 0.74, 0.86)}


def _emit(**fields: Any) -> None:
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def _load_shared_mano(toolkit_root: Path) -> Any:
    root = toolkit_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MANO toolkit root not found: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    source = Path(__file__).resolve().parents[2] / "mano" / "mano(1).py"
    spec = importlib.util.spec_from_file_location("orbbec_qc_shared_mano", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared MANO module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shape_scale(episode_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    subject_dir = episode_dir.parents[1]
    shape_path = subject_dir / "shape.npy"
    if not shape_path.is_file():
        metadata_path = episode_dir / "mano" / "episode" / "mano_episode.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate = Path(str(metadata.get("shape_source") or ""))
            if candidate.is_file():
                shape_path = candidate
        except Exception:
            pass
    if not shape_path.is_file():
        raise FileNotFoundError(f"shape.npy not found: {shape_path}")
    shape = np.load(shape_path, allow_pickle=False).astype(np.float32).reshape(1, 10).repeat(2, axis=0)
    scale_path = subject_dir / "scale.npy"
    scale_value = 1.0
    if scale_path.is_file():
        scale_value = float(np.load(scale_path, allow_pickle=False).reshape(-1)[0])
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"invalid MANO scale {scale_value}: {scale_path}")
    return shape, np.asarray([[scale_value], [scale_value]], dtype=np.float32)


def _source_frame(episode_dir: Path, rgb_cache_dir: Path, camera: str, frame: int) -> Path:
    cached = rgb_cache_dir / camera / f"{frame:05d}.png"
    if cached.is_file():
        return cached
    rgb_dir = episode_dir / camera / "RGB"
    matches = sorted(path for path in rgb_dir.glob(f"{frame:05d}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"RGB frame not found: camera={camera} frame={frame}")
    return matches[0]


def _create_hand_mesh(vertices: np.ndarray, faces: np.ndarray, color: Tuple[float, float, float], trimesh: Any) -> Any:
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    rgba = np.asarray([int(channel * 255) for channel in color] + [255], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba.reshape(1, 4), (vertices.shape[0], 1))
    rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rotation)
    return mesh


def _add_lights(scene: Any, pyrender: Any) -> None:
    for index, translation in enumerate(((0.0, 0.0, 0.0), (0.0, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5))):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
        scene.add_node(
            pyrender.Node(
                name=f"hand-light-{index}",
                light=pyrender.PointLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )


class CameraMeshRenderer:
    def __init__(self, *, width: int, height: int, intrinsics: np.ndarray, render_factor: float):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.trimesh = trimesh
        self.width = int(width)
        self.height = int(height)
        self.factor = float(render_factor)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=max(1, int(round(self.width * self.factor))),
            viewport_height=max(1, int(round(self.height * self.factor))),
            point_size=1.0,
        )

    def close(self) -> None:
        self.renderer.delete()

    def composite(self, image: np.ndarray, vertices_by_hand: Mapping[int, np.ndarray], faces: Mapping[int, np.ndarray]) -> np.ndarray:
        pyrender = self.pyrender
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.25, 0.25, 0.25))
        for hand in (0, 1):
            mesh = _create_hand_mesh(vertices_by_hand[hand], faces[hand], HAND_COLORS_RGB[hand], self.trimesh)
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True), name=f"hand_mesh_{hand}")
        k = self.intrinsics
        camera = pyrender.IntrinsicsCamera(
            fx=float(k[0, 0] * self.factor),
            fy=float(k[1, 1] * self.factor),
            cx=float(k[0, 2] * self.factor),
            cy=float(k[1, 2] * self.factor),
            zfar=1e12,
        )
        scene.add_node(pyrender.Node(camera=camera, matrix=np.eye(4, dtype=np.float32)))
        _add_lights(scene, pyrender)
        rgba, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        layer = rgba.astype(np.float32) / 255.0
        if self.factor != 1.0:
            layer = cv2.resize(layer, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image_rgb = image.astype(np.float32)[:, :, ::-1] / 255.0
        alpha = layer[:, :, 3:]
        composite_rgb = image_rgb * (1.0 - alpha) + layer[:, :, :3] * alpha
        return np.clip(composite_rgb[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)


def render_request(request: Mapping[str, Any]) -> None:
    import torch

    from label.mano_view import load_episode_cameras

    episode_dir = Path(str(request["episode_dir"])).expanduser().resolve()
    rgb_cache_dir = Path(str(request["rgb_cache_dir"])).expanduser().resolve()
    output_dir = Path(str(request["output_dir"])).expanduser().resolve()
    toolkit_root = Path(str(request["mano_toolkit_root"])).expanduser().resolve()
    model_dir = Path(str(request["mano_model_dir"])).expanduser().resolve()
    cameras = [str(value) for value in request.get("cameras") or []][:6]
    frames = sorted({int(value) for value in request.get("frames") or []})
    factor = float(request.get("render_factor") or 1.0)
    if not cameras or not frames:
        raise ValueError("mesh render request requires cameras and frames")
    pose_dir = episode_dir / "optimized_pose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"optimized_pose not found: {pose_dir}")

    mano = _load_shared_mano(toolkit_root)
    layers = mano.build_mano_layers(model_dir)
    faces = {hand: mano.mano_faces(layers[hand]) for hand in (0, 1)}
    betas, scales = _load_shape_scale(episode_dir)
    camera_params = load_episode_cameras(episode_dir, cameras)
    renderers: Dict[str, CameraMeshRenderer] = {}
    completed = {camera: 0 for camera in cameras}
    total = len(frames)
    progress_step = max(1, total // 100)
    try:
        for camera in cameras:
            (output_dir / camera).mkdir(parents=True, exist_ok=True)
            completed[camera] = sum(1 for frame in frames if (output_dir / camera / f"{frame:05d}.jpg").is_file())
            _emit(camera=camera, status="mesh_pending", rendered=completed[camera], total=total)

        for frame in frames:
            targets = {camera: output_dir / camera / f"{frame:05d}.jpg" for camera in cameras}
            missing_cameras = [camera for camera in cameras if not targets[camera].is_file()]
            if not missing_cameras:
                continue
            pose_path = pose_dir / f"{frame:05d}.npy"
            pose = mano.load_pose(pose_path)
            outputs = mano.mano_outputs_from_pose(pose, betas, scales, layers)
            for camera in missing_cameras:
                source_path = _source_frame(episode_dir, rgb_cache_dir, camera, frame)
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read RGB frame: {source_path}")
                height, width = image.shape[:2]
                renderer = renderers.get(camera)
                if renderer is None:
                    renderer = CameraMeshRenderer(
                        width=width,
                        height=height,
                        intrinsics=camera_params[camera].k,
                        render_factor=factor,
                    )
                    renderers[camera] = renderer
                r = torch.as_tensor(camera_params[camera].r, dtype=torch.float32)
                t = torch.as_tensor(camera_params[camera].t, dtype=torch.float32)
                vertices_by_hand: Dict[int, np.ndarray] = {}
                for hand in (0, 1):
                    vertices = outputs[hand]["vertices"][0]
                    vertices_camera = torch.matmul(vertices, r.transpose(0, 1)) + t
                    vertices_by_hand[hand] = vertices_camera.detach().cpu().numpy()
                rendered = renderer.composite(image, vertices_by_hand, faces)
                target = targets[camera]
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.jpg")
                if not cv2.imwrite(str(temporary), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"failed to write mesh frame: {temporary}")
                os.replace(temporary, target)
                completed[camera] += 1
                if completed[camera] == total or completed[camera] % progress_step == 0:
                    _emit(camera=camera, status="mesh_rendering", rendered=completed[camera], total=total)
        for camera in cameras:
            _emit(camera=camera, status="mesh_done", rendered=completed[camera], total=total)
    finally:
        for renderer in renderers.values():
            renderer.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-render synchronized QC MANO mesh frames")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, Mapping):
            raise ValueError("mesh render request must be a JSON object")
        render_request(request)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
