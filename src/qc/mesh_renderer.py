from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


HAND_COLORS_RGB = {0: (0.85, 0.45, 0.45), 1: (0.65, 0.74, 0.86)}

_WORKER_STATE: Dict[str, Any] = {}
_WORKER_RENDERERS: Dict[str, "CameraMeshRenderer"] = {}
EGO_CAMERA = "ego"


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


def _intrinsic_matrix(entry: Mapping[str, Any]) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = float(entry["fx"])
    matrix[1, 1] = float(entry["fy"])
    matrix[0, 2] = float(entry["cx"])
    matrix[1, 2] = float(entry["cy"])
    return matrix


def load_ego_camera(episode_dir: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load the original Pico fisheye calibration used by the raw RGB frames."""
    camera_path = episode_dir / "ego" / "camera_params.json"
    data = json.loads(camera_path.read_text(encoding="utf-8"))
    try:
        rgb = data["ego"]["RGB"]
        intrinsic = rgb["intrinsic"]
        distortion = rgb["distortion"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing ego RGB calibration in {camera_path}") from exc
    if distortion.get("modelName") != "opencv_fisheye":
        raise ValueError(f"ego distortion model must be opencv_fisheye: {camera_path}")
    matrix = _intrinsic_matrix(intrinsic)
    coefficients = np.asarray(
        [distortion[f"k{index}"] for index in range(1, 5)], dtype=np.float32
    ).reshape(4, 1)
    image_size = (int(intrinsic["width"]), int(intrinsic["height"]))
    return matrix, coefficients, image_size


def load_ego_extrinsics(episode_dir: Path) -> Dict[int, np.ndarray]:
    """Load p_ego = T_ego_from_reference * p_reference for every QC frame."""
    candidates = (episode_dir / "ego_pose.json", episode_dir / "ego_extrinsic.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    data = json.loads(path.read_text(encoding="utf-8"))
    transforms: Dict[int, np.ndarray] = {}
    if isinstance(data, Mapping) and isinstance(data.get("frames"), list):
        convention = data.get("coordinate_convention")
        if isinstance(convention, Mapping):
            reference_view = str(convention.get("reference_view") or "00")
            if reference_view != "00":
                raise ValueError(f"ego pose reference_view must be 00, got {reference_view}: {path}")
        entries = data["frames"]
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            raw_frame = entry.get("frame_index")
            frame_text = "" if raw_frame is None else str(raw_frame).strip()
            value = entry.get("T_ego_from_reference")
            if not frame_text or value is None:
                continue
            frame = int(frame_text)
            transform = np.asarray(value, dtype=np.float32)
            if transform.shape != (4, 4):
                raise ValueError(f"ego extrinsic must have shape (4, 4), got {transform.shape} for frame {frame}")
            if not np.isfinite(transform).all():
                raise ValueError(f"non-finite ego extrinsic for frame {frame}: {path}")
            transforms[frame] = transform
    elif isinstance(data, Mapping):
        for frame_text, value in data.items():
            if not str(frame_text).isdigit():
                raise ValueError(f"invalid ego extrinsic frame key: {frame_text}")
            transform = np.asarray(value, dtype=np.float32)
            if transform.shape != (4, 4):
                raise ValueError(f"ego extrinsic must have shape (4, 4), got {transform.shape} for frame {frame_text}")
            transforms[int(frame_text)] = transform
    if not transforms:
        raise ValueError(f"no Pico ego extrinsics found in {path}")
    return transforms


def _source_frame(
    episode_dir: Path,
    rgb_cache_dir: Path,
    camera: str,
    frame: int,
    *,
    wait_seconds: float = 0.0,
) -> Path:
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        for suffix in ("jpg", "png"):
            cached = rgb_cache_dir / camera / f"{frame:05d}.{suffix}"
            if cached.is_file():
                return cached
        rgb_dir = episode_dir / camera / "RGB"
        matches = sorted(path for path in rgb_dir.glob(f"{frame:05d}.*") if path.is_file())
        if matches:
            return matches[0]
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"RGB frame not found: camera={camera} frame={frame}")
        time.sleep(0.01)


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


def build_fisheye_render_map(
    image_size: Tuple[int, int],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match mano/ego_pose.py: map raw fisheye pixels into the pinhole mesh layer."""
    width, height = image_size
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    fisheye_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    pinhole_pixels = cv2.fisheye.undistortPoints(
        fisheye_pixels,
        np.asarray(intrinsic, dtype=np.float32),
        np.asarray(distortion, dtype=np.float32).reshape(4, 1),
        P=np.asarray(intrinsic, dtype=np.float32),
    ).reshape(height, width, 2)
    return pinhole_pixels[..., 0], pinhole_pixels[..., 1]


class CameraMeshRenderer:
    def __init__(
        self,
        *,
        camera: str,
        width: int,
        height: int,
        intrinsics: np.ndarray,
        render_factor: float,
        fisheye_distortion: np.ndarray | None = None,
    ):
        self.camera = str(camera)
        self.width = int(width)
        self.height = int(height)
        self.factor = float(render_factor)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.fisheye_distortion = (
            None
            if fisheye_distortion is None
            else np.asarray(fisheye_distortion, dtype=np.float32).reshape(4, 1)
        )
        self.fisheye_map = (
            None
            if self.fisheye_distortion is None
            else build_fisheye_render_map(
                (self.width, self.height), self.intrinsics, self.fisheye_distortion
            )
        )
        self.pyrender = None
        self.trimesh = None
        self.renderer = None
        self._initialize_opengl()

    def _initialize_opengl(self) -> None:
        prefer_integrated = os.environ.get("ORBBEC_MESH_PREFER_INTEGRATED_GPU", "0") == "1"
        # The 9950X QC hosts use the AMD iGPU even when another display adapter is
        # installed. EGL lets us select that DRM device without depending on Xorg.
        if prefer_integrated:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender
            import trimesh

            if prefer_integrated and os.environ.get("PYOPENGL_PLATFORM") == "egl":
                was_selected = os.environ.get("EGL_DEVICE_ID") is not None
                selected = self._select_integrated_egl_device()
                if selected is not None and not was_selected:
                    _emit(camera=self.camera, status="mesh_gpu", device=selected)

            self.pyrender = pyrender
            self.trimesh = trimesh
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=max(1, int(round(self.width * self.factor))),
                viewport_height=max(1, int(round(self.height * self.factor))),
                point_size=1.0,
            )
        except Exception as exc:
            self.pyrender = None
            self.trimesh = None
            self.renderer = None
            _emit(
                camera=self.camera,
                status="mesh_software_fallback",
                error=f"OpenGL unavailable; using software mesh renderer: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _select_integrated_egl_device() -> str | None:
        if os.environ.get("EGL_DEVICE_ID") is not None:
            return f"EGL device {os.environ['EGL_DEVICE_ID']}"
        from pyrender.platforms.egl import query_devices

        devices = query_devices()
        for index, device in enumerate(devices):
            name = str(device.name or "")
            card = Path(name).name
            vendor_path = Path("/sys/class/drm") / card / "device" / "vendor"
            try:
                vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue
            if vendor == "0x1002":
                os.environ["EGL_DEVICE_ID"] = str(index)
                return f"{name} (AMD, EGL_DEVICE_ID={index})"
        return None

    def close(self) -> None:
        if self.renderer is None:
            return
        try:
            self.renderer.delete()
        except Exception:
            pass
        self.renderer = None

    def composite(self, image: np.ndarray, vertices_by_hand: Mapping[int, np.ndarray], faces: Mapping[int, np.ndarray]) -> np.ndarray:
        if self.renderer is None or self.pyrender is None or self.trimesh is None:
            return self._software_composite(image, vertices_by_hand, faces)
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
        try:
            rgba, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        except Exception as exc:
            _emit(
                camera=self.camera,
                status="mesh_software_fallback",
                error=f"OpenGL render failed; using software mesh renderer: {type(exc).__name__}: {exc}",
            )
            self.close()
            self.pyrender = None
            self.trimesh = None
            return self._software_composite(image, vertices_by_hand, faces)
        layer = rgba.astype(np.float32) / 255.0
        if self.factor != 1.0:
            layer = cv2.resize(layer, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.fisheye_map is not None:
            # Keep the raw Pico RGB unchanged and distort only the transparent
            # MANO layer, exactly as mano/ego_pose.py does.
            layer = cv2.remap(
                layer,
                self.fisheye_map[0],
                self.fisheye_map[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        image_rgb = image.astype(np.float32)[:, :, ::-1] / 255.0
        alpha = layer[:, :, 3:]
        composite_rgb = image_rgb * (1.0 - alpha) + layer[:, :, :3] * alpha
        return np.clip(composite_rgb[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)

    def _software_composite(
        self,
        image: np.ndarray,
        vertices_by_hand: Mapping[int, np.ndarray],
        faces: Mapping[int, np.ndarray],
    ) -> np.ndarray:
        """Depth-sort and rasterize MANO triangles without an OpenGL context."""
        triangles = []
        k = self.intrinsics
        for hand in (0, 1):
            vertices = np.asarray(vertices_by_hand[hand], dtype=np.float32)
            z = vertices[:, 2]
            valid = np.isfinite(vertices).all(axis=1) & (z > 1e-6)
            uv = np.full((len(vertices), 2), np.nan, dtype=np.float32)
            if self.fisheye_distortion is not None and valid.any():
                projected, _ = cv2.fisheye.projectPoints(
                    vertices[valid].reshape(-1, 1, 3),
                    np.zeros((3, 1), dtype=np.float32),
                    np.zeros((3, 1), dtype=np.float32),
                    k,
                    self.fisheye_distortion,
                )
                uv[valid] = projected.reshape(-1, 2)
            else:
                uv[valid, 0] = vertices[valid, 0] / z[valid] * k[0, 0] + k[0, 2]
                uv[valid, 1] = vertices[valid, 1] / z[valid] * k[1, 1] + k[1, 2]
            base_rgb = np.asarray(HAND_COLORS_RGB[hand], dtype=np.float32) * 255.0
            base_bgr = base_rgb[::-1]
            for face in np.asarray(faces[hand], dtype=np.int32):
                indices = face[:3]
                if not np.all(valid[indices]):
                    continue
                points = uv[indices]
                if not np.isfinite(points).all():
                    continue
                edge_a = vertices[indices[1]] - vertices[indices[0]]
                edge_b = vertices[indices[2]] - vertices[indices[0]]
                normal = np.cross(edge_a, edge_b)
                normal_length = float(np.linalg.norm(normal))
                facing = 0.0 if normal_length <= 1e-8 else abs(float(normal[2])) / normal_length
                brightness = 0.58 + 0.42 * facing
                color = tuple(int(value) for value in np.clip(base_bgr * brightness, 0, 255))
                triangles.append((float(np.mean(z[indices])), np.rint(points).astype(np.int32), color))

        rendered = image.copy()
        for _depth, points, color in sorted(triangles, key=lambda item: item[0], reverse=True):
            cv2.fillConvexPoly(rendered, points, color, lineType=cv2.LINE_AA)
        return rendered


def _close_worker_renderers() -> None:
    for renderer in _WORKER_RENDERERS.values():
        renderer.close()
    _WORKER_RENDERERS.clear()


def _write_display_preview(image: np.ndarray, target: Path, *, max_width: int) -> None:
    """Write a smaller QC-only image after full-resolution fisheye compositing."""
    height, width = image.shape[:2]
    limit = max(320, int(max_width))
    if width > limit:
        scale = float(limit) / float(width)
        image = cv2.resize(
            image,
            (limit, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.jpg")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"failed to write mesh preview frame: {temporary}")
    os.replace(temporary, target)


def _initialize_render_worker(state: Mapping[str, Any]) -> None:
    """Initialize one process that renders complete six-view frame groups."""
    global _WORKER_STATE
    _WORKER_STATE = dict(state)
    _WORKER_RENDERERS.clear()
    if bool(_WORKER_STATE.get("prefer_integrated_gpu", False)):
        os.environ["ORBBEC_MESH_PREFER_INTEGRATED_GPU"] = "1"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    cv2.setNumThreads(1)
    atexit.register(_close_worker_renderers)


def _render_frame_worker(frame: int, vertices_world: Mapping[int, np.ndarray]) -> Tuple[int, List[str]]:
    if not _WORKER_STATE:
        raise RuntimeError("mesh render worker was not initialized")
    episode_dir = Path(str(_WORKER_STATE["episode_dir"]))
    rgb_cache_dir = Path(str(_WORKER_STATE["rgb_cache_dir"]))
    output_dir = Path(str(_WORKER_STATE["output_dir"]))
    preview_output_dir = Path(str(_WORKER_STATE["preview_output_dir"]))
    cameras = [str(value) for value in _WORKER_STATE["cameras"]]
    camera_params = _WORKER_STATE["camera_params"]
    faces = _WORKER_STATE["faces"]
    factor = float(_WORKER_STATE["render_factor"])
    source_wait_seconds = float(_WORKER_STATE.get("source_wait_seconds") or 0.0)
    ego_preview_max_width = int(_WORKER_STATE.get("ego_preview_max_width") or 960)
    rendered_cameras: List[str] = []

    for camera in cameras:
        target = output_dir / camera / f"{int(frame):05d}.jpg"
        display_target = (
            preview_output_dir / camera / f"{int(frame):05d}.jpg"
            if camera == EGO_CAMERA
            else target
        )
        if display_target.is_file():
            continue
        if camera == EGO_CAMERA and target.is_file():
            existing = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if existing is None:
                raise RuntimeError(f"failed to read existing Pico mesh frame: {target}")
            _write_display_preview(existing, display_target, max_width=ego_preview_max_width)
            rendered_cameras.append(camera)
            continue
        source_path = _source_frame(
            episode_dir,
            rgb_cache_dir,
            camera,
            int(frame),
            wait_seconds=source_wait_seconds,
        )
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read RGB frame: {source_path}")
        height, width = image.shape[:2]
        if camera == EGO_CAMERA:
            expected_size = tuple(int(value) for value in camera_params[camera]["image_size"])
            if (width, height) != expected_size:
                raise ValueError(
                    f"Pico RGB size mismatch for frame {frame}: expected {expected_size}, got {(width, height)}"
                )
        renderer = _WORKER_RENDERERS.get(camera)
        if renderer is None:
            distortion = camera_params[camera].get("distortion")
            renderer = CameraMeshRenderer(
                camera=camera,
                width=width,
                height=height,
                intrinsics=np.asarray(camera_params[camera]["k"], dtype=np.float32),
                render_factor=factor,
                fisheye_distortion=(
                    None if distortion is None else np.asarray(distortion, dtype=np.float32)
                ),
            )
            _WORKER_RENDERERS[camera] = renderer

        if camera == EGO_CAMERA:
            transform = np.asarray(_WORKER_STATE["ego_transforms"][int(frame)], dtype=np.float32)
            r = transform[:3, :3]
            t = transform[:3, 3]
        else:
            r = np.asarray(camera_params[camera]["r"], dtype=np.float32)
            t = np.asarray(camera_params[camera]["t"], dtype=np.float32)
        vertices_by_hand = {
            hand: np.matmul(np.asarray(vertices_world[hand], dtype=np.float32), r.T) + t
            for hand in (0, 1)
        }
        rendered = renderer.composite(image, vertices_by_hand, faces)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.jpg")
        if not cv2.imwrite(str(temporary), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write mesh frame: {temporary}")
        os.replace(temporary, target)
        if camera == EGO_CAMERA:
            _write_display_preview(rendered, display_target, max_width=ego_preview_max_width)
        rendered_cameras.append(camera)
    return int(frame), rendered_cameras


def _world_vertices_for_frame(
    *,
    frame: int,
    pose_dir: Path,
    mano: Any,
    betas: np.ndarray,
    scales: np.ndarray,
    layers: Mapping[int, Any],
) -> Dict[int, np.ndarray]:
    pose = mano.load_pose(pose_dir / f"{int(frame):05d}.npy")
    outputs = mano.mano_outputs_from_pose(pose, betas, scales, layers)
    return {
        hand: outputs[hand]["vertices"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        for hand in (0, 1)
    }


def render_request(request: Mapping[str, Any]) -> None:
    import torch

    from label.mano_view import load_episode_cameras

    episode_dir = Path(str(request["episode_dir"])).expanduser().resolve()
    rgb_cache_dir = Path(str(request["rgb_cache_dir"])).expanduser().resolve()
    output_dir = Path(str(request["output_dir"])).expanduser().resolve()
    preview_output_dir = Path(
        str(request.get("preview_output_dir") or (output_dir.parent / "mesh_preview"))
    ).expanduser().resolve()
    toolkit_root = Path(str(request["mano_toolkit_root"])).expanduser().resolve()
    model_dir = Path(str(request["mano_model_dir"])).expanduser().resolve()
    cameras = [str(value) for value in request.get("cameras") or []][:6]
    frames = sorted({int(value) for value in request.get("frames") or []})
    factor = float(request.get("render_factor") or 1.0)
    requested_workers = max(1, min(32, int(request.get("workers") or 1)))
    ego_preview_max_width = max(320, int(request.get("ego_preview_max_width") or 960))
    if not cameras or not frames:
        raise ValueError("mesh render request requires cameras and frames")
    pose_dir = episode_dir / "optimized_pose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"optimized_pose not found: {pose_dir}")

    mano = _load_shared_mano(toolkit_root)
    layers = mano.build_mano_layers(model_dir)
    faces = {hand: mano.mano_faces(layers[hand]) for hand in (0, 1)}
    betas, scales = _load_shape_scale(episode_dir)
    rgb_cameras = [camera for camera in cameras if camera != EGO_CAMERA]
    camera_params = load_episode_cameras(episode_dir, rgb_cameras)
    ego_transforms: Dict[int, np.ndarray] = {}
    ego_intrinsic: np.ndarray | None = None
    ego_distortion: np.ndarray | None = None
    ego_image_size: Tuple[int, int] | None = None
    if EGO_CAMERA in cameras:
        ego_transforms = load_ego_extrinsics(episode_dir)
        missing_ego_frames = [frame for frame in frames if frame not in ego_transforms]
        if missing_ego_frames:
            preview = ", ".join(str(frame) for frame in missing_ego_frames[:5])
            raise ValueError(f"Pico ego extrinsics missing {len(missing_ego_frames)} QC frame(s): {preview}")
        ego_intrinsic, ego_distortion, ego_image_size = load_ego_camera(episode_dir)
    completed = {camera: 0 for camera in cameras}
    total = len(frames)
    progress_step = max(1, total // 100)
    if EGO_CAMERA in cameras:
        (preview_output_dir / EGO_CAMERA).mkdir(parents=True, exist_ok=True)
        for frame in frames:
            full_path = output_dir / EGO_CAMERA / f"{frame:05d}.jpg"
            preview_path = preview_output_dir / EGO_CAMERA / f"{frame:05d}.jpg"
            if preview_path.is_file() or not full_path.is_file():
                continue
            existing = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
            if existing is None:
                raise RuntimeError(f"failed to read existing Pico mesh frame: {full_path}")
            _write_display_preview(existing, preview_path, max_width=ego_preview_max_width)

    def rendered_path(camera: str, frame: int) -> Path:
        root = preview_output_dir if camera == EGO_CAMERA else output_dir
        return root / camera / f"{frame:05d}.jpg"

    for camera in cameras:
        (output_dir / camera).mkdir(parents=True, exist_ok=True)
        completed[camera] = sum(1 for frame in frames if rendered_path(camera, frame).is_file())
        _emit(camera=camera, status="mesh_pending", rendered=completed[camera], total=total)

    pending_frames = [
        frame
        for frame in frames
        if any(not rendered_path(camera, frame).is_file() for camera in cameras)
    ]
    worker_count = min(requested_workers, max(1, len(pending_frames)))
    camera_state: Dict[str, Dict[str, Any]] = {
        camera: {
            "k": np.asarray(camera_params[camera].k, dtype=np.float32),
            "r": np.asarray(camera_params[camera].r, dtype=np.float32),
            "t": np.asarray(camera_params[camera].t, dtype=np.float32),
        }
        for camera in rgb_cameras
    }
    if EGO_CAMERA in cameras:
        assert ego_intrinsic is not None and ego_distortion is not None and ego_image_size is not None
        camera_state[EGO_CAMERA] = {
            "k": ego_intrinsic,
            "distortion": ego_distortion,
            "image_size": ego_image_size,
        }
    worker_state = {
        "episode_dir": str(episode_dir),
        "rgb_cache_dir": str(rgb_cache_dir),
        "output_dir": str(output_dir),
        "preview_output_dir": str(preview_output_dir),
        "ego_preview_max_width": ego_preview_max_width,
        "cameras": cameras,
        "camera_params": camera_state,
        "ego_transforms": ego_transforms,
        "faces": faces,
        "render_factor": factor,
        "prefer_integrated_gpu": bool(request.get("prefer_integrated_gpu", False)),
        "source_wait_seconds": float(request.get("source_wait_seconds") or 0.0),
    }

    def record_result(result: Tuple[int, List[str]]) -> None:
        _frame, rendered_cameras = result
        for camera in rendered_cameras:
            completed[camera] += 1
            if completed[camera] == total or completed[camera] % progress_step == 0:
                _emit(camera=camera, status="mesh_rendering", rendered=completed[camera], total=total)

    if worker_count == 1:
        _initialize_render_worker(worker_state)
        try:
            for frame in pending_frames:
                vertices_world = _world_vertices_for_frame(
                    frame=frame,
                    pose_dir=pose_dir,
                    mano=mano,
                    betas=betas,
                    scales=scales,
                    layers=layers,
                )
                record_result(_render_frame_worker(frame, vertices_world))
        finally:
            _close_worker_renderers()
    elif pending_frames:
        # MANO is evaluated once per frame in the coordinator. Expensive image,
        # rasterization, and JPEG work is distributed without duplicating MANO.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        context = multiprocessing.get_context("spawn")
        in_flight: Dict[Future[Tuple[int, List[str]]], int] = {}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_render_worker,
            initargs=(worker_state,),
        ) as executor:
            for frame in pending_frames:
                vertices_world = _world_vertices_for_frame(
                    frame=frame,
                    pose_dir=pose_dir,
                    mano=mano,
                    betas=betas,
                    scales=scales,
                    layers=layers,
                )
                future = executor.submit(_render_frame_worker, frame, vertices_world)
                in_flight[future] = frame
                if len(in_flight) >= worker_count * 2:
                    done, _not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                    for finished in done:
                        in_flight.pop(finished, None)
                        record_result(finished.result())
            while in_flight:
                done, _not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                for finished in done:
                    in_flight.pop(finished, None)
                    record_result(finished.result())

    for camera in cameras:
        _emit(camera=camera, status="mesh_done", rendered=completed[camera], total=total)


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
