from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


HAND_COLORS_RGB = {0: (0.85, 0.45, 0.45), 1: (0.65, 0.74, 0.86)}

_WORKER_STATE: Dict[str, Any] = {}
_WORKER_RENDERERS: Dict[str, "CameraMeshRenderer"] = {}


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
    def __init__(self, *, camera: str, width: int, height: int, intrinsics: np.ndarray, render_factor: float):
        self.camera = str(camera)
        self.width = int(width)
        self.height = int(height)
        self.factor = float(render_factor)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.pyrender = None
        self.trimesh = None
        self.renderer = None
        self._initialize_opengl()

    def _initialize_opengl(self) -> None:
        # QC runs as a desktop application. Prefer the host's working GLX/pyglet
        # context when a display is present; reserve EGL for truly headless use.
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        try:
            import pyrender
            import trimesh

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


def _initialize_render_worker(state: Mapping[str, Any]) -> None:
    """Initialize one process that renders complete six-view frame groups."""
    global _WORKER_STATE
    _WORKER_STATE = dict(state)
    _WORKER_RENDERERS.clear()
    cv2.setNumThreads(1)
    atexit.register(_close_worker_renderers)


def _render_frame_worker(frame: int, vertices_world: Mapping[int, np.ndarray]) -> Tuple[int, List[str]]:
    if not _WORKER_STATE:
        raise RuntimeError("mesh render worker was not initialized")
    episode_dir = Path(str(_WORKER_STATE["episode_dir"]))
    rgb_cache_dir = Path(str(_WORKER_STATE["rgb_cache_dir"]))
    output_dir = Path(str(_WORKER_STATE["output_dir"]))
    cameras = [str(value) for value in _WORKER_STATE["cameras"]]
    camera_params = _WORKER_STATE["camera_params"]
    faces = _WORKER_STATE["faces"]
    factor = float(_WORKER_STATE["render_factor"])
    rendered_cameras: List[str] = []

    for camera in cameras:
        target = output_dir / camera / f"{int(frame):05d}.jpg"
        if target.is_file():
            continue
        source_path = _source_frame(episode_dir, rgb_cache_dir, camera, int(frame))
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read RGB frame: {source_path}")
        height, width = image.shape[:2]
        renderer = _WORKER_RENDERERS.get(camera)
        if renderer is None:
            renderer = CameraMeshRenderer(
                camera=camera,
                width=width,
                height=height,
                intrinsics=np.asarray(camera_params[camera]["k"], dtype=np.float32),
                render_factor=factor,
            )
            _WORKER_RENDERERS[camera] = renderer

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
    toolkit_root = Path(str(request["mano_toolkit_root"])).expanduser().resolve()
    model_dir = Path(str(request["mano_model_dir"])).expanduser().resolve()
    cameras = [str(value) for value in request.get("cameras") or []][:6]
    frames = sorted({int(value) for value in request.get("frames") or []})
    factor = float(request.get("render_factor") or 1.0)
    requested_workers = max(1, min(32, int(request.get("workers") or 1)))
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
    completed = {camera: 0 for camera in cameras}
    total = len(frames)
    progress_step = max(1, total // 100)
    for camera in cameras:
        (output_dir / camera).mkdir(parents=True, exist_ok=True)
        completed[camera] = sum(1 for frame in frames if (output_dir / camera / f"{frame:05d}.jpg").is_file())
        _emit(camera=camera, status="mesh_pending", rendered=completed[camera], total=total)

    pending_frames = [
        frame
        for frame in frames
        if any(not (output_dir / camera / f"{frame:05d}.jpg").is_file() for camera in cameras)
    ]
    worker_count = min(requested_workers, max(1, len(pending_frames)))
    camera_state = {
        camera: {
            "k": np.asarray(camera_params[camera].k, dtype=np.float32),
            "r": np.asarray(camera_params[camera].r, dtype=np.float32),
            "t": np.asarray(camera_params[camera].t, dtype=np.float32),
        }
        for camera in cameras
    }
    worker_state = {
        "episode_dir": str(episode_dir),
        "rgb_cache_dir": str(rgb_cache_dir),
        "output_dir": str(output_dir),
        "cameras": cameras,
        "camera_params": camera_state,
        "faces": faces,
        "render_factor": factor,
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
