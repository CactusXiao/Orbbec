from __future__ import annotations

import os
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from task_backend.optimized_pose_source import load_archive_frame

import cv2
import numpy as np
import torch
import trimesh

from common.config import load_config
from common.episode import (
    first_camera_extrinsics,
    frame_path,
    load_extrinsics,
    load_intrinsics,
    normalize_view_name,
)
from common.mano import (
    build_mano_layers,
    load_pose,
    mano_faces,
    mano_outputs_from_pose,
    transform_points,
)
from visualizer.config import parse_visual_config


HAND_COLORS_RGB = {0: (0.85, 0.45, 0.45), 1: (0.65, 0.74, 0.86)}


def _load_shape_scale(subject_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    shape_path = subject_dir / "shape.npy"
    if not shape_path.is_file():
        raise FileNotFoundError(f"shape.npy not found: {shape_path}")
    shape = np.load(shape_path).astype(np.float32).reshape(1, 10).repeat(2, axis=0)
    scale_path = subject_dir / "scale.npy"
    scale_value = 1.0 if not scale_path.is_file() else float(
        np.load(scale_path).reshape(-1)[0]
    )
    return shape, np.asarray([[scale_value], [scale_value]], dtype=np.float32)


def create_hand_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
) -> trimesh.Trimesh:
    # Pyrender uses OpenGL camera coordinates, so rotate camera-space MANO vertices into its convention.
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    vertex_color = np.asarray([int(channel * 255) for channel in color] + [255], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(vertex_color.reshape(1, 4), (vertices.shape[0], 1))
    render_rotation = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(render_rotation)
    return mesh


def build_scene_meshes(
    outputs: dict[int, dict[str, torch.Tensor]],
    faces: dict[int, np.ndarray],
    T_world_to_camera: torch.Tensor,
) -> list[trimesh.Trimesh]:
    # Build smooth shaded hand meshes in camera space before image compositing.
    scene_meshes = []
    for hand in (0, 1):
        vertices_camera = (
            transform_points(outputs[hand]["vertices"][0], T_world_to_camera)
            .detach()
            .cpu()
            .numpy()
        )
        scene_meshes.append(
            create_hand_mesh(vertices_camera, faces[hand], HAND_COLORS_RGB[hand])
        )
    return scene_meshes


def add_scene_lights(scene: object, pyrender_module: object) -> None:
    # Camera-local lights keep the smooth surface readable from all hand poses.
    light_poses = (
        (0.0, 0.0, 0.0),
        (0.0, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    )
    for index, translation in enumerate(light_poses):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
        scene.add_node(
            pyrender_module.Node(
                name=f"hand-light-{index}",
                light=pyrender_module.PointLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )


def render_scene_meshes_on_image(
    image: np.ndarray,
    scene_meshes: list[trimesh.Trimesh],
    K: torch.Tensor,
    render_factor: float = 2.0,
    fisheye_map: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    # Import pyrender after setting the platform so headless rendering fails loudly only when the renderer is used.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    if render_factor <= 0.0:
        raise ValueError(f"render_factor must be positive, got {render_factor}")

    height, width = image.shape[:2]
    render_width = int(round(width * render_factor))
    render_height = int(round(height * render_factor))
    intrinsics = K.detach().cpu().numpy().astype(np.float32, copy=False)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.25, 0.25, 0.25))
    for mesh_index, mesh in enumerate(scene_meshes):
        scene.add(
            pyrender.Mesh.from_trimesh(mesh, smooth=True),
            name=f"hand_mesh_{mesh_index}",
        )

    camera = pyrender.IntrinsicsCamera(
        fx=float(intrinsics[0, 0] * render_factor),
        fy=float(intrinsics[1, 1] * render_factor),
        cx=float(intrinsics[0, 2] * render_factor),
        cy=float(intrinsics[1, 2] * render_factor),
        zfar=1e12,
    )
    scene.add_node(pyrender.Node(camera=camera, matrix=np.eye(4, dtype=np.float32)))
    add_scene_lights(scene, pyrender)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=render_width,
        viewport_height=render_height,
        point_size=1.0,
    )
    try:
        render_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    render_rgba = render_rgba.astype(np.float32) / 255.0
    if render_factor != 1.0:
        render_rgba = cv2.resize(render_rgba, (width, height), interpolation=cv2.INTER_AREA)
    if fisheye_map is not None:
        map_x, map_y = fisheye_map
        if map_x.shape != (height, width) or map_y.shape != (height, width):
            raise ValueError(
                f"fisheye map shape must match image {(height, width)}, "
                f"got {map_x.shape} and {map_y.shape}"
            )
        # Distort only the transparent mesh layer; the source camera image remains unchanged.
        render_rgba = cv2.remap(
            render_rgba,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    image_rgb = image.astype(np.float32)[:, :, ::-1] / 255.0
    alpha = render_rgba[:, :, 3:]
    composite_rgb = image_rgb * (1.0 - alpha) + render_rgba[:, :, :3] * alpha
    return np.clip(composite_rgb[:, :, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)


def run(config: dict) -> None:
    visual_config = parse_visual_config(config, ("pose_dir", "camera_stream", "mano_dir"))
    spec = visual_config.spec
    views = visual_config.views
    frames = visual_config.frames
    intrinsics = load_intrinsics(spec.episode_dir, views, config["camera_stream"])
    reference_view = normalize_view_name(config.get("camera_reference_view", 0))
    extrinsic_views = list(dict.fromkeys([reference_view, *views]))
    extrinsics = first_camera_extrinsics(load_extrinsics(spec.episode_dir, extrinsic_views), views, reference_view)
    betas, scales = _load_shape_scale(spec.subject_dir)
    layers = build_mano_layers(config["mano_dir"])
    faces = {hand: mano_faces(layers[hand]) for hand in (0, 1)}
    render_factor = float(config.get("render_factor", 2.0))
    out_root = visual_config.output_dir
    for frame in frames:
        pose_dir = spec.episode_dir / config["pose_dir"]
        pose = (load_archive_frame(pose_dir, frame) if (pose_dir / "poses.npz").is_file()
                else load_pose(pose_dir / f"{frame:05d}.npy"))
        outputs = mano_outputs_from_pose(pose, betas, scales, layers)
        for view in views:
            image = cv2.imread(
                str(frame_path(spec.episode_dir, view, "RGB", frame, config["image_ext"]))
            )
            if image is None:
                raise RuntimeError(f"failed to read image for view={view} frame={frame}")
            T_world_to_camera = torch.tensor(extrinsics[view], dtype=torch.float32)
            K = torch.tensor(intrinsics[view], dtype=torch.float32)
            scene_meshes = build_scene_meshes(outputs, faces, T_world_to_camera)
            rendered = render_scene_meshes_on_image(image, scene_meshes, K, render_factor)
            out_dir = out_root / "pose_mesh" / view
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{frame:05d}.jpg"), rendered)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
