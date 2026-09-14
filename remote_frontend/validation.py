"""Isolated lab fixture. All writable outputs go under --state-dir, never NAS.

Raw video is referenced read-only by symlink. No production backend is called.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from task_backend.job_service import JobService
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_store import WorkflowStore


def create_fixture(root: Path, source: Path, frames: int, port: int):
    if root.exists():
        raise ValueError("fixture state-dir already exists; use a new directory")
    root.mkdir(parents=True)
    nas = root / "nas"
    cameras = [c for c in ("00", "02", "03", "05", "06") if (source / c).is_dir()]
    if len(cameras) != 5:
        raise ValueError("validation needs all five QC RGB cameras")
    for role in ("qc", "label"):
        target = nas / "pilot" / role / "episode1"
        target.mkdir(parents=True)
        for name in ("camera_params.json", "extrinsics.json", "ego_pose.json", "timestamps.csv"):
            shutil.copy2(source / name, target / name)
        for name in ("mano", "optimized_pose", "joints_vis"):
            if (source / name).is_dir():
                shutil.copytree(source / name, target / name)
        for camera in [*cameras, "ego"]:
            for path in (source / camera).rglob("*"):
                if not path.is_file() or "Depth" in path.parts:
                    continue
                dest = target / camera / path.relative_to(source / camera)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix in {".h265", ".mkv", ".mp4"}:
                    dest.symlink_to(path.resolve())
                elif path.suffix in {".json", ".csv"}:
                    shutil.copy2(path, dest)
    shape = source.parents[1] / "shape.npy"
    if shape.is_file():
        shutil.copy2(shape, nas / "pilot" / "shape.npy")
    # The fixture never starts Publisher or NAS status synchronization workers.
    service = JobService(WorkflowStore(root / "workflow.sqlite3"), nas_mounts={"nas://pilot": str(nas)})
    for role in ("qc", "label"):
        episode_id = f"pilot-{role}"
        service.store.create_or_update_episode(
            episode_id=episode_id, subject_id="pilot", task_name=role, episode_index=1,
            status="auto_labeled" if role == "qc" else "manual_correction_pending",
            episode_uri=f"nas://pilot/pilot/{role}/episode1", cameras=cameras, frame_count=frames)
        if role == "qc":
            service.create_dev_job({"type": "qc", "episode_id": episode_id,
                                    "payload": {"frames": list(range(frames))}})
        else:
            service.store.create_segment(segment_id="pilot-segment", episode_id=episode_id,
                                         start_frame=0, end_frame=2)
            service._create_manual_label_episode_job(episode_id, reason="isolated browser validation")
    toolkit = Path("/home/ubuntu/WorkSpace/zhenghao/opt_toolkits")
    config = {"backend_url": f"http://127.0.0.1:{port}", "operator_id": "browser-pilot",
              "nas_mounts": {"nas://pilot": str(nas)}, "mesh_renderer_python": str(toolkit / ".venv/bin/python"),
              "mano_toolkit_root": str(toolkit), "mano_model_dir": str(toolkit / "ckpt/mano"),
              "mesh_render_workers": 4, "mesh_prebuffer_frames": 30, "mesh_render_factor": 0.5,
              "mesh_prefer_integrated_gpu": True, "playback_fps": 30}
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    root = args.state_dir.resolve()
    service = create_fixture(root, args.episode.resolve(), args.frames, args.port)
    runtime = BackendRuntime(TaskInstanceRegistry(root / "registry", seed_task_files=[]), service)
    server = TaskHTTPServer(("127.0.0.1", args.port), RequestHandler, runtime)
    print(f"Isolated validation backend: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.viewer_manager.shutdown()


if __name__ == "__main__":
    main()
