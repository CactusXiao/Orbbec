"""Isolated calculation worker. Inputs/outputs live in the browser cache only."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image, ImageDraw


def run(body, folder):
    from label.mano_view import ManoViewRuntime
    from mano.joint_order import SMPLX_MANO_SKELETON_EDGES
    episode = Path(body["episode"])
    states = {c: (s["points"], s["visible"]) for c, s in body["samples"].items()}
    if body["action"] == "track":
        from label.tracking import CoTrackerRuntime
        tracker = CoTrackerRuntime()
        result, errors = {}, []
        for camera, selected in body["selected"].items():
            if not selected:
                continue
            points, visible = states[camera]
            selected = [[h,j] for h,j in selected if visible[h][j]]
            if not selected:
                continue
            mask = np.zeros((2, 21), dtype=bool)
            for h, j in selected:
                mask[h, j] = visible[h][j]
            try:
                points, visible = tracker.track_points(
                    episode_dir=episode, cam_id=camera,
                    prev_frame_idx=body["frame"], frame_idx=body["target"],
                    points=points, visible=mask.tolist(), rgb_path_template=body["template"])
                if any(not np.isfinite(points[h][j]).all() or min(points[h][j]) < 0 for h,j in selected):
                    raise ValueError("跟踪返回了无效关节坐标。")
                result[camera] = dict(points=points, visible=visible, selected=selected)
            except Exception as exc:
                errors.append(f"Camera {camera}: {exc}")
        return {"samples": result, "target": body["target"], "errors": errors}
    from .browser_label import BrowserLabelRuntime
    runtime = BrowserLabelRuntime(body.get("frame", 0))
    joints = runtime.build_skeleton(episode_dir=episode, camera_ids=list(states), view_states=states)
    for camera, sample in body["samples"].items():
        points, visible = runtime.project_skeleton(episode_dir=episode, cam_id=camera, joints_3d=joints)
        image = Image.new("RGBA", (sample["width"], sample["height"]))
        draw = ImageDraw.Draw(image)
        for h, color in enumerate(("#37c7ff", "#ff8a3d")):
            for a, b in SMPLX_MANO_SKELETON_EDGES:
                if visible[h][a] and visible[h][b]:
                    draw.line([tuple(points[h][a]), tuple(points[h][b])], fill=color, width=4)
        image.save(folder / f"{camera}.png")
    return {"cameras": list(states)}


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    result = run(json.loads((folder / "input.json").read_text()), folder)
    (folder / "result.json").write_text(json.dumps(result, allow_nan=False))
