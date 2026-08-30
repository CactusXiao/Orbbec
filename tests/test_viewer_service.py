from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from task_backend.viewer_service import ViewerSessionManager, render_viewer_page


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class ViewerSessionManagerTest(unittest.TestCase):
    def _episode(self, root: Path) -> Path:
        episode = root / "episode_1"
        for camera in ("00", "01", "02", "03", "04", "05"):
            rgb = episode / camera / "RGB"
            rgb.mkdir(parents=True)
            (rgb / "00000.png").write_bytes(PNG_1X1)
            (rgb / "00001.png").write_bytes(PNG_1X1)
        (episode / "timestamps.csv").write_text(
            "frame_index,ref_timestamp_us\n0,1000000\n1,1033333\n", encoding="utf-8"
        )
        return episode

    @staticmethod
    def _wait(session, mode: str = "") -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            payload = session.payload()
            status = payload["modes"][mode]["status"] if mode else payload["state"]
            if status in {"ready", "failed"}:
                return
            time.sleep(0.02)
        raise AssertionError("viewer preparation timed out")

    def test_rgb_session_uses_six_sources_and_removes_temp_dir_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ViewerSessionManager(temp_root=root / "viewer")
            session = manager.create("episode-id", self._episode(root))
            self._wait(session)
            payload = session.payload()
            self.assertEqual(payload["state"], "ready")
            self.assertEqual(payload["frames"], [0, 1])
            self.assertEqual(len([source for source in payload["sources"] if source["kind"] == "multiview"]), 6)
            path, content_type = manager.media_path(session.session_id, "rgb", "mv:00", 0)
            self.assertTrue(path.is_file())
            self.assertEqual(content_type, "image/png")
            temp_dir = session.temp_dir
            self.assertTrue(manager.close(session.session_id))
            self.assertFalse(temp_dir.exists())

    def test_pointcloud_is_prepared_only_after_mode_request(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("point-cloud runtime dependencies are not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = self._episode(root)
            params = {}
            extrinsics = {}
            for index, camera in enumerate(("00", "01", "02", "03", "04", "05")):
                depth_dir = episode / camera / "Depth"
                depth_dir.mkdir()
                cv2.imwrite(str(depth_dir / "00000.png"), np.array([[1000]], dtype=np.uint16))
                cv2.imwrite(str(depth_dir / "00001.png"), np.array([[1100]], dtype=np.uint16))
                intrinsic = {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 1, "height": 1}
                params[camera] = {"RGB": {"intrinsic": intrinsic}, "Depth": {"intrinsic": intrinsic}}
                extrinsics[camera] = {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation": [index * 0.01, 0, 0],
                }
            (episode / "camera_params.json").write_text(json.dumps(params), encoding="utf-8")
            (episode / "extrinsics.json").write_text(json.dumps(extrinsics), encoding="utf-8")
            manager = ViewerSessionManager(temp_root=root / "viewer")
            session = manager.create("episode-id", episode)
            self._wait(session)
            self.assertEqual(session.payload()["modes"]["pointcloud"]["status"], "idle")
            manager.prepare_mode(session.session_id, "pointcloud")
            self._wait(session, "pointcloud")
            self.assertEqual(session.payload()["modes"]["pointcloud"]["status"], "ready")
            path, content_type = manager.media_path(session.session_id, "pointcloud", "cloud", 0)
            self.assertEqual(content_type, "application/octet-stream")
            data = path.read_bytes()
            frame, count = __import__("struct").unpack("<II", data[:8])
            self.assertEqual(frame, 0)
            self.assertEqual(count, 6)
            self.assertEqual(len(data), 8 + count * 16)
            manager.close_all()

    def test_viewer_page_contains_four_modes_and_cleanup_beacon(self) -> None:
        page = render_viewer_page("episode<script>")
        self.assertIn("6 路 RGB", page)
        self.assertIn("Pico + 眼动", page)
        self.assertIn("彩色融合点云", page)
        self.assertIn("MANO mesh 视频", page)
        self.assertIn("sendBeacon", page)
        self.assertIn("renderSix", page)
        self.assertIn("img.frame:not(.active)", page)
        self.assertNotIn("$('#grid').innerHTML=sources.map(s=>`<div class=\"tile\"><img src=", page)
        self.assertNotIn("episode<script></div>", page)

    def test_pico_world_gaze_projection_matches_native_center_projection(self) -> None:
        row = {
            "gaze_valid": "true",
            "xr_head_valid": "true",
            "gaze_world_direction_x": "0",
            "gaze_world_direction_y": "0",
            "gaze_world_direction_z": "1",
            "eye_pose_position_unity_x": "0",
            "eye_pose_position_unity_y": "0",
            "eye_pose_position_unity_z": "0",
            "xr_head_pos_x": "0",
            "xr_head_pos_y": "0",
            "xr_head_pos_z": "0",
            "xr_head_rot_x": "0",
            "xr_head_rot_y": "0",
            "xr_head_rot_z": "0",
            "xr_head_rot_w": "1",
            "width": "1280",
            "height": "960",
        }
        params = {"undistort_intrinsic": {"fx": 600, "fy": 600, "cx": 640, "cy": 480}}
        pixel = ViewerSessionManager._project_gaze(
            row,
            params,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            (1280, 960),
        )
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], 640.0)  # type: ignore[index]
        self.assertAlmostEqual(pixel[1], 480.0)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
