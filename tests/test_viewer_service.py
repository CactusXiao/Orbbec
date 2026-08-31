from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from task_backend.viewer_service import ViewerSession, ViewerSessionManager, ViewerSource, render_viewer_page


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

    def test_viewer_page_contains_five_modes_and_cleanup_beacon(self) -> None:
        page = render_viewer_page("episode<script>")
        self.assertIn("6 路 RGB", page)
        self.assertIn("Pico + 眼动", page)
        self.assertIn("彩色融合点云", page)
        self.assertIn("MANO mesh 视频", page)
        self.assertIn("Pico 手部 Pose", page)
        self.assertIn('data-mode="picohand"', page)
        self.assertIn("sendBeacon", page)
        self.assertIn("renderSix", page)
        self.assertIn("img.frame:not(.active)", page)
        self.assertIn('id="badRanges"', page)
        self.assertIn('id="manoSource"', page)
        self.assertNotIn("$('#grid').innerHTML=sources.map(s=>`<div class=\"tile\"><img src=", page)
        self.assertNotIn("episode<script></div>", page)

    def test_picohand_uses_aligned_pico_rgb_and_qc_preview_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / "episode"
            (episode / "optimized_pose").mkdir(parents=True)
            (episode / "ego_pose.json").write_text("{}", encoding="utf-8")
            decoded = root / "decoded" / "ego" / "RGB"
            decoded.mkdir(parents=True)
            (decoded / "00007.jpg").write_bytes(PNG_1X1)
            session = ViewerSession(
                session_id="test",
                episode_id="episode",
                episode_dir=episode,
                temp_dir=root / "viewer",
                ffmpeg="ffmpeg",
                state="ready",
                frame_indices=[42],
                sources=[
                    ViewerSource(
                        source_id="ego",
                        label="Pico",
                        kind="ego",
                        camera="ego",
                        rgb_frame_map={42: 7},
                        decoded_rgb=decoded,
                    )
                ],
            )
            session.temp_dir.mkdir()
            manager = ViewerSessionManager(
                temp_root=root / "sessions",
                mano_toolkit_root=root / "mano-toolkit",
                mano_model_dir=root / "mano-model",
            )
            try:
                with patch.object(manager, "_run_mesh_renderer") as render:
                    manager._prepare_picohand(session, session.modes["picohand"])
                linked = session.temp_dir / "picohand_rgb" / "ego" / "00042.jpg"
                self.assertTrue(linked.is_file())
                kwargs = render.call_args.kwargs
                self.assertEqual(kwargs["cameras"], ["ego"])
                self.assertEqual(kwargs["frames"], [42])
                self.assertEqual(kwargs["available_dir"], session.temp_dir / "modes" / "picohand")
                self.assertEqual(kwargs["preview_output_dir"], session.temp_dir / "modes" / "picohand")
            finally:
                manager.shutdown()

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
        params = {
            "width": 1280,
            "height": 960,
            "undistort": {"new_intrinsic": {"fx": 600, "fy": 600, "cx": 640, "cy": 480}},
        }
        pixel = ViewerSessionManager._project_gaze(
            row,
            params,
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            (1280, 960),
        )
        self.assertIsNotNone(pixel)
        self.assertAlmostEqual(pixel[0], 640.0)  # type: ignore[index]
        self.assertAlmostEqual(pixel[1], 480.0)  # type: ignore[index]

    def test_mano_context_marks_qc_ranges_and_detects_corrected_pose_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            episode = Path(tmp)
            (episode / "qc").mkdir()
            (episode / "qc" / "qc_report.json").write_text(
                json.dumps(
                    {
                        "kind": "orbbec_qc_report",
                        "passed": False,
                        "segments": [{"start_frame": 1, "end_frame": 2}],
                    }
                ),
                encoding="utf-8",
            )
            manual = episode / "manual_2d" / "segments" / "job" / "00"
            poses = episode / "optimized_pose"
            manual.mkdir(parents=True)
            poses.mkdir()
            for frame in (1, 2):
                manual_path = manual / f"{frame:05d}.npy"
                pose_path = poses / f"{frame:05d}.npy"
                manual_path.write_bytes(b"manual")
                pose_path.write_bytes(b"pose")
                os.utime(manual_path, (20, 20))
                os.utime(pose_path, (10, 10))

            initial = ViewerSessionManager._mano_context(episode, [0, 1, 2])
            self.assertEqual(initial["bad_ranges"], [{"start_frame": 1, "end_frame": 2}])
            self.assertEqual(initial["source"], "auto_label")

            for frame in (1, 2):
                os.utime(poses / f"{frame:05d}.npy", (30, 30))
            corrected = ViewerSessionManager._mano_context(episode, [0, 1, 2])
            self.assertEqual(corrected["source"], "corrected_3d")


if __name__ == "__main__":
    unittest.main()
