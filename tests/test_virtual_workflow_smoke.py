from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.virtual_workflow.orbbec_virtual_workflow import (
    NasSimulator,
    cameras_from_payload,
    frames_from_payload,
    write_corrected_artifact_for_payload,
    write_prediction_artifact_for_payload,
)


class FixtureDetector:
    available = True

    def detect_values(self, *_args, **_kwargs):
        return [10.0, 12.0] * (2 * 21)


class VirtualWorkflowSmokeTest(unittest.TestCase):
    def test_virtual_workers_infer_payload_context_from_nas_episode_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nas = NasSimulator(tmp_path / "nas", "nas://ego")
            payload = {
                "episode_uri": "nas://ego/S001/pick_object/episode_001",
                "episode_id": "episode_001",
                "segment_id": "segment_001",
            }
            episode_dir = nas.nas_path_for_uri(payload["episode_uri"])
            for camera in ("00", "01"):
                rgb = episode_dir / camera / "RGB"
                rgb.mkdir(parents=True)
                (rgb / "00001.png").write_bytes(b"rgb")
                (rgb / "00002.png").write_bytes(b"rgb")

            cameras = cameras_from_payload(payload, {}, nas)
            frames = frames_from_payload(payload, {}, nas, cameras)
            self.assertEqual(cameras, ["00", "01"])
            self.assertEqual(frames, [1, 2])

            pred_uri = write_prediction_artifact_for_payload(nas, payload, {}, cameras, frames, FixtureDetector())  # type: ignore[arg-type]
            corrected_uri = write_corrected_artifact_for_payload(nas, payload, {}, cameras, frames)
            self.assertTrue((episode_dir / "pred_2d" / "00" / "00001.npy").exists())
            self.assertTrue((episode_dir / "manual_2d" / "segments" / "segment_001" / "01" / "00002.npy").exists())
            self.assertTrue(pred_uri.endswith("/pred_2d"))
            self.assertTrue(corrected_uri.endswith("/manual_2d/segments/segment_001"))


if __name__ == "__main__":
    unittest.main()
