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


class VirtualWorkflowSmokeTest(unittest.TestCase):
    def test_virtual_workers_infer_payload_context_from_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            episode_dir = tmp_path / "S001" / "pick_object" / "episode_001"
            for camera in ("00", "01"):
                rgb = episode_dir / camera / "RGB"
                rgb.mkdir(parents=True)
                (rgb / "00001.png").write_bytes(b"rgb")
                (rgb / "00002.png").write_bytes(b"rgb")

            nas = NasSimulator(tmp_path / "virtual_nas")
            payload = {
                "data_uri": "local://" + str(episode_dir),
                "resolved_data_path": str(episode_dir),
                "prediction_dir": "auto_pred",
                "correction_dir": "human_fixed",
            }
            cameras = cameras_from_payload(payload, {}, nas)
            frames = frames_from_payload(payload, {}, nas, cameras)
            self.assertEqual(cameras, ["00", "01"])
            self.assertEqual(frames, [1, 2])

            pred_uri = write_prediction_artifact_for_payload(nas, payload, {}, cameras, frames, "auto_pred")
            corrected_uri = write_corrected_artifact_for_payload(nas, payload, {}, cameras, frames, "human_fixed")
            self.assertTrue((episode_dir / "auto_pred" / "00" / "00001.npy").exists())
            self.assertTrue((episode_dir / "human_fixed" / "01" / "00002.npy").exists())
            self.assertTrue(pred_uri.endswith("/auto_pred"))
            self.assertTrue(corrected_uri.endswith("/human_fixed"))


if __name__ == "__main__":
    unittest.main()
