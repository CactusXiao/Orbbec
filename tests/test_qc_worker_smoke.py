from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.qc.state_store import QcProgress, QcStateStore, first_sample_after, normalize_ranges


class QcWorkerSmokeTest(unittest.TestCase):
    def test_normalize_bad_frame_ranges_merges_overlap_touching_and_small_gaps(self) -> None:
        ranges = normalize_ranges([(20, 25), (10, 12), (13, 14), (30, 31)], max_gap_frames=5)

        self.assertEqual(ranges, [(10, 14), (20, 31)])

    def test_first_sample_after_uses_sampling_sequence(self) -> None:
        self.assertEqual(first_sample_after(147, first_frame=0, last_frame=300, sample_interval=10), 150)
        self.assertEqual(first_sample_after(150, first_frame=0, last_frame=300, sample_interval=10), 160)
        self.assertEqual(first_sample_after(297, first_frame=0, last_frame=300, sample_interval=10), 300)
        self.assertEqual(first_sample_after(300, first_frame=0, last_frame=300, sample_interval=10), 301)

    def test_state_store_round_trips_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = QcStateStore(Path(tmp))
            progress = QcProgress(
                task_name="pick_object",
                episode_id="episode_001",
                job_id="qc_episode_001",
                worker_machine_id="worker_a",
                lease_until="2099-01-01T00:00:00Z",
                sample_interval=10,
                current_frame=20,
                frames=[0, 10, 20],
                bad_frame_ranges=[(12, 15)],
            )

            store.save(progress)
            loaded = store.list_progress(worker_machine_id="worker_a")

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].episode_id, "episode_001")
            self.assertEqual(loaded[0].bad_frame_ranges, [(12, 15)])


if __name__ == "__main__":
    unittest.main()
