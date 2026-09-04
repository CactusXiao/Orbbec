from __future__ import annotations

import unittest

from label.backend_client import episode_display_id as label_episode_display_id
from src.qc.backend import episode_display_id as qc_episode_display_id


class EpisodeDisplayIdTest(unittest.TestCase):
    def test_numeric_index_is_preferred_over_uuid(self) -> None:
        value = {"episode_index": 7, "storage_name": "episode7", "episode_id": "backend-uuid"}
        self.assertEqual(label_episode_display_id(value), "7")
        self.assertEqual(qc_episode_display_id(value), "7")

    def test_storage_name_is_the_only_fallback(self) -> None:
        value = {"storage_name": "episode8", "episode_id": "backend-uuid"}
        self.assertEqual(label_episode_display_id(value), "episode8")
        self.assertEqual(qc_episode_display_id(value), "episode8")

    def test_uuid_is_never_shown_as_a_display_id(self) -> None:
        value = {"episode_id": "backend-uuid"}
        self.assertEqual(label_episode_display_id(value), "-")
        self.assertEqual(qc_episode_display_id(value), "-")

    def test_later_enriched_source_can_supply_index(self) -> None:
        job = {"episode_id": "backend-uuid"}
        payload = {"episode_index": 9}
        self.assertEqual(label_episode_display_id(job, payload), "9")
        self.assertEqual(qc_episode_display_id(job, payload), "9")


if __name__ == "__main__":
    unittest.main()
