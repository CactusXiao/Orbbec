from __future__ import annotations

import importlib.machinery
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


STATUS_SCRIPT = Path(__file__).resolve().parents[1] / "publisher" / "nas-uploader-status"


def load_status_module():
    loader = importlib.machinery.SourceFileLoader("nas_uploader_status", str(STATUS_SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("cannot load nas-uploader-status")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class PublisherStatusTest(unittest.TestCase):
    def test_read_only_episode_query_includes_quality_queue_state(self) -> None:
        module = load_status_module()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "episodes.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE episodes (
                        episode_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        result_manifest_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE quality_queue (
                        episode_id TEXT PRIMARY KEY,
                        queue_state TEXT NOT NULL
                    );
                    INSERT INTO episodes VALUES (
                        'S001/task1/episode1', 'labeled', '2026-08-27T00:00:00Z', 2,
                        'source-sha', 'result-sha'
                    );
                    INSERT INTO quality_queue VALUES ('S001/task1/episode1', 'pending');
                    """
                )

            result = module.query_episode(db_path, "S001/task1/episode1")

            self.assertTrue(result["found"])
            self.assertEqual(result["state"], "labeled")
            self.assertEqual(result["generation"], 2)
            self.assertEqual(result["quality_queue_state"], "pending")
            self.assertEqual(module.query_episode(db_path, "S001/task1/missing")["found"], False)

    def test_episode_id_requires_exactly_three_safe_parts(self) -> None:
        module = load_status_module()
        self.assertEqual(module.validate_episode_id("S001/task1/episode1"), "S001/task1/episode1")
        for invalid in ("S001/task1", "/S001/task1/episode1", "S001/../episode1"):
            with self.assertRaises(ValueError):
                module.validate_episode_id(invalid)

    def test_legacy_quality_queue_row_is_reported_as_pending(self) -> None:
        module = load_status_module()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "episodes.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE episodes (
                        episode_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        result_manifest_sha256 TEXT NOT NULL
                    );
                    CREATE TABLE quality_queue (
                        episode_id TEXT PRIMARY KEY,
                        generation INTEGER NOT NULL,
                        enqueued_at TEXT NOT NULL
                    );
                    INSERT INTO episodes VALUES (
                        'S001/task1/episode1', 'labeled', '2026-08-27T00:00:00Z', 2,
                        'source-sha', 'result-sha'
                    );
                    INSERT INTO quality_queue VALUES (
                        'S001/task1/episode1', 2, '2026-08-27T00:00:00Z'
                    );
                    """
                )

            result = module.query_episode(db_path, "S001/task1/episode1")

            self.assertTrue(result["found"])
            self.assertEqual(result["quality_queue_state"], "pending")

    def test_query_works_without_quality_queue_table(self) -> None:
        module = load_status_module()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "episodes.sqlite3"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE episodes (
                        episode_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        result_manifest_sha256 TEXT NOT NULL
                    );
                    INSERT INTO episodes VALUES (
                        'S001/task1/episode1', 'uploading', '2026-08-27T00:00:00Z', 1,
                        'source-sha', ''
                    );
                    """
                )

            result = module.query_episode(db_path, "S001/task1/episode1")

            self.assertTrue(result["found"])
            self.assertIsNone(result["quality_queue_state"])


if __name__ == "__main__":
    unittest.main()
