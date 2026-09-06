from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from task_backend.job_service import JobService
from task_backend.publisher_bridge import PublisherBridge, PublisherBridgeConfig
from task_backend.server import TaskBackend, render_episode_detail
from task_backend.workflow_models import WorkflowError
from task_backend.workflow_store import WorkflowStore


class HandshapeWorkflowTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = WorkflowStore(self.root / 'workflow.sqlite3')
        self.service = JobService(self.store, nas_mounts={'nas://ego': str(self.root / 'nas')})
        self.body = {'subject_id': 'S001', 'capture_token': 'one',
                     'episode_uri': 'nas://ego/S001/task_handshapeCalibration/episode1'}
        registered = self.service.register_shape_calibration(self.body)
        self.job = self.store.get_job(registered["job"]["job_id"])
        self.episode_id = self.job['episode_id']

    def results(self):
        directory = self.root / 'nas/S001/shape_calibration_result'
        directory.mkdir(parents=True)
        for name in ('shape.npy', 'scale.npy', 'pose_mesh.png', 'pose_2d.png'):
            (directory / name).write_bytes(b'publisher-verified-fixture')
        return directory

    def finish(self):
        return self.service.complete_job(self.job['job_id'], {
            'result': {'publisher_state': 'shape_calibrated'},
            'artifacts': [{'kind': 'shape_calibration_result', 'uri': 'nas://ego/S001/shape_calibration_result'}],
        })

    def test_registration_is_idempotent_and_has_no_capture_quota_or_upload_job(self):
        same = self.service.register_shape_calibration(self.body)
        self.assertEqual(same['job']['job_id'], self.job['job_id'])
        self.assertEqual([j['type'] for j in self.store.jobs_for_episode(self.episode_id)], ['auto_label'])

    def test_completion_stops_at_stage_three_without_fake_pose_or_qc(self):
        self.results()
        self.finish()
        self.finish()
        episode = self.store.get_episode(self.episode_id)
        self.assertEqual(episode['status'], 'mano_optimized')
        self.assertTrue(episode['metadata']['tracking_complete'])
        self.assertEqual([j['type'] for j in self.store.jobs_for_episode(self.episode_id)], ['auto_label'])
        self.assertEqual({a['kind'] for a in self.store.artifacts_for_episode(self.episode_id)}, {'shape_calibration_result'})
        self.assertFalse((self.root / 'nas/S001/task_handshapeCalibration/episode1/mano').exists())

    def test_missing_results_cannot_complete(self):
        with self.assertRaisesRegex(WorkflowError, 'incomplete'):
            self.finish()
        self.assertEqual(self.store.get_job(self.job['job_id'])['status'], 'queued')

    def test_downstream_job_creation_is_rejected(self):
        for job_type in ('qc', 'review', 'manual_label', 'manual_3d'):
            with self.subTest(job_type=job_type), self.assertRaises(WorkflowError):
                self.service._create_job_once(job_id=job_type, job_type=job_type,
                                             episode_id=self.episode_id, payload={})
        with self.assertRaises(WorkflowError):
            self.service.create_manual_label_job({'episode_id': self.episode_id, 'episode_uri': self.body['episode_uri']})
        self.assertEqual(len(self.store.jobs_for_episode(self.episode_id)), 1)

    def test_bridge_returns_after_shape_result_without_materialization_or_publish(self):
        self.results()
        publisher = Mock()
        publisher.status.return_value = {'found': True, 'state': 'shape_calibrated', 'generation': 1}
        materializer = Mock()
        bridge = PublisherBridge(self.service, PublisherBridgeConfig(enabled=True),
                                 publisher_client=publisher, materializer=materializer)
        episode = self.store.get_episode(self.episode_id)
        self.assertTrue(bridge._process_leased_job(self.job['job_id'], 'publisher_bridge:test', self.job['payload'], episode))
        publisher.status.assert_called_once()
        publisher.publish.assert_not_called()
        materializer.run.assert_not_called()
        self.assertEqual(self.store.get_job(self.job['job_id'])['status'], 'succeeded')

    def test_previous_completion_cannot_finish_new_calibration(self):
        self.results()
        self.finish()
        next_job = self.service.register_shape_calibration({**self.body, "capture_token": "two"})
        self.assertNotEqual(next_job["job"]["job_id"], self.job["job_id"])
        with self.assertRaisesRegex(WorkflowError, "stale"):
            self.finish()
        self.assertFalse(self.store.get_episode(self.episode_id)["metadata"]["tracking_complete"])

    def test_cannot_replace_an_active_calibration(self):
        with self.assertRaisesRegex(WorkflowError, "still running"):
            self.service.register_shape_calibration({**self.body, "capture_token": "two"})

    def test_detail_page_ends_at_stage_three(self):
        self.results()
        self.finish()
        task_file = self.root / 'tasks.json'
        task_file.write_text(json.dumps({'tasks': [{'task_name': 'normal', 'total': 2}]}))
        backend = TaskBackend(self.root / 'state', task_file, workflow_service=self.service)
        html = render_episode_detail(backend.episode_detail_model(self.episode_id))
        self.assertIn('3. 自动标注 + Episode 3D', html)
        self.assertIn('手型标定结果已回传，流程结束', html)
        self.assertNotIn('4. QC 质检', html)
        self.assertNotIn('5. Episode 人工纠偏', html)


if __name__ == '__main__':
    unittest.main()
