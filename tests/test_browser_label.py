import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from remote_frontend.browser_label import BrowserCorrectionTask, BrowserLabelRuntime, decode_label
from label.storage import CorrectionTask
from label.mano_view import CameraParams
from tests import test_browser_batch


class BrowserEgoTest(unittest.TestCase):
    def test_aligned_decode_preserves_reference_indices_and_desktop_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cam in ['00', 'ego']:
                (root/cam/'RGB').mkdir(parents=True)
            for f in [10,11]:
                Image.new('RGB',(8,8),'blue').save(root/'00/RGB'/f'{f:05d}.png')
            Image.new('RGB',(8,8),'red').save(root/'ego/RGB/00002.png')
            (root/'timestamps.csv').write_text('frame_index,ego_frame_index\n10,2\n11,2\n')
            task=BrowserCorrectionTask(1,tmp,'s','t','e',['00'],[10,11],nas_root_path=tmp)
            self.assertEqual(task.cameras,['00','ego'])
            self.assertEqual(replace(task,frames=[10]).cameras,['00','ego'])
            desktop=CorrectionTask(1,tmp,'s','t','e',['00','ego'],[10],nas_root_path=tmp)
            self.assertEqual(desktop.cameras,['00'])
            decoded=decode_label(task,{},cache_root=root/'cache')
            self.assertEqual(decoded.cameras,['00','ego'])
            for f in task.frames:
                path=Path(decoded.rgb_path_template.format(camera='ego',frame=f))
                self.assertEqual(Image.open(path).getpixel((0,0)),(255,0,0))
            cached = root/'cache/aligned/ego/00010.jpg'
            original = (cached.read_bytes(), cached.stat().st_mtime_ns)
            with patch('remote_frontend.browser_label.ensure_decoded_rgb_frames') as fixed_decode, \
                 patch('label.ego_preview.EgoPreview') as ego_decode:
                reused = decode_label(task,{},cache_root=root/'cache')
                fixed_decode.assert_not_called()
                ego_decode.assert_not_called()
                self.assertEqual(reused.rgb_path_template, decoded.rgb_path_template)
                self.assertEqual((cached.read_bytes(), cached.stat().st_mtime_ns), original)

            # A missing Ego frame repairs only that frame and keeps fixed RGB cached.
            from label.ego_preview import EgoPreview
            (root/'cache/aligned/ego/00011.jpg').unlink()
            with patch('remote_frontend.browser_label.ensure_decoded_rgb_frames') as fixed_decode, \
                 patch('label.ego_preview.EgoPreview', wraps=EgoPreview) as ego_decode:
                decode_label(task,{},cache_root=root/'cache')
                fixed_decode.assert_not_called()
                self.assertEqual(ego_decode.call_args.args[0].frames, [11])
                self.assertEqual((cached.read_bytes(), cached.stat().st_mtime_ns), original)
                self.assertTrue((root/'cache/aligned/ego/00011.jpg').is_file())

            import threading
            stopped = threading.Event(); stopped.set()
            with self.assertRaises(InterruptedError):
                decode_label(task,{},cache_root=root/'cache',stop_event=stopped)

    def test_ego_is_required_in_new_batch_and_saved_with_visibility(self):
        fixture=test_browser_batch.BrowserBatchTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        (fixture.root/'nas/subject/label/episode1/ego/RGB').mkdir(parents=True)
        item,episode=fixture.create('label')
        self.assertIn('ego',fixture.batch.manifest(item['id'])['cameras'])
        body=fixture.body(item)
        from task_backend.workflow_models import WorkflowError
        with self.assertRaises(WorkflowError): fixture.batch.submit(item['id'],body)
        for f in range(3):
            sample=copy.deepcopy(body['result']['samples'][f'{f}:00'])
            sample['points'][0][0]=[123,234]
            sample['visible'][0][1]=False
            body['result']['samples'][f'{f}:ego']=sample
        fixture.batch.submit(item['id'],body)
        saved=list((episode/'manual_2d').glob('**/ego/00000.npy'))
        self.assertEqual(len(saved),1)
        self.assertEqual(np.load(saved[0])[0,0].tolist(),[123,234])
        masks=list((episode/'manual_joints_vis').glob('**/ego/00000.npy'))
        self.assertEqual(int(np.load(masks[0])[0,1]),0)

    def test_ego_projection_uses_frame_extrinsics_and_fisheye(self):
        try: import cv2
        except ImportError: self.skipTest('OpenCV required for geometric integration')
        runtime=BrowserLabelRuntime()
        k=np.array([[400.,0,320],[0,400,240],[0,0,1]])
        transform=np.eye(4);transform[0,3]=.1
        runtime.ego_calibration['episode']=(k,np.zeros((4,1)),(640,480),{0:np.eye(4),1:transform})
        joints=np.zeros((2,21,3));joints[:,:,2]=1
        uv,vis=runtime.project_skeleton(episode_dir=Path('episode'),cam_id='ego',joints_3d=joints)
        self.assertEqual(uv[0][0],[320,240]);self.assertTrue(vis[0][0])
        runtime.frame=1
        moved,_=runtime.project_skeleton(episode_dir=Path('episode'),cam_id='ego',joints_3d=joints)
        self.assertAlmostEqual(moved[0][0][0],320+400*np.arctan(.1),places=4)
        rgb=CameraParams(k,np.zeros(5),np.eye(3),np.zeros(3))
        with patch('remote_frontend.browser_label.load_episode_cameras',return_value={'00':rgb}):
            recovered=runtime.build_skeleton(episode_dir=Path('episode'),camera_ids=['00','ego'],view_states={'00':(uv,vis),'ego':(moved,vis)})
        np.testing.assert_allclose(recovered,joints,atol=1e-5)

    def test_tracking_rejects_cross_interval_targets(self):
        from types import SimpleNamespace
        from remote_frontend.browser_compute import BrowserCompute
        from task_backend.workflow_models import WorkflowError
        fixture=test_browser_batch.BrowserBatchTest();fixture.setUp();self.addCleanup(fixture.doCleanups)
        item,_=fixture.create('label')
        task=BrowserCorrectionTask(1,'/tmp','s','t','e',['00'],[0,1,2],segments=[dict(start_frame=0,end_frame=1),dict(start_frame=2,end_frame=2)])
        compute=BrowserCompute(fixture.batch,SimpleNamespace(directory=lambda sid:fixture.root/sid),{})
        self.addCleanup(compute.pool.shutdown)
        sample=fixture.body(item)['result']['samples']['0:00'];sample.update(width=640,height=480)
        with patch('remote_frontend.browser_compute.browser_task',return_value=task),patch.object(compute.pool,'submit') as submit:
            with self.assertRaises(WorkflowError):compute.start(item['id'],dict(action='track',frame=1,target=2,samples={'00':sample},selected={'00':[[0,0]]}))
            submit.assert_not_called()

if __name__=='__main__':unittest.main()
