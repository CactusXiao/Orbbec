from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from remote_frontend.batch import BatchService
from task_backend.job_service import JobService
from task_backend.workflow_store import WorkflowStore
from task_backend.workflow_models import WorkflowError


class BrowserBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'workflow.sqlite3'
        self.mounts = {'nas://test': str(self.root/'nas')}
        self.desktop = JobService(WorkflowStore(self.db), nas_mounts=self.mounts)
        self.batch = BatchService(self.db, self.mounts, 'alice')

    def create(self, role):
        episode = self.root/'nas'/'subject'/role/'episode1'
        (episode/'00'/'RGB').mkdir(parents=True)
        self.desktop.store.create_or_update_episode(episode_id=role, subject_id='test', task_name=role,
            episode_index=1, status='auto_labeled' if role=='qc' else 'manual_correction_pending',
            episode_uri=f'nas://test/subject/{role}/episode1', cameras=['00'], frame_count=3)
        if role=='qc':
            self.desktop.create_dev_job({'type':'qc','episode_id':role,'payload':{'frames':[0,1,2]}})
        else:
            self.desktop.store.create_segment(segment_id='seg',episode_id=role,start_frame=0,end_frame=2)
            self.desktop._create_manual_label_episode_job(role, reason='test')
        job = self.batch.available(role)[0]
        session = self.batch.lease(role, job['job_id'])
        self.batch.heartbeat(session['id'])
        return session, episode

    def body(self, item):
        if item['role']=='label':
            samples={f'{f}:00':{'points':np.zeros((2,21,2)).tolist(), 'visible':np.ones((2,21),dtype=bool).tolist()} for f in range(3)}
            result={'confirmed':[0,1,2],'samples':samples}
        else:
            result={'reviewed':[0,1,2],'bad_ranges':[],'ego_ranges':[[1,2]],'bad_episode':False}
        return {'submission_id':str(uuid.uuid4()),'revision':item['revision'],'result':result}

    def test_qc_primary_camera_reaches_label_manifest(self):
        item, episode = self.create('qc')
        body = self.body(item)
        body['result'].update(bad_ranges=[[0, 1]], bad_segments=[dict(start_frame=0, end_frame=1, primary_camera='00')],
                              ego_segments=[dict(start_frame=1, end_frame=2, primary_camera='ego')])
        self.batch.submit(item['id'], body)
        report = json.loads((episode / 'qc/qc_report.json').read_text())
        self.assertEqual(report['segments'][0]['primary_camera'], '00')
        ego = json.loads((episode / 'ego/ego_pose_qc.json').read_text())
        self.assertEqual(ego['segments'][0]['primary_camera'], 'ego')
        job = self.batch.available('label')[0]
        label = self.batch.lease('label', job['job_id'])
        manifest = self.batch.manifest(label['id'])
        self.assertEqual(manifest['qc_segments'][0]['primary_camera'], '00')

    def test_qc_rejects_invalid_primary_camera_or_mismatched_ranges(self):
        item, _ = self.create('qc')
        for segment in [dict(start_frame=0, end_frame=1, primary_camera='missing'),
                        dict(start_frame=1, end_frame=2, primary_camera='00')]:
            body = self.body(item)
            body['result'].update(bad_ranges=[[0, 1]], bad_segments=[segment])
            with self.assertRaises(WorkflowError):
                self.batch.submit(item['id'], body)

    def test_calculation_validates_scope_before_scheduling(self):
        from types import SimpleNamespace
        from remote_frontend.browser_compute import BrowserCompute
        item, episode = self.create('label')
        media = SimpleNamespace(directory=lambda sid: self.root / 'browser-cache' / sid)
        compute = BrowserCompute(self.batch, media, {})
        self.addCleanup(compute.pool.shutdown)
        sample = self.body(item)['result']['samples']['0:00']
        sample.update(width=640, height=480)
        good = dict(action='track', frame=0, target=1, selected={'00':[[0,0]]}, samples={'00':sample})
        invalid = [dict(good, frame=9), dict(good, target=-1), dict(good, selected={'../escape':[[0,0]]}),
                   dict(good, selected={'00':[[0,21]]}), dict(good, action='run_shell')]
        with patch.object(compute.pool, 'submit') as submit:
            for body in invalid:
                with self.assertRaises(WorkflowError): compute.start(item['id'], body)
            submit.assert_not_called()
            result = compute.start(item['id'], dict(good, episode='/untrusted/path', template='/untrusted/template'))
            args = submit.call_args.args
            self.assertEqual(args[4].episode_dir().resolve(), episode.resolve())
            self.assertNotIn('episode', args[5])
            self.assertNotIn('template', args[5])
            with self.assertRaises(WorkflowError): compute.start(item['id'], good)
        self.assertFalse((episode/'manual_2d').exists())
        self.assertFalse(compute.status(item['id'], result['id'])['ready'])
        with self.assertRaises(WorkflowError): compute.directory(item['id'], '../escape')

    def test_skeleton_worker_only_produces_preview_in_cache(self):
        from remote_frontend.compute_worker import run
        from label.mano_view import ManoViewRuntime
        item, episode = self.create('label')
        sample = self.body(item)['result']['samples']['0:00']
        sample.update(width=640, height=480)
        folder = self.root/'preview'; folder.mkdir()
        body = dict(action='skeleton', episode=str(episode), samples={'00':sample})
        with patch.object(ManoViewRuntime,'build_skeleton',return_value=np.ones((2,21,3))), \
             patch.object(ManoViewRuntime,'project_skeleton',return_value=(sample['points'],sample['visible'])):
            self.assertEqual(run(body,folder)['cameras'],['00'])
        self.assertTrue((folder/'00.png').is_file())
        self.assertFalse((episode/'manual_2d').exists())

    def test_missing_reference_keeps_existing_annotation_editable(self):
        from remote_frontend.browser_media import label_states
        from label.storage import correction_task_from_backend_payload
        from label.mano_view import ManoViewRuntime
        item, episode = self.create('label')
        task = correction_task_from_backend_payload(item['payload'], mounts=self.mounts)
        points = np.ones((2,21,2)).tolist(); points[0][0] = [123,456]
        visible = np.ones((2,21),dtype=bool).tolist()
        with patch.object(ManoViewRuntime,'project_mano_frame',return_value=(points,visible)), \
             patch('remote_frontend.browser_media.load_joint_visibility',return_value=None), \
             patch('remote_frontend.browser_media.source_frame_path',return_value=episode/'saved.npy'), \
             patch('remote_frontend.browser_media.view_state_from_bundle',return_value=(points,visible)):
            saved, mask, refs = label_states(ManoViewRuntime(),task,None,0,'00')
        self.assertEqual(saved[0][0],[123,456])
        self.assertTrue(mask[0][0])
        self.assertIn('mano_visible', refs['errors'])
        self.assertEqual(refs['mano_visible']['points'][0][0],[-1,-1])
        self.assertEqual(refs['mano']['points'][0][0],[123,456])
        with patch.object(ManoViewRuntime,'project_mano_frame',return_value=None), \
             patch('remote_frontend.browser_media.source_frame_path',return_value=None):
            saved, mask, refs = label_states(ManoViewRuntime(),task,None,0,'00')
        self.assertFalse(np.asarray(mask).any())
        self.assertTrue(np.isfinite(saved).all())
        self.assertIn('mano',refs['errors'])

    def test_start_does_not_reset_desktop_controls_or_recover_running_outbox(self):
        self.create('qc')
        with self.desktop.store.connect() as conn:
            conn.execute("UPDATE nas_sync_outbox SET status='running'")
        self.desktop.store.set_stage_control(job_type='auto_label',lease_enabled=False,updated_by='system',note='paused')
        with self.desktop.store.connect() as conn:
            before = [tuple(r) for r in conn.execute('SELECT * FROM workflow_stage_controls')]
        BatchService(self.db,self.mounts,'bob')
        with self.desktop.store.connect() as conn:
            self.assertEqual(before,[tuple(r) for r in conn.execute('SELECT * FROM workflow_stage_controls')])
            self.assertEqual(conn.execute('SELECT status FROM nas_sync_outbox').fetchone()[0], 'running')

    def test_label_writes_only_on_submit_then_one_downstream_job_on_retry(self):
        item,episode=self.create('label')
        self.assertFalse((episode/'manual_2d').exists())
        body=self.body(item)
        body['result']['samples']['0:00']['points'][0][0]=[321.5,219.25]
        receipt=self.batch.submit(item['id'],body)
        self.assertEqual(receipt,self.batch.submit(item['id'],body))
        coords=list((episode/'manual_2d').rglob('*.npy'))
        self.assertEqual(len(coords),3)
        self.assertTrue(any(np.array_equal(np.load(p)[0,0],[321.5,219.25]) for p in coords))
        self.assertEqual(len(list((episode/'manual_joints_vis').rglob('*.npy'))),3)
        jobs=self.desktop.store.jobs_for_episode('label')
        self.assertEqual(sum(j['type']=='manual_3d' for j in jobs),1)
        changed=copy.deepcopy(body);changed['result']['samples']['0:00']['points'][0][0][0]+=1
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],changed)

    def test_legacy_visibility_encoding_remains_compatible(self):
        from types import SimpleNamespace
        from remote_frontend.batch import correction_task_from_backend_payload
        item,episode=self.create('label');body=self.body(item)
        body['result']['samples']['0:00']['visible'][0][0]=False
        body['result']['samples']['0:00']['points'][0][0]=[100,200]
        actual=correction_task_from_backend_payload(item['payload'],mounts=self.mounts)
        legacy=SimpleNamespace(frames=actual.frames,cameras=actual.cameras,
            correction_dir=actual.correction_dir,episode_dir=actual.episode_dir)
        with patch('remote_frontend.batch.correction_task_from_backend_payload',return_value=legacy):
            self.batch.submit(item['id'],body)
        point_file=episode/actual.correction_dir/'00/00000.npy'
        self.assertTrue(np.array_equal(np.load(point_file)[0,0],[-1,-1]))
        visibility_file=episode/'manual_joints_vis/segments'/item['job_id']/'00/00000.npy'
        self.assertEqual(np.load(visibility_file)[0,0],0)

    def test_qc_result_preserves_separate_ego_report(self):
        item,episode=self.create('qc')
        self.assertFalse((episode/'qc').exists())
        self.batch.submit(item['id'],self.body(item))
        self.assertTrue(json.loads((episode/'qc/qc_report.json').read_text())['passed'])
        ego=json.loads((episode/'ego/ego_pose_qc.json').read_text())
        self.assertEqual(ego['segments'],[{'start_frame':1,'end_frame':2}])
        self.assertEqual(self.desktop.store.get_episode('qc')['status'],'finalized')

    def test_invalid_frames_nan_incomplete_camera_or_revision_never_publish(self):
        item,episode=self.create('label')
        bodies=[]
        b=self.body(item);b['result']['confirmed']=[0,1];bodies.append(b)
        b=self.body(item);b['result']['samples']['0:ego']=b['result']['samples'].pop('0:00');bodies.append(b)
        b=self.body(item);b['result']['samples']['0:00']['points'][0][0][0]=float('nan');bodies.append(b)
        b=self.body(item);b['revision']='stale';bodies.append(b)
        for body in bodies:
            with self.assertRaises((WorkflowError,ValueError)):self.batch.submit(item['id'],body)
        self.assertFalse((episode/'manual_2d').exists())

    def test_expired_lease_can_resume_but_cannot_take_over_desktop_owner(self):
        item,episode=self.create('label')
        with self.desktop.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until='2000-01-01T00:00:00Z' WHERE job_id=?",(item['job_id'],))
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],self.body(item))
        self.batch.heartbeat(item['id'])
        with self.desktop.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until='2000-01-01T00:00:00Z' WHERE job_id=?",(item['job_id'],))
        self.desktop.lease_label_episode({'job_id':item['job_id'],'lease_owner':'desktop'})
        with self.assertRaises(WorkflowError):self.batch.heartbeat(item['id'])
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],self.body(item))
        self.assertFalse((episode/'manual_2d').exists())

    def test_updated_source_is_rejected(self):
        item,episode=self.create('label')
        (episode/'camera_params.json').write_text('{}')
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],self.body(item))
        self.assertFalse((episode/'manual_2d').exists())

    def test_failed_transition_rolls_back_job_receipt_and_downstream_queue(self):
        item,episode=self.create('label');body=self.body(item)
        original=self.batch.service._after_job_complete
        def fail(*args):
            original(*args)
            raise RuntimeError('simulated downstream failure')
        with patch.object(self.batch.service,'_after_job_complete',side_effect=fail):
            with self.assertRaises(RuntimeError):self.batch.submit(item['id'],body)
        self.assertEqual(list((episode/'manual_2d').rglob('*.npy')),[])
        self.assertFalse((episode/'workflow/final_3d_sources.json').exists())
        self.assertNotEqual(self.desktop.store.get_job(item['job_id'])['status'],'succeeded')
        self.assertFalse(any(j['type']=='manual_3d' for j in self.desktop.store.jobs_for_episode('label')))
        self.assertTrue(self.batch.submit(item['id'],body)['accepted'])

    def test_concurrent_duplicate_submission_has_one_receipt_and_one_job(self):
        item,_=self.create('label');body=self.body(item)
        with ThreadPoolExecutor(2) as pool:
            receipts=list(pool.map(lambda _:self.batch.submit(item['id'],body),range(2)))
        self.assertEqual(receipts[0],receipts[1])
        self.assertEqual(sum(j['type']=='manual_3d' for j in self.desktop.store.jobs_for_episode('label')),1)


    def test_http_auth_origin_range_and_lost_ack_retry(self):
        import threading
        from http.server import ThreadingHTTPServer
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        from http.client import RemoteDisconnected
        from types import SimpleNamespace
        from remote_frontend.browser_server import Handler
        item,_=self.create('label');body=self.body(item)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        from remote_frontend.browser_login import Accounts
        server.auth=Accounts(self.root/'accounts.sqlite3')
        server.auth.create('alice','test-password-strong-2026',['label','qc'],['*'],operator='alice',must_change=False)
        server.cookie_name='test_auth';server.secure_cookie=False
        server.allowed_hosts={f'127.0.0.1:{server.server_port}'}
        server.allowed_origins={f'http://127.0.0.1:{server.server_port}'}
        server.workbench=lambda user:SimpleNamespace(batch=self.batch, media=SimpleNamespace(directory=lambda sid:self.root), compute=None)
        (self.root/'preview.mp4').write_bytes(b'0123456789')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        def call(path,body=None,**headers):
            data=None if body is None else json.dumps(body).encode()
            if data is not None:headers.update({'Content-Type':'application/json','X-Orbbec-Request':'1'})
            return urlopen(Request(base+path,data=data,headers=headers),timeout=3)
        with self.assertRaises(HTTPError) as denied:call('/api/jobs/label')
        self.assertEqual(denied.exception.code,401)
        with call('/api/login',{'username':'alice','password':'test-password-strong-2026'}) as response:
            cookie=response.headers['Set-Cookie'].split(';')[0]
            self.assertIn('HttpOnly',response.headers['Set-Cookie'])
        with self.assertRaises(HTTPError) as denied:
            call('/api/lease',{},Cookie=cookie,Origin='https://untrusted.example')
        self.assertEqual(denied.exception.code,403)
        with call(f'/api/sessions/{item["id"]}/preview.mp4',Cookie=cookie,Range='bytes=3-6') as response:
            self.assertEqual(response.status,206);self.assertEqual(response.read(),b'3456')
        original=Handler.json
        def drop_ack(handler,value,*args,**kwargs):
            if value.get('accepted'):raise BrokenPipeError('lost acknowledgement')
            return original(handler,value,*args,**kwargs)
        with patch.object(Handler,'json',drop_ack):
            with self.assertRaises((RemoteDisconnected,URLError)):
                call(f'/api/sessions/{item["id"]}/submit',body,Cookie=cookie)
        with call(f'/api/sessions/{item["id"]}/submit',body,Cookie=cookie) as response:
            receipt=json.load(response)
            self.assertTrue(receipt['accepted'])
            self.assertEqual(receipt['submission_id'],body['submission_id'])
        self.assertEqual(sum(j['type']=='manual_3d' for j in self.desktop.store.jobs_for_episode('label')),1)

    def test_http_account_boundaries_and_retired_shared_keys(self):
        from http.server import ThreadingHTTPServer
        from remote_frontend.browser_server import configure, Handler
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        import threading
        from task_backend.server import AccountStore
        native = AccountStore(self.root/'native-accounts')
        native.register({'username':'xjz','password':'xjz','password_repeat':'xjz'})
        original_accounts = native.accounts_file.read_bytes()
        item,_=self.create('label')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        configure(server,self.db,{'nas_mounts':self.mounts, 'backend_accounts_file':str(native.accounts_file)},self.root/'state','alice')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        root=f'http://127.0.0.1:{server.server_port}'
        def call(path,body=None,cookie='',custom=True,origin=None):
            headers={'Cookie':cookie}
            if body is not None:
                headers['Content-Type']='application/json'
                if custom:headers['X-Orbbec-Request']='1'
            if origin:headers['Origin']=origin
            req=Request(root+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
            try:
                with urlopen(req,timeout=5) as res:return res.status,json.load(res),res.headers
            except HTTPError as res:return res.code,json.load(res),res.headers
        credential=json.loads((self.root/'state/initial-admin.json').read_text())
        login={'username':credential['username'],'password':credential['temporary_password']}
        self.assertEqual(call('/api/login',{'key':'old-key'})[0],401)
        self.assertEqual(call('/api/login',login,custom=False)[0],403)
        self.assertEqual(call('/api/login',login,origin='https://evil.example')[0],403)
        code,user,headers=call('/api/login',login);self.assertEqual(code,200)
        cookie=headers['Set-Cookie'].split(';')[0]
        self.assertTrue(user['must_change'])
        self.assertEqual(call('/api/jobs/label',cookie=cookie)[0],428)
        self.assertEqual(call('/api/password',{'old_password':login['password'],'new_password':'changed-private-password-2026'},cookie)[0],200)
        self.assertEqual(call('/api/identity',cookie=cookie)[0],401)
        _,_,headers=call('/api/login',dict(username='admin',password='changed-private-password-2026'))
        admin=headers['Set-Cookie'].split(';')[0]
        code,worker,_=call('/api/accounts',dict(username='vendor1',password='vendor-initial-password-2026',roles=['qc'],tasks=[]),admin)
        self.assertEqual(code,201)
        self.assertNotIn('password',worker)
        _,_,headers=call('/api/login',dict(username='vendor1',password='vendor-initial-password-2026'))
        cookie=headers['Set-Cookie'].split(';')[0]
        self.assertEqual(call('/api/accounts',cookie=cookie)[0],428)
        call('/api/password',dict(old_password='vendor-initial-password-2026',new_password='vendor-changed-password-2026'),cookie)
        _,_,headers=call('/api/login',dict(username='vendor1',password='vendor-changed-password-2026'))
        cookie=headers['Set-Cookie'].split(';')[0]
        self.assertEqual(call('/api/accounts',cookie=cookie)[0],403)
        self.assertEqual(call('/api/jobs/label',cookie=cookie)[0],403)
        self.assertEqual(call('/api/jobs/qc',cookie=cookie)[1],[])
        self.assertEqual(call(f'/api/sessions/{item["id"]}/samples.json',cookie=cookie)[0],403)
        call('/api/accounts/vendor1',{'enabled':False},admin)
        self.assertEqual(call('/api/identity',cookie=cookie)[0],401)
        self.assertEqual(call('/api/accounts',dict(username='xjz',auth_source='backend',roles=['qc'],tasks=[]),admin)[0],201)
        code,user,headers=call('/api/login',dict(username='xjz',password='xjz'))
        self.assertEqual(code,200);self.assertEqual(user['auth_source'],'backend');self.assertFalse(user['must_change'])
        linked_cookie=headers['Set-Cookie'].split(';')[0]
        self.assertEqual(call('/api/jobs/qc',cookie=linked_cookie)[1],[])
        self.assertEqual(call('/api/jobs/label',cookie=linked_cookie)[0],403)
        self.assertEqual(call('/api/password',dict(old_password='xjz',new_password='some other password!'),linked_cookie)[0],403)
        self.assertEqual(native.accounts_file.read_bytes(),original_accounts)
        call('/api/logout',{},admin)
        self.assertEqual(call('/api/identity',cookie=admin)[0],401)

    def test_scope_is_enforced_on_lease_and_existing_session(self):
        item,_=self.create('label')
        other=BatchService(self.db,self.mounts,'bob',roles=['label'],tasks=['label'])
        with self.assertRaises(WorkflowError) as denied:other.session(item['id'])
        self.assertEqual(denied.exception.status,403)
        denied_scope=BatchService(self.db,self.mounts,'alice',roles=['label'],tasks=[])
        with self.assertRaises(WorkflowError):denied_scope.session(item['id'])
        self.batch.release(item['id'])
        self.assertEqual(denied_scope.available('label'),[])
        with self.assertRaises(WorkflowError):denied_scope.lease('label',item['job_id'])
        denied_role=BatchService(self.db,self.mounts,'alice',roles=['qc'],tasks=['*'])
        with self.assertRaises(WorkflowError):denied_role.lease('label',item['job_id'])
        self.assertEqual(self.desktop.store.get_job(item['job_id'])['status'],'queued')

    def test_release_resume_preserves_draft_identity_but_cannot_steal(self):
        item,_=self.create('qc')
        self.batch.release(item['id'])
        self.assertEqual(self.desktop.store.get_job(item['job_id'])['status'],'queued')
        m=self.batch.resume(item['id'])
        self.assertEqual(m['id'],item['id'])
        self.assertFalse(m['released'])
        self.assertEqual(self.desktop.store.get_job(item['job_id'])['lease_owner'],item['owner'])
        self.batch.release(item['id'])
        other=BatchService(self.db,self.mounts,'bob')
        newer=other.lease('qc',item['job_id'])
        with self.assertRaises(WorkflowError):self.batch.resume(item['id'])
        with self.assertRaises(WorkflowError):self.batch.release(item['id'])
        self.assertEqual(self.desktop.store.get_job(item['job_id'])['lease_owner'],newer['owner'])

    def test_qc_native_completion_and_exception_reports(self):
        item,episode=self.create('qc')
        body=self.body(item);body['result'].update(reviewed=[],playback_complete=True,bad_ranges=[[0,0],[2,2]])
        self.assertTrue(self.batch.submit(item['id'],body)['accepted'])
        report=json.loads((episode/'qc/qc_report.json').read_text())
        self.assertTrue(report)

    def test_qc_rejects_out_of_range_or_unreviewed_result(self):
        item,episode=self.create('qc')
        body=self.body(item);body['result']['bad_ranges']=[[1,4]]
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],body)
        body=self.body(item);body['result']['reviewed']=[0,2]
        with self.assertRaises(WorkflowError):self.batch.submit(item['id'],body)
        self.assertFalse((episode/'qc/qc_report.json').exists())


if __name__=='__main__':unittest.main()
