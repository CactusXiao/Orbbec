import io
import json
from pathlib import Path
import tempfile
import tarfile
import time
import threading
import subprocess
import sys
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import unittest
from unittest.mock import Mock, patch

from remote_frontend.qc_dispatch import QCDispatch, RoutedMedia, capability, valid_capability
from remote_frontend.qc_worker import extract_inputs, Agent, MediaHandler
from task_backend.workflow_models import WorkflowError


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'secret').write_text('s'*48)
        self.now = 1000
        self.config = dict(secret_file=str(self.root/'secret'), workers={
            'a':'https://a.example.test','b':'https://b.example.test'})
        self.d = QCDispatch(self.root, self.config, clock=lambda:self.now)
        self.batch = Mock()
        self.batch.session.side_effect = lambda sid: dict(role='qc', payload={'episode_id':sid[0]},
            revision=sid[0], job_id=sid[0], owner='owner')
        self.batch.store.get_job.return_value = dict(lease_owner='owner',status='running')
        self.task = patch('remote_frontend.qc_dispatch.browser_task')
        task=self.task.start().return_value
        task.episode_dir.return_value=self.root/'episode'
        task.frames=[0,1];task.cameras=['00']
    def tearDown(self):
        self.task.stop()
        self.tmp.cleanup()
    def poll(self, who, **kwargs):
        return self.d.poll(dict(worker_id=who,accepting=True,jobs={},**kwargs))
    def test_one_episode_per_host_and_completed_capacity(self):
        self.d.status(self.batch,'x1'); self.d.status(self.batch,'x2'); self.d.status(self.batch,'y1')
        self.assertEqual(len(self.d.jobs),2)
        a=self.poll('a')['jobs']; self.assertEqual(len(a),1)
        self.assertEqual(len(self.poll('a')['jobs']),1)
        b=self.poll('b')['jobs']; self.assertEqual(len(b),1)
        self.assertNotEqual(a[0]['id'],b[0]['id'])
        self.d.status(self.batch,'z1')
        j=a[0]
        result=self.d.poll(dict(worker_id='a',accepting=True,jobs={j['id']:dict(generation=j['generation'],state={'ready':True,'complete':True})}))
        self.assertEqual(len(result['jobs']),2)
    def test_late_heartbeat_is_fenced_and_requeued(self):
        self.d.status(self.batch,'x1'); old=self.poll('a')['jobs'][0]
        self.now += 31
        new=self.poll('b')['jobs'][0]
        self.assertNotEqual(old['generation'],new['generation'])
        self.d.poll(dict(worker_id='a',accepting=True,jobs={old['id']:dict(generation=old['generation'],state={'ready':True,'complete':True})}))
        self.assertFalse(self.d.jobs[old['id']]['state']['ready'])
        with self.assertRaises(WorkflowError): self.d.input_source('a',old['id'],old['generation'])
    def test_release_reference_count_restart_and_backpressure(self):
        self.d.status(self.batch,'x1'); self.d.status(self.batch,'x2')
        self.d.cancel(self.batch,'x1'); self.assertEqual(len(self.d.jobs),1)
        self.assertEqual(self.d.poll(dict(worker_id='a',accepting=False,jobs={}))['jobs'],[])
        old=self.poll('a')['jobs'][0]
        restored=QCDispatch(self.root,self.config,clock=lambda:self.now)
        self.assertIsNone(restored.jobs[old['id']]['worker'])
        self.assertNotEqual(restored.jobs[old['id']]['generation'],old['generation'])
        self.d.cancel(self.batch,'x2'); self.assertEqual(self.poll('a')['jobs'],[])
    def test_qc_cannot_use_local_media_label_unchanged(self):
        local=Mock(); local.batch=self.batch
        router=RoutedMedia(local,self.d)
        router.status('x1'); local.status.assert_not_called()
        with self.assertRaises(WorkflowError): router.directory('x1')
        local.directory.assert_not_called()
        self.batch.session.side_effect=lambda sid:dict(role='label')
        router.status('l'); local.status.assert_called_once_with('l')
    def test_worker_restart_fences_completed_outputs(self):
        self.d.status(self.batch,'x1')
        old=self.d.poll(dict(worker_id='a',boot_id='old',accepting=True,jobs={}))['jobs'][0]
        new=self.d.poll(dict(worker_id='a',boot_id='new',accepting=True,jobs={}))['jobs'][0]
        self.assertNotEqual(old['generation'],new['generation'])
        self.assertFalse(self.d.jobs[new['id']]['state']['ready'])

    def test_auth_scope_and_expiry(self):
        with self.assertRaises(WorkflowError): self.d.authenticate('Bearer wrong')
        with self.assertRaises(WorkflowError): self.poll('unknown')
        token=capability('s'*48,'x','gen',int(time.time())+60)
        self.assertTrue(valid_capability('s'*48,'x','gen',token))
        self.assertFalse(valid_capability('s'*48,'x','other',token))
        self.assertFalse(valid_capability('s'*48,'y','gen',token))
        self.assertFalse(valid_capability('s'*48,'x','gen',capability('s'*48,'x','gen',1)))
    def test_archive_path_link_and_disk_limits(self):
        for name,kind,limit in [('../escape',tarfile.REGTYPE,100),('subject/link',tarfile.SYMTYPE,100),('subject/task/episode/f',tarfile.REGTYPE,0)]:
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode='w') as archive:
                member=tarfile.TarInfo(name); member.type=kind
                member.size=1 if kind==tarfile.REGTYPE else 0
                archive.addfile(member,io.BytesIO(b'x'))
            stream.seek(0)
            with self.assertRaises(ValueError): extract_inputs(stream,self.root/'extract',limit)

class MediaHTTPTests(unittest.TestCase):
    def test_cancellation_reaps_renderer_children_after_parent_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent=Agent.__new__(Agent);agent.root=Path(tmp)
            code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
            process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True)
            child=int(process.stdout.readline())
            job=dict(id='a'*32,generation='b'*32,process=process,log=(Path(tmp)/'worker.log').open('w'))
            try:
                agent.halt(job)
                self.assertIsNotNone(process.poll())
                path=Path('/proc')/str(child)/'stat'
                for _ in range(100):
                    if not path.exists() or path.read_text().rsplit(')',1)[1].split()[0]=='Z':
                        break
                    time.sleep(.01)
                else:
                    self.fail('Renderer child survived cancellation')
            finally:
                if process.poll() is None:process.kill();process.wait()
                process.stdout.close()

    def test_direct_range_cors_scope_and_lease_revocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent=Agent.__new__(Agent)
            agent.root=Path(tmp); agent.secret='secret'*8
            agent.config={'browser_origins':['https://workbench.example']}
            agent.lock=threading.RLock(); agent.last_contact=time.monotonic(); agent.lease_seconds=30
            job={'id':'a'*32,'generation':'b'*32,'payload':{'cameras':['00'],'frames':[0]}}
            agent.jobs={job['id']:job}
            folder=agent.folder(job)/'media'/job['id']/'layered-v1'/'chunks'
            folder.mkdir(parents=True); (folder/'0.mp4').write_bytes(b'0123456789')
            token=capability(agent.secret,job['id'],job['generation'],int(time.time())+60)
            server=ThreadingHTTPServer(('127.0.0.1',0),MediaHandler); server.agent=agent
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f"http://127.0.0.1:{server.server_port}/media/{job['id']}/{job['generation']}/{token}/chunks/0.mp4"
            try:
                with urlopen(Request(url,headers={'Range':'bytes=2-5','Origin':'https://workbench.example'})) as r:
                    self.assertEqual(r.status,206);self.assertEqual(r.read(),b'2345')
                    self.assertEqual(r.headers['Content-Range'],'bytes 2-5/10')
                    self.assertEqual(r.headers['Access-Control-Allow-Origin'],'https://workbench.example')
                with self.assertRaises(HTTPError) as e:urlopen(Request(url,headers={'Range':'bytes=100-200'}))
                self.assertEqual(e.exception.code,416)
                with self.assertRaises(HTTPError) as e:urlopen(url.replace(token,'0.invalid'))
                self.assertEqual(e.exception.code,403)
                agent.jobs.clear()
                with self.assertRaises(HTTPError) as e:urlopen(url)
                self.assertEqual(e.exception.code,403)
            finally:
                server.shutdown();server.server_close();thread.join()

if __name__=='__main__': unittest.main()
