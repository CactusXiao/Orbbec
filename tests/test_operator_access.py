import http.cookiejar
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, HTTPCookieProcessor, build_opener

from task_backend.server import BackendRuntime, RequestHandler, TaskBackend, TaskHTTPServer, TaskInstanceRegistry
from task_backend.job_service import JobService
from task_backend.workflow_store import WorkflowStore
from task_backend.access_policy import management_allowed


class LanServer(TaskHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        return connection, ('192.0.2.10', address[1])


class CampusServer(TaskHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        return connection, ('10.230.194.204', address[1])


class OperatorAccessTest(unittest.TestCase):
    def test_network_roles(self):
        for peer in ('127.0.0.1', '::1', '10.162.208.158', '10.230.194.204', '::ffff:10.1.2.3'):
            self.assertTrue(management_allowed(peer), peer)
        for peer in ('192.168.50.177', '192.168.1.2', '172.16.1.2', '203.0.113.1', '::ffff:192.168.50.177'):
            self.assertFalse(management_allowed(peer), peer)

    @patch.dict('os.environ', {'ORBBEC_OPERATOR_NAT_NETWORKS': '10.192.35.102/32'})
    def test_operator_nat_is_not_campus_management(self):
        self.assertFalse(management_allowed('10.192.35.102'))
        self.assertFalse(management_allowed('::ffff:10.192.35.102'))
        self.assertTrue(management_allowed('10.230.194.204'))
        self.assertTrue(management_allowed('127.0.0.1'))

    def test_lan_cannot_access_admin_or_other_operators_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / 'tasks.json'
            catalog.write_text(json.dumps({'first': {'total': 1}, 'second': {'total': 1}}))
            for name in ('first', 'second'):
                directory = root / 'nas' / 'tasks' / name
                directory.mkdir(parents=True)
                (directory / 'demo.mp4').write_bytes(b'test-video-bytes')
                (directory / 'task.json').write_text(json.dumps({'task_name': name, 'description_cn': name}))
            runtime = BackendRuntime(TaskInstanceRegistry(root / 'registry'), JobService(WorkflowStore(root / 'workflow.sqlite3')))
            runtime.backend = TaskBackend(root / 'progress', task_file=catalog, nas_root=root / 'nas')
            runtime.accounts.register({'username': 'alice', 'password': 'pw', 'password_repeat': 'pw'})
            servers = [cls(('127.0.0.1', 0), RequestHandler, runtime) for cls in (LanServer, TaskHTTPServer, CampusServer)]
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
            for thread in threads:
                thread.start()
            client = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
            base = f'http://127.0.0.1:{servers[0].server_port}'
            def request(path, method='GET', body=None, headers=None):
                return client.open(Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                           headers={'Content-Type': 'application/json', **(headers or {})}), timeout=5)
            def denied(path, method='GET', status=403, headers=None):
                with self.assertRaises(HTTPError) as error:
                    request(path, method, headers=headers)
                self.assertEqual(error.exception.code, status)
            try:
                self.assertEqual(request('/operator').status, 200)
                denied('/api/v1/operator/task', status=401)
                denied('/task-assets/first/demo.mp4', status=401)
                for path, method in [('/', 'GET'), ('/tasks', 'GET'), ('/people', 'GET'), ('/setup', 'GET'),
                                     ('/manage/tasks/new', 'GET'), ('/api/v1/tasks', 'POST'), ('/api/v1/tasks/preview', 'POST'),
                                     ('/api/v1/personnel', 'GET'), ('/api/v1/auth/register', 'POST'),
                                     ('/api/v1/collection/assignment', 'POST'), ('/api/v1/episodes/reserve', 'POST'),
                                     ('/api/v1/episodes/confirm', 'POST'), ('/api/v1/jobs/lease', 'POST'),
                                     ('/api/v1/viewer/sessions/fake', 'DELETE'), ('/operator', 'PUT'),
                                     ('/%6danage/tasks/new', 'GET')]:
                    denied(path, method, headers={'Host': 'localhost', 'X-Forwarded-For': '127.0.0.1', 'X-Real-IP': '127.0.0.1'})
                request('/api/v1/operator/login', 'POST', {'username': 'alice', 'password': 'pw'}).close()
                self.assertEqual(json.load(request('/api/v1/operator/task?username=other'))['username'], 'alice')
                response = request('/task-assets/first/demo.mp4', headers={'Range': 'bytes=0-3'})
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b'test')
                denied('/task-assets/second/demo.mp4')
                denied('/task-assets/first/task.json')
                denied('/manage/tasks/new')  # Login never grants management privileges.
                local = f'http://127.0.0.1:{servers[1].server_port}'
                self.assertEqual(client.open(local + '/manage/tasks/new').status, 200)
                self.assertEqual(client.open(local + '/tasks').status, 200)
                campus = f'http://127.0.0.1:{servers[2].server_port}'
                self.assertEqual(client.open(campus + '/manage/tasks/new').status, 200)
                self.assertEqual(client.open(campus + '/api/v1/personnel').status, 200)
                request('/api/v1/operator/logout', 'POST', {}).close()
                denied('/task-assets/first/demo.mp4', status=401)
            finally:
                for server in servers:
                    server.shutdown()
                    server.server_close()
                for thread in threads:
                    thread.join(3)


if __name__ == '__main__':
    unittest.main()
