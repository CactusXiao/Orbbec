import http.cookiejar
import json
import tempfile
import threading
import unittest
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

from task_backend.server import BackendError, BackendRuntime, RequestHandler, TaskBackend, TaskHTTPServer, TaskInstanceRegistry
from task_backend.job_service import JobService
from task_backend.workflow_store import WorkflowStore


class CollectionAssignmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / 'tasks.json'
        self.catalog.write_text(json.dumps({'first': {'total': 2}, 'second': {'total': 1}, 'third': {'total': 1}}))
        self.backend = TaskBackend(self.root / 'state', self.catalog)

    def reserve(self, user='alice', task='first', operator=None):
        return self.backend.reserve(dict(subject_id=user, operator_id=operator or user, client_id='capture', task_name=task))

    def confirm(self, reservation, user='alice'):
        return self.backend.confirm({**reservation, 'subject_id': user, 'operator_id': user, 'idempotency_key': reservation['reservation_id']})

    def test_whole_task_stays_owned_after_release_restart_and_completion(self):
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'first')
        first = self.reserve()
        self.backend.release({**first, 'subject_id': 'alice', 'operator_id': 'alice'})
        self.backend = TaskBackend(self.root / 'state', self.catalog)
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'first')
        self.assertEqual(self.backend.assigned_task('bob')['task']['task_name'], 'second')
        result = self.confirm(self.reserve())
        self.assertEqual(result['assigned_tasks'][0]['task_name'], 'first')
        reservation = self.reserve()
        result = self.confirm(reservation)
        self.assertEqual(result['assigned_tasks'][0]['task_name'], 'third')
        self.assertEqual(self.confirm(reservation)['assigned_tasks'][0]['task_name'], 'third')
        self.assertIsNone(self.backend.assigned_task('charlie')['task'])
        with self.assertRaises(BackendError):
            self.reserve('bob', 'first')

    def test_client_cannot_pick_another_task_or_change_operator(self):
        self.backend.assigned_task('alice')
        with self.assertRaises(BackendError):
            self.reserve('alice', 'second')
        with self.assertRaises(BackendError):
            self.reserve('alice', 'first', operator='bob')
        reservation = self.reserve()
        with self.assertRaises(BackendError):
            self.backend.confirm({**reservation, 'subject_id': 'alice', 'operator_id': 'bob', 'idempotency_key': 'bad'})

    def test_concurrent_clients_get_distinct_tasks_and_same_account_is_stable(self):
        def assign(user):
            return TaskBackend(self.root / 'state', self.catalog).assigned_task(user)['task']['task_name']
        with ThreadPoolExecutor(8) as pool:
            assigned = list(pool.map(assign, ['alice'] * 8))
        self.assertEqual(assigned, ['first'] * 8)
        with ThreadPoolExecutor(2) as pool:
            others = list(pool.map(assign, ['bob', 'charlie']))
        self.assertEqual(set(others), {'second', 'third'})

    def test_pending_uploads_cannot_over_reserve_or_advance_early(self):
        first, second = self.reserve(), self.reserve()
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'first')
        with self.assertRaises(BackendError):
            self.reserve()
        self.confirm(second)
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'first')
        self.assertEqual(self.confirm(first)['assigned_tasks'][0]['task_name'], 'second')

    def test_legacy_unfinished_task_resumes_before_unclaimed_tasks(self):
        self.backend.state_file.write_text(json.dumps({'subjects': {'alice': {'reservations': {
            'legacy': {'reservation_id': 'legacy', 'task_name': 'third', 'subject_id': 'alice', 'operator_id': 'alice', 'status': 'reserved', 'episode_number': 1}
        }}}}))
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'third')
        self.assertEqual(self.backend.assigned_task('bob')['task']['task_name'], 'first')

    def test_no_tasks_then_live_addition(self):
        self.catalog.write_text('{}')
        self.assertIsNone(self.backend.assigned_task('alice')['task'])
        self.catalog.write_text(json.dumps({'added': {'total': 1}}))
        self.assertEqual(self.backend.assigned_task('alice')['task']['task_name'], 'added')

    def test_native_client_against_real_http_server(self):
        cc, cxx = shutil.which('cc'), shutil.which('c++')
        if not cc or not cxx:
            self.skipTest('C/C++ compiler unavailable')
        repo = Path(__file__).resolve().parents[1]
        obj, binary = self.root / 'cjson.o', self.root / 'client-test'
        subprocess.run([cc, '-c', str(repo / 'src/sync/utils/cJSON.c'), '-o', str(obj)], check=True, capture_output=True)
        subprocess.run([cxx, '-std=c++17', '-I' + str(repo / 'src/sync'), '-I' + str(repo / 'task_backend'),
                        str(repo / 'tests/task_backend_assignment_client_test.cpp'), str(repo / 'task_backend/task_backend_client.cpp'),
                        str(obj), '-o', str(binary)], check=True, capture_output=True)
        runtime = BackendRuntime(TaskInstanceRegistry(self.root / 'registry'), JobService(WorkflowStore(self.root / 'workflow.sqlite3')))
        runtime.backend = self.backend
        server = TaskHTTPServer(('127.0.0.1', 0), RequestHandler, runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([str(binary), f'http://127.0.0.1:{server.server_port}'], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_web_session_uses_authenticated_account_and_logout_revokes_it(self):
        runtime = BackendRuntime(TaskInstanceRegistry(self.root / 'registry'), JobService(WorkflowStore(self.root / 'workflow.sqlite3')))
        runtime.backend = self.backend
        runtime.accounts.register({'username': 'alice', 'password': 'pw', 'password_repeat': 'pw'})
        runtime.accounts.register({'username': 'bob', 'password': 'pw', 'password_repeat': 'pw'})
        server = TaskHTTPServer(('127.0.0.1', 0), RequestHandler, runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        client = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        def request(path, body=None):
            return client.open(Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                       headers={'Content-Type': 'application/json'}), timeout=5)
        with self.assertRaises(HTTPError) as denied:
            request('/api/v1/operator/task?username=bob')
        self.assertEqual(denied.exception.code, 401)
        with self.assertRaises(HTTPError):
            request('/api/v1/operator/login', {'username': 'alice', 'password': 'wrong'})
        response = request('/api/v1/operator/login', {'username': 'alice', 'password': 'pw'})
        cookie = response.headers['Set-Cookie']
        self.assertIn('HttpOnly', cookie)
        self.assertNotIn('pw', response.read().decode())
        runtime.backend = None
        with request('/api/v1/operator/task') as response:
            waiting = json.load(response)
        self.assertEqual(waiting['username'], 'alice')
        self.assertTrue(waiting['waiting_for_setup'])
        runtime.backend = self.backend
        with request('/api/v1/operator/task?username=bob') as response:
            data = json.load(response)
        self.assertEqual(data['username'], 'alice')
        self.assertEqual(data['task']['task_name'], 'first')
        with request('/api/v1/collection/assignment', {'subject_id': 'alice'}) as response:
            self.assertEqual(json.load(response)['task'], data['task'])
        request('/api/v1/operator/logout', {}).close()
        with self.assertRaises(HTTPError):
            request('/api/v1/operator/task')
        with self.assertRaises(HTTPError):
            client.open(Request(base + '/api/v1/operator/task', headers={'Cookie': cookie.split(';')[0]}))
        request('/api/v1/operator/login', {'username': 'bob', 'password': 'pw'}).close()
        with request('/api/v1/operator/task') as response:
            self.assertEqual(json.load(response)['task']['task_name'], 'second')


if __name__ == '__main__':
    unittest.main()
