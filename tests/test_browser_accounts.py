import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from remote_frontend.browser_login import Accounts
from task_backend.workflow_models import WorkflowError

PASSWORD='A long private phrase 2026!'
class AccountTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.auth=Accounts(self.root/'accounts.sqlite3')
        self.alice=self.auth.create('alice',PASSWORD,['label'],['task1'],must_change=False)
    def test_password_and_tokens_are_not_stored_plaintext(self):
        token,user=self.auth.login('ALICE',PASSWORD,'peer')
        with self.auth.connect() as db:
            row=db.execute('SELECT password FROM accounts').fetchone()[0]
            stored=db.execute('SELECT token FROM logins').fetchone()[0]
        self.assertTrue(row.startswith('scrypt$131072$8$1$'))
        self.assertNotIn(PASSWORD,row);self.assertNotEqual(token,stored)
        self.assertEqual(self.auth.authenticate(token)['username'],'alice')
        self.assertEqual(self.auth.path.stat().st_mode&0o777,0o600)
    def test_expiry_and_logout_are_enforced_by_server(self):
        token,_=self.auth.login('alice',PASSWORD,'peer');now=time.time()
        with patch('remote_frontend.browser_login.time.time',return_value=now+1801):
            with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        self.auth.logout(token)
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        token,_=self.auth.login('alice',PASSWORD,'peer')
        with self.auth.connect() as db:db.execute('UPDATE logins SET created=?,touched=?',(now-28801,now))
        with self.assertRaises(WorkflowError):self.auth.authenticate(token,touch=True)
    def test_background_reads_do_not_extend_idle_lifetime(self):
        token,_=self.auth.login('alice',PASSWORD,'peer')
        with self.auth.connect() as db:initial=db.execute('SELECT touched FROM logins').fetchone()[0]
        with patch('remote_frontend.browser_login.time.time',return_value=initial+900):self.auth.authenticate(token)
        with patch('remote_frontend.browser_login.time.time',return_value=initial+1801):
            with self.assertRaises(WorkflowError):self.auth.authenticate(token)
    def test_limits_persist_and_unknown_user_is_generic(self):
        messages=[]
        for name in ['alice','unknown']:
            with self.assertRaises(WorkflowError) as error:self.auth.login(name,'wrong','peer')
            messages.append(str(error.exception))
        self.assertEqual(*messages)
        for _ in range(7):
            with self.assertRaises(WorkflowError):self.auth.login('alice','wrong','peer')
        restarted=Accounts(self.auth.path)
        with self.assertRaises(WorkflowError) as error:restarted.login('alice',PASSWORD,'different-peer')
        self.assertEqual(error.exception.status,429)
    def test_change_reset_and_disable_revoke_all_sessions(self):
        token,user=self.auth.login('alice',PASSWORD,'peer')
        second,_=self.auth.login('alice',PASSWORD,'peer')
        self.auth.change_password(user,PASSWORD,PASSWORD+' changed','peer')
        for t in [token,second]:
            with self.assertRaises(WorkflowError):self.auth.authenticate(t)
        token,_=self.auth.login('alice',PASSWORD+' changed','peer')
        self.auth.update({'username':'admin'},'alice',{'password':PASSWORD+' reset'})
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        token,user=self.auth.login('alice',PASSWORD+' reset','peer');self.assertTrue(user['must_change'])
        self.auth.update({'username':'admin'},'alice',{'enabled':False})
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
    def test_bootstrap_is_one_time_and_preserves_legacy_operator(self):
        auth=Accounts(self.root/'bootstrap'/'accounts.sqlite3');auth.bootstrap('legacy-operator')
        credential=json.loads((auth.path.parent/'initial-admin.json').read_text())
        token,user=auth.login(credential['username'],credential['temporary_password'],'peer')
        self.assertTrue(user['must_change']);self.assertTrue(user['admin']);self.assertEqual(user['operator'],'legacy-operator')
        auth.bootstrap('different');self.assertEqual(len(auth.all()),1)
        with self.assertRaises(WorkflowError):auth.create('bad','short',['label'],['*'])
    def test_permission_updates_preserve_operator_and_revoke(self):
        token,user=self.auth.login('alice',PASSWORD,'peer')
        updated=self.auth.update({'username':'admin'},'alice',{'roles':['qc'],'tasks':[]})
        self.assertEqual(updated['operator'],user['operator']);self.assertEqual(updated['tasks'],[])
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)


class BackendAccountTest(unittest.TestCase):
    def setUp(self):
        from task_backend.server import AccountStore
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.native=AccountStore(self.root/'native')
        self.native.register({'username':'xjz','password':'xjz','password_repeat':'xjz'})
        self.path=self.native.accounts_file
        self.before=self.path.read_bytes()
        self.auth=Accounts(self.root/'browser/accounts.sqlite3',backend_accounts=self.path)
        self.auth.link_backend('xjz',['label','qc'],['task1'],operator='xjz')
    def test_native_credentials_work_without_copy_or_password_change(self):
        token,user=self.auth.login('xjz','xjz','peer')
        self.assertEqual(user['auth_source'],'backend');self.assertFalse(user['must_change'])
        self.assertFalse(user['admin']);self.assertEqual(user['operator'],'xjz')
        self.assertEqual(self.auth.authenticate(token)['tasks'],['task1'])
        self.assertEqual(self.path.read_bytes(),self.before)
        with self.auth.connect() as db:self.assertEqual(db.execute('select password from accounts').fetchone()[0],'backend')
        with self.assertRaises(WorkflowError):self.auth.login('xjz','incorrect','peer')
    def test_native_password_change_invalidates_old_sessions_and_old_password(self):
        token,_=self.auth.login('xjz','xjz','peer')
        data=json.loads(self.path.read_text());row=data['users']['xjz']
        row['password_hash']=self.native._hash_password('changed in native',row['password_salt'])
        self.path.write_text(json.dumps(data))
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        with self.assertRaises(WorkflowError):self.auth.login('xjz','xjz','peer')
        fresh,_=self.auth.login('xjz','changed in native','peer')
        self.assertEqual(self.auth.authenticate(fresh)['username'],'xjz')
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
    def test_native_deletion_and_registry_unavailability_fail_closed(self):
        token,_=self.auth.login('xjz','xjz','peer')
        self.path.write_text(json.dumps({'users':{}}))
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        with self.assertRaises(WorkflowError):self.auth.login('xjz','xjz','peer')
        self.path.unlink()
        with self.assertRaises(WorkflowError) as error:self.auth.authenticate(token)
        self.assertEqual(error.exception.status,503)
    def test_link_does_not_allow_password_reset_identity_collision_or_registration(self):
        _,user=self.auth.login('xjz','xjz','peer')
        with self.assertRaises(WorkflowError):self.auth.change_password(user,'xjz',PASSWORD,'peer')
        with self.assertRaises(WorkflowError):self.auth.update({'username':'admin'},'xjz',{'password':PASSWORD})
        with self.assertRaises(WorkflowError):self.auth.link_backend('xjz',['label'],['*'])
        with self.assertRaises(WorkflowError):self.auth.link_backend('unknown',['label'],['*'])
        self.assertEqual(self.path.read_bytes(),self.before)
    def test_external_permissions_and_disable_remain_enforced(self):
        token,user=self.auth.login('xjz','xjz','peer')
        self.auth.update({'username':'admin'},'xjz',{'tasks':[],'roles':['qc']})
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        token,user=self.auth.login('xjz','xjz','peer')
        self.assertEqual(user['tasks'],[]);self.assertEqual(user['roles'],['qc'])
        self.auth.update({'username':'admin'},'xjz',{'enabled':False})
        with self.assertRaises(WorkflowError):self.auth.authenticate(token)
        with self.assertRaises(WorkflowError):self.auth.login('xjz','xjz','peer')
        self.assertEqual(self.path.read_bytes(),self.before)

if __name__=='__main__':unittest.main()
