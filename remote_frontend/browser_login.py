"""Private account store for the browser branch; no native workflow migrations.

OWASP scrypt fallback: N=2**17, r=8, p=1. Tokens are random, stored hashed,
revocable, and bounded by an absolute and idle lifetime. No URL credentials.
"""
from contextlib import contextmanager
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from .batch import reject


class Accounts:
    lifetime = 8 * 3600
    idle = 30 * 60

    def __init__(self, path, *, backend_accounts=None):
        self.path = Path(path)
        self.backend_accounts = Path(backend_accounts) if backend_accounts else None
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.hash_slots = threading.BoundedSemaphore(2)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS accounts (
                    username TEXT PRIMARY KEY, operator TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL, roles TEXT NOT NULL, tasks TEXT NOT NULL,
                    admin INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                    must_change INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS logins (
                    token TEXT PRIMARY KEY, username TEXT NOT NULL,
                    created REAL NOT NULL, touched REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS throttle (
                    bucket TEXT PRIMARY KEY, start REAL NOT NULL, count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (
                    time REAL NOT NULL, actor TEXT NOT NULL, event TEXT NOT NULL,
                    subject TEXT NOT NULL, peer TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS backend_links (
                    username TEXT PRIMARY KEY, backend_username TEXT UNIQUE NOT NULL);
                CREATE TABLE IF NOT EXISTS backend_sessions (
                    token TEXT PRIMARY KEY, signature TEXT NOT NULL);
            ''')
        self.path.chmod(0o600)
        # Unknown users take the same expensive verification path.
        self.dummy = self.password_hash(secrets.token_urlsafe(32))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def username(value):
        name = str(value or '').strip().lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{2,63}', name):
            reject('账号需为 3–64 位字母、数字、点、下划线或连字符', 400)
        return name

    @staticmethod
    def validate_password(value):
        if not isinstance(value, str) or not 15 <= len(value) <= 128:
            reject('密码需为 15–128 个字符，可以使用较长的短语', 400)
        if len(set(value)) < 5 or value.lower() in {
            'password123456789', '1234567890123456', 'qwertyuiopasdfgh',
            'admin123456789012', 'orbbec1234567890'}:
            reject('密码过于容易猜测，请使用不同的长密码', 400)

    def password_hash(self, value, salt=None):
        if not self.hash_slots.acquire(blocking=False):
            reject('登录服务繁忙，请稍后重试', 429)
        try:
            salt = salt or secrets.token_hex(16)
            result = hashlib.scrypt(value.encode(), salt=bytes.fromhex(salt),
                n=2**17, r=8, p=1, maxmem=256*1024*1024, dklen=32)
            return f'scrypt$131072$8$1${salt}${result.hex()}'
        finally:
            self.hash_slots.release()

    def verify(self, value, encoded):
        value = value if isinstance(value, str) and len(value) <= 128 else ''
        return secrets.compare_digest(self.password_hash(value, encoded.split('$')[4]), encoded)

    @staticmethod
    def public(row):
        return dict(username=row['username'], operator=row['operator'],
            roles=json.loads(row['roles']), tasks=json.loads(row['tasks']),
            admin=bool(row['admin']), enabled=bool(row['enabled']),
            auth_source='backend' if row['password'] == 'backend' else 'local',
            must_change=bool(row['must_change']))

    def backend_record(self, name):
        if not self.backend_accounts:
            reject('尚未配置原后端账号来源', 503)
        try:
            users = json.loads(self.backend_accounts.read_text())['users']
            row = users.get(name)
        except (OSError, ValueError, KeyError, AttributeError):
            reject('原后端账号服务暂不可用，请稍后重试', 503)
        if not isinstance(row, dict) or row.get('enabled') is False:
            return None
        if not row.get('password_salt') or not row.get('password_hash'):
            return None
        return row

    @staticmethod
    def backend_signature(row):
        if row is None:
            return None
        return hashlib.sha256(json.dumps({k: row.get(k) for k in
            ('username', 'password_salt', 'password_hash', 'hash', 'iterations')},
            sort_keys=True).encode()).hexdigest()

    def linked_record(self, name):
        with self.connect() as db:
            link = db.execute('SELECT backend_username FROM backend_links WHERE username=?', (name,)).fetchone()
        return self.backend_record(link[0]) if link else None

    def link_backend(self, name, roles, tasks, *, actor='setup', operator=None, enabled=True):
        backend_name = str(name or '').strip()
        name = self.username(backend_name)
        if self.backend_record(backend_name) is None:
            reject('原后端未注册这个账号', 400)
        roles, tasks = self.permissions(roles, tasks)
        if type(enabled) is not bool:
            reject('账号状态无效', 400)
        try:
            with self.connect() as db:
                db.execute('INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)',
                    (name, operator or uuid.uuid4().hex, 'backend', roles, tasks, 0, int(enabled), 0))
                db.execute('INSERT INTO backend_links VALUES (?,?)', (name, backend_name))
                db.execute('INSERT INTO audit VALUES (?,?,?,?,?)',
                    (time.time(), actor, 'backend-account-linked', name, ''))
        except sqlite3.IntegrityError:
            reject('账号已存在，不能覆盖其身份或改换验证来源', 409)
        return self.get(name)

    def verify_backend(self, password, row):
        # Reuse the native implementation, without constructing AccountStore:
        # no registration, last-login write, or password migration in its file.
        from task_backend.server import AccountStore
        if not self.hash_slots.acquire(blocking=False):
            reject('登录服务繁忙，请稍后重试', 429)
        try:
            password = password if isinstance(password, str) and len(password) <= 128 else ''
            salt = row['password_salt'] if row else '0' * 32
            actual = AccountStore._hash_password(password, salt)
            return bool(row) and secrets.compare_digest(actual, row['password_hash'])
        except (ValueError, KeyError, TypeError):
            return False
        finally:
            self.hash_slots.release()

    @staticmethod
    def permissions(roles, tasks):
        if not isinstance(roles, list) or not roles or not set(roles) <= {'label', 'qc'}:
            reject('至少选择一项 Label / QC 权限', 400)
        if (not isinstance(tasks, list) or len(tasks) > 200 or
                any(not isinstance(t, str) or not t.strip() or len(t) > 200 for t in tasks)):
            reject('任务范围格式无效', 400)
        # Empty is deny-all. Only an explicit * grants all tasks.
        if '*' in tasks and tasks != ['*']:
            reject('全部任务应单独使用 *', 400)
        return json.dumps(sorted(set(roles))), json.dumps(sorted(set(tasks)))

    def audit(self, actor, event, subject='', peer=''):
        with self.connect() as db:
            db.execute('INSERT INTO audit VALUES (?,?,?,?,?)',
                (time.time(), actor, event, subject, peer))

    def create(self, name, password, roles, tasks, *, admin=False, operator=None,
               must_change=True, actor='setup', enabled=True):
        name = self.username(name)
        self.validate_password(password)
        roles, tasks = self.permissions(roles, tasks)
        if type(enabled) is not bool:
            reject('账号状态无效', 400)
        encoded = self.password_hash(password)
        try:
            with self.connect() as db:
                db.execute('INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)',
                    (name, operator or uuid.uuid4().hex, encoded, roles, tasks, int(admin), int(enabled), int(must_change)))
                db.execute('INSERT INTO audit VALUES (?,?,?,?,?)',
                    (time.time(), actor, 'account-created', name, ''))
        except sqlite3.IntegrityError:
            reject('账号已存在', 409)
        return self.get(name)

    def get(self, name):
        with self.connect() as db:
            row = db.execute('SELECT * FROM accounts WHERE username=?', (name,)).fetchone()
        return self.public(row) if row else None

    def all(self):
        with self.connect() as db:
            return [self.public(r) for r in db.execute('SELECT * FROM accounts ORDER BY username')]

    def limit(self, name, peer):
        now = time.time()
        blocked = False
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM throttle WHERE start<?', (now - 900,))
            for bucket, maximum in [('account:' + name, 8), ('peer:' + peer, 100)]:
                row = db.execute('SELECT count FROM throttle WHERE bucket=?', (bucket,)).fetchone()
                if row and row['count'] >= maximum:
                    blocked = True
            if not blocked:
                for bucket in ['account:' + name, 'peer:' + peer]:
                    db.execute('INSERT INTO throttle VALUES (?,?,1) ON CONFLICT(bucket) DO UPDATE SET count=count+1', (bucket, now))
        if blocked:
            reject('登录尝试过多，请在 15 分钟后重试或联系管理员', 429)

    def login(self, name, password, peer):
        name = str(name or '').strip().lower()[:128]
        self.limit(name, peer)
        with self.connect() as db:
            row = db.execute('SELECT * FROM accounts WHERE username=?', (name,)).fetchone()
        backend = bool(row and row['password'] == 'backend')
        record = self.linked_record(name) if backend else None
        signature = self.backend_signature(record)
        local_valid = self.verify(password, row['password'] if row and not backend else self.dummy)
        # With native login enabled, both credential types and unknown names pay
        # the same two hash costs; timing must not reveal a linked account.
        backend_valid = self.verify_backend(password, record) if self.backend_accounts else False
        valid = backend_valid if backend else local_valid
        if not valid or not row or not row['enabled']:
            self.audit(name, 'login-failed', peer=peer)
            reject('账号或密码错误，或账号不可用', 401)
        token, now = secrets.token_urlsafe(32), time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT * FROM accounts WHERE username=?', (name,)).fetchone()
            if not current['enabled'] or current['password'] != row['password']:
                reject('账号或密码已变更，请重新登录', 401)
            db.execute('DELETE FROM logins WHERE created<? OR touched<?', (now-self.lifetime, now-self.idle))
            db.execute('DELETE FROM throttle WHERE bucket=?', ('account:' + name,))
            # At most five active browser sessions per account.
            db.execute('DELETE FROM logins WHERE token IN (SELECT token FROM logins WHERE username=? ORDER BY created DESC LIMIT -1 OFFSET 4)', (name,))
            db.execute('INSERT INTO logins VALUES (?,?,?,?)', (self.token_hash(token), name, now, now))
            if backend:
                db.execute('INSERT INTO backend_sessions VALUES (?,?)', (self.token_hash(token), signature))
            db.execute('DELETE FROM backend_sessions WHERE token NOT IN (SELECT token FROM logins)')
            db.execute('INSERT INTO audit VALUES (?,?,?,?,?)', (now, name, 'login-ok', '', peer))
        return token, self.public(current)

    @staticmethod
    def token_hash(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def authenticate(self, token, *, touch=False):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            reject('请登录工作台', 401)
        digest, now = self.token_hash(token), time.time()
        with self.connect() as db:
            row = db.execute('SELECT a.*, l.created, l.touched, b.signature FROM logins l JOIN accounts a USING(username) LEFT JOIN backend_sessions b USING(token) WHERE l.token=?', (digest,)).fetchone()
            if not row or not row['enabled'] or now-row['created'] >= self.lifetime or now-row['touched'] >= self.idle:
                reject('登录已过期，请重新登录；本机草稿仍保留', 401)
            if row['password'] == 'backend':
                signature = self.backend_signature(self.linked_record(row['username']))
                if not signature or signature != row['signature']:
                    reject('原后端账号已变更，请重新登录；本机草稿仍保留', 401)
            if touch:
                db.execute('UPDATE logins SET touched=? WHERE token=?', (now, digest))
        return {**self.public(row), 'expires_at': row['created'] + self.lifetime}

    def logout(self, token):
        with self.connect() as db:
            db.execute('DELETE FROM logins WHERE token=?', (self.token_hash(token),))

    def change_password(self, user, old, new, peer):
        if self.get(user['username'])['auth_source'] == 'backend':
            reject('此账号使用原后端密码，请在原账号管理流程修改', 403)
        self.limit(user['username'], peer)
        with self.connect() as db:
            row = db.execute('SELECT password FROM accounts WHERE username=?', (user['username'],)).fetchone()
        if not self.verify(old, row['password']):
            self.audit(user['username'], 'password-change-failed', peer=peer)
            reject('当前密码错误', 400)
        self.validate_password(new)
        if secrets.compare_digest(str(old), new):
            reject('新密码不能与当前密码相同', 400)
        encoded = self.password_hash(new)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            updated = db.execute('UPDATE accounts SET password=?, must_change=0 WHERE username=? AND password=? AND enabled=1', (encoded, user['username'], row['password']))
            if updated.rowcount != 1:
                reject('账号已变更，请重新登录', 401)
            db.execute('DELETE FROM logins WHERE username=?', (user['username'],))
            db.execute('DELETE FROM throttle WHERE bucket=?', ('account:' + user['username'],))
            db.execute('INSERT INTO audit VALUES (?,?,?,?,?)', (time.time(), user['username'], 'password-changed', '', peer))
        return {'ok': True}

    def update(self, actor, name, body):
        name = self.username(name)
        if name == actor['username']:
            reject('不能在此修改自己的权限或停用自己', 400)
        current = self.get(name)
        if not current:
            reject('账号不存在', 404)
        if current['admin']:
            reject('管理员账号只能通过服务器维护', 403)
        roles, tasks = self.permissions(body.get('roles', current['roles']), body.get('tasks', current['tasks']))
        enabled = body.get('enabled', current['enabled'])
        if type(enabled) is not bool:
            reject('账号状态无效', 400)
        password = body.get('password')
        if password is not None and current['auth_source'] == 'backend':
            reject('关联账号的密码由原后端管理，这里只能调整外包权限', 403)
        if password is not None:
            self.validate_password(password)
        encoded = self.password_hash(password) if password is not None else None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('UPDATE accounts SET roles=?, tasks=?, enabled=? WHERE username=?', (roles, tasks, int(enabled), name))
            if encoded:
                db.execute('UPDATE accounts SET password=?, must_change=1 WHERE username=?', (encoded, name))
            db.execute('DELETE FROM logins WHERE username=?', (name,))
            db.execute('INSERT INTO audit VALUES (?,?,?,?,?)', (time.time(), actor['username'], 'account-updated' + ('-password-reset' if encoded else ''), name, ''))
        return self.get(name)

    def bootstrap(self, operator):
        if self.all():
            return
        password = secrets.token_urlsafe(24)
        self.create('admin', password, ['label', 'qc'], ['*'], admin=True, operator=operator)
        # Local administrator delivery only; never returned by HTTP or put in a URL.
        path = self.path.parent / 'initial-admin.json'
        with open(path, 'x', opener=lambda p, f: __import__('os').open(p, f, 0o600)) as stream:
            json.dump({'username': 'admin', 'temporary_password': password,
                'note': 'First login requires a new password. This temporary password then stops working.'}, stream)
