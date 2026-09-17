"""Local, transactional review history. Raw recordings are never modified."""
import json
import sqlite3
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, id TEXT, at TEXT, before TEXT, after TEXT);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, body TEXT NOT NULL);
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def setting(self, key, default):
        with self.connect() as db:
            row = db.execute('SELECT body FROM settings WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, json.dumps(value)))

    @staticmethod
    def empty():
        return dict(grade='', tags=[], note='', instruction='', segments=[], revision=0, updated_at=None, source_fingerprint=None, event_decisions={}, event_evidence={})

    def all(self):
        with self.connect() as db:
            return {key: {**self.empty(),**json.loads(body)} for key, body in db.execute('SELECT id,body FROM reviews')}

    def history(self, ep):
        with self.connect() as db:
            rows = db.execute('SELECT seq,at,before,after FROM audit WHERE id=? ORDER BY seq DESC LIMIT 50', (ep,)).fetchall()
        return [dict(seq=s, at=t, before=json.loads(b), after=json.loads(a)) for s,t,b,a in rows]

    def save_many(self, changes):
        saved = {}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for ep, patch in changes:
                row = db.execute('SELECT body FROM reviews WHERE id=?', (ep,)).fetchone()
                old = {**self.empty(),**json.loads(row[0])} if row else self.empty()
                if patch.get('revision') != old['revision']:
                    raise Conflict('标注已在其他窗口更新，请刷新后重试。')
                new = {**old, **{k:v for k,v in patch.items() if k in old},
                       'revision': old['revision'] + 1, 'updated_at': now()}
                body = json.dumps(new, ensure_ascii=False)
                db.execute('INSERT OR REPLACE INTO reviews VALUES (?,?)', (ep,body))
                db.execute('INSERT INTO audit(id,at,before,after) VALUES (?,?,?,?)',
                           (ep,now(),json.dumps(old,ensure_ascii=False),body))
                saved[ep] = new
        return saved
