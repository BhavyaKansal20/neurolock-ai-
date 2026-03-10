"""
NeuroLock AI v2 — Database Layer (SQLite)
Stores students, sessions, emotion logs, and reports.
"""

import os
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'neurolock.db')


@contextmanager
def get_conn(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    """Create all tables if not exist."""
    with get_conn(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            class_name  TEXT,
            roll_no     TEXT,
            image_path  TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            teacher      TEXT,
            subject      TEXT,
            location     TEXT,
            start_time   TEXT,
            end_time     TEXT,
            phase        TEXT DEFAULT 'before',
            status       TEXT DEFAULT 'active',
            notes        TEXT
        );

        CREATE TABLE IF NOT EXISTS emotion_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            student_id      TEXT,
            student_name    TEXT,
            timestamp       TEXT DEFAULT (datetime('now')),
            phase           TEXT,
            dominant        TEXT,
            confidence      REAL,
            emotions_json   TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS reports (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            student_id   TEXT,
            report_json  TEXT,
            created_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_logs_session  ON emotion_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_logs_student  ON emotion_logs(student_id);
        CREATE INDEX IF NOT EXISTS idx_logs_phase    ON emotion_logs(phase);
        """)
    print(f"  Database initialized: {db_path}")


# ── Students ──────────────────────────────────────────────────

def add_student(student_id, name, class_name='', roll_no='', image_path='', db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO students (id, name, class_name, roll_no, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, name, class_name, roll_no, image_path))

def get_students(db_path=DB_PATH) -> list:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM students ORDER BY name").fetchall()
        return [dict(r) for r in rows]

def delete_student(student_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM students WHERE id=?", (student_id,))

# ── Sessions ──────────────────────────────────────────────────

def create_session(session_id, name, teacher='', subject='', location='', db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO sessions (id, name, teacher, subject, location, start_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, name, teacher, subject, location, datetime.now().isoformat()))
    return session_id

def update_session_phase(session_id, phase, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("UPDATE sessions SET phase=? WHERE id=?", (phase, session_id))

def end_session(session_id, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            UPDATE sessions SET end_time=?, status='completed' WHERE id=?
        """, (datetime.now().isoformat(), session_id))

def get_session(session_id, db_path=DB_PATH) -> dict:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

def get_sessions(db_path=DB_PATH) -> list:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC").fetchall()
        return [dict(r) for r in rows]

# ── Emotion Logs ──────────────────────────────────────────────

def log_emotion(session_id, phase, dominant, confidence, emotions_dict,
                student_id=None, student_name=None, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO emotion_logs
                (session_id, student_id, student_name, phase, dominant, confidence, emotions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, student_id, student_name, phase, dominant,
               confidence, json.dumps(emotions_dict)))

def get_logs(session_id, student_id=None, phase=None, db_path=DB_PATH) -> list:
    with get_conn(db_path) as conn:
        query = "SELECT * FROM emotion_logs WHERE session_id=?"
        params = [session_id]
        if student_id:
            query += " AND student_id=?"; params.append(student_id)
        if phase:
            query += " AND phase=?"; params.append(phase)
        query += " ORDER BY timestamp"
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d['emotions'] = json.loads(d.pop('emotions_json', '{}'))
            result.append(d)
        return result

# ── Reports ───────────────────────────────────────────────────

def save_report(session_id, report_data, student_id=None, db_path=DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO reports (session_id, student_id, report_json)
            VALUES (?, ?, ?)
        """, (session_id, student_id, json.dumps(report_data)))

def get_report(session_id, student_id=None, db_path=DB_PATH) -> dict:
    with get_conn(db_path) as conn:
        if student_id:
            row = conn.execute(
                "SELECT * FROM reports WHERE session_id=? AND student_id=? ORDER BY created_at DESC LIMIT 1",
                (session_id, student_id)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM reports WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
                (session_id,)
            ).fetchone()
        if row:
            d = dict(row)
            d['report'] = json.loads(d.pop('report_json', '{}'))
            return d
        return None
