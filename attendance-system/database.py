"""
database.py
-----------
Handles all SQLite operations for the Attendance System.

Schema:
    users(id INTEGER PK, roll_no TEXT UNIQUE, name TEXT, created_at TEXT)
    attendance(id INTEGER PK, user_id INTEGER FK -> users.id,
               date TEXT, time TEXT, timestamp TEXT)

Notes for viva/interview:
- roll_no has a UNIQUE constraint -> enforces entity integrity.
- attendance.user_id is a FOREIGN KEY -> enforces referential integrity.
- date is stored separately from timestamp so "has X already been marked
  present today" is a cheap indexed lookup instead of a string-prefix scan.
- Swapping this module for MySQL later only means changing the connector
  (sqlite3 -> mysql.connector) and placeholder style (? -> %s); the SQL
  itself is portable ANSI SQL.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "db", "attendance.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_cursor(commit=False):
    """Context manager so every route doesn't repeat connect/close boilerplate."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with db_cursor(commit=True) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, date)   -- one attendance mark per person per day
            )
        """)
        # Index speeds up the common "attendance for date X" report query
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")


# ---------- Users ----------

def create_user(roll_no, name):
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO users (roll_no, name, created_at) VALUES (?, ?, ?)",
            (roll_no, name, datetime.now().isoformat()),
        )
        return cur.lastrowid


def get_user_by_roll(roll_no):
    with db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE roll_no = ?", (roll_no,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_user(user_id):
    """Used to roll back a registration when no valid face samples were saved."""
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))


def get_user_by_id(user_id):
    with db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_users():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM users ORDER BY name")
        return [dict(r) for r in cur.fetchall()]


# ---------- Attendance ----------

def mark_attendance(user_id):
    """
    Returns (success, message).
    success=False if already marked today (UNIQUE constraint catches it).
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO attendance (user_id, date, time, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, date_str, time_str, now.isoformat()),
            )
        return True, f"Attendance marked at {time_str}"
    except sqlite3.IntegrityError:
        return False, "Attendance already marked for today"


def get_attendance_by_date(date_str):
    with db_cursor() as cur:
        cur.execute("""
            SELECT a.id, u.roll_no, u.name, a.date, a.time
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.date = ?
            ORDER BY a.time
        """, (date_str,))
        return [dict(r) for r in cur.fetchall()]


def get_attendance_by_user(user_id, start_date=None, end_date=None):
    query = "SELECT * FROM attendance WHERE user_id = ?"
    params = [user_id]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    with db_cursor() as cur:
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def get_attendance_summary():
    """Simple aggregation: total presents per user - useful for a dashboard."""
    with db_cursor() as cur:
        cur.execute("""
            SELECT u.roll_no, u.name, COUNT(a.id) as days_present
            FROM users u
            LEFT JOIN attendance a ON a.user_id = u.id
            GROUP BY u.id
            ORDER BY days_present DESC
        """)
        return [dict(r) for r in cur.fetchall()]
