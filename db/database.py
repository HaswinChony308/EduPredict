"""
database.py — SQLite Database Layer for AAPT
=============================================
This module handles ALL database interactions. We use SQLite because:
  1. It requires zero setup (no separate DB server needed)
  2. The entire database is a single file (easy to back up / share)
  3. Perfect for a project running on one machine

Tables:
  - students:        one row per student, holds latest risk prediction
  - predictions:     historical log of every prediction made
  - engagement_log:  weekly engagement metrics per student
  - drift_log:       record of every drift detection check
"""

import sqlite3
import os
from datetime import datetime

# Import the database path from our central config
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SQLITE_DB_PATH


def get_connection():
    """
    Create and return a new SQLite connection.
    
    Why we create a new connection each time:
      - SQLite connections are NOT thread-safe by default
      - Streamlit may run handlers on different threads
      - Creating fresh connections avoids "database is locked" errors
    
    row_factory = sqlite3.Row lets us access columns by name
    instead of by index, e.g.  row["student_id"] instead of row[0].
    """
    # Ensure the directory for the database file exists
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def init_db():
    """
    Create all required tables if they don't already exist.
    
    Called once when the app starts. The IF NOT EXISTS clause means
    it's safe to call multiple times — it won't drop existing data.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # ── Students table ──────────────────────────────────────
    # Stores the current state of each student.
    # risk_level: latest prediction (Excelling / On Track / Moderate Risk / High Risk)
    # status: "Static" (normal prediction) or "Dynamic Risk" (drift detected)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id    TEXT PRIMARY KEY,
            name          TEXT,
            email         TEXT,
            teacher_email TEXT,
            course_id     TEXT,
            risk_level    TEXT,
            confidence_score REAL,
            status        TEXT DEFAULT 'Static',
            last_updated  TIMESTAMP
        )
    """)
    
    # ── Predictions table ───────────────────────────────────
    # Historical log — every prediction ever made is recorded here.
    # This lets us build "grade trajectory" line charts over time.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            prediction_date TIMESTAMP,
            risk_level      TEXT,
            confidence_score REAL,
            top_factor_1    TEXT,
            top_factor_2    TEXT,
            top_factor_3    TEXT
        )
    """)
    
    # ── Engagement log ──────────────────────────────────────
    # Weekly engagement data used for drift detection.
    # We log clicks and days_active each week.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS engagement_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT,
            week_number INTEGER,
            clicks      INTEGER,
            days_active INTEGER,
            logged_at   TIMESTAMP
        )
    """)
    
    # ── Drift log ───────────────────────────────────────────
    # Record of every drift detection check performed.
    # drift_detected: 0 = no drift, 1 = drift detected
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id     TEXT,
            checked_at     TIMESTAMP,
            z_score        REAL,
            drift_detected INTEGER,
            alert_message  TEXT
        )
    """)
    
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STUDENT CRUD OPERATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insert_student(student_id, name="", email="", teacher_email="",
                   course_id="", risk_level="Unknown", confidence_score=0.0,
                   status="Static"):
    """
    Insert a new student or update if they already exist.
    
    We use INSERT OR REPLACE (also known as UPSERT) so that:
      - New students get inserted
      - Existing students get their record updated
    This avoids duplicate key errors during batch uploads.
    """
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO students
        (student_id, name, email, teacher_email, course_id,
         risk_level, confidence_score, status, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (student_id, name, email, teacher_email, course_id,
          risk_level, confidence_score, status,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_student(student_id):
    """
    Retrieve a single student by their ID.
    Returns a dict-like Row object, or None if not found.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?",
        (str(student_id),)
    ).fetchone()
    conn.close()
    return row


def get_all_students():
    """
    Retrieve all students from the database.
    Returns a list of Row objects.
    """
    conn = get_connection()
    rows = conn.execute("SELECT * FROM students ORDER BY last_updated DESC").fetchall()
    conn.close()
    return rows


def get_students_by_teacher(teacher_email):
    """
    Get all students assigned to a specific teacher.
    Used on the Teacher Dashboard to show only their students.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM students WHERE teacher_email = ? ORDER BY risk_level",
        (teacher_email,)
    ).fetchall()
    conn.close()
    return rows


def update_student_risk(student_id, risk_level, confidence_score, status="Static"):
    """
    Update a student's risk level and confidence score.
    
    Called after:
      - A new prediction is made (status = "Static")
      - Drift is detected (status = "Dynamic Risk")
    """
    conn = get_connection()
    conn.execute("""
        UPDATE students
        SET risk_level = ?, confidence_score = ?, status = ?, last_updated = ?
        WHERE student_id = ?
    """, (risk_level, confidence_score, status,
          datetime.now().isoformat(), str(student_id)))
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENGAGEMENT LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_engagement(student_id, week_number, clicks, days_active):
    """
    Record a student's engagement for a specific week.
    
    This data feeds into drift detection (ml/drift.py).
    We track:
      - clicks: total VLE clicks that week
      - days_active: number of days they logged in
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO engagement_log (student_id, week_number, clicks, days_active, logged_at)
        VALUES (?, ?, ?, ?, ?)
    """, (str(student_id), week_number, clicks, days_active,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_engagement_history(student_id):
    """
    Fetch all engagement records for a student, ordered by week.
    
    Used by drift.py to compute the baseline and check for drops.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM engagement_log
        WHERE student_id = ?
        ORDER BY week_number ASC
    """, (str(student_id),)).fetchall()
    conn.close()
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICTION LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_prediction(student_id, risk_level, confidence_score,
                   top_factor_1="", top_factor_2="", top_factor_3=""):
    """
    Record a prediction in the history table.
    
    Every prediction is stored permanently so we can show
    "grade trajectory" charts showing how a student's risk
    has changed over time.
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO predictions
        (student_id, prediction_date, risk_level, confidence_score,
         top_factor_1, top_factor_2, top_factor_3)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (str(student_id), datetime.now().isoformat(), risk_level,
          confidence_score, top_factor_1, top_factor_2, top_factor_3))
    conn.commit()
    conn.close()


def get_prediction_history(student_id):
    """
    Fetch all past predictions for a student, ordered by date.
    Used to build grade trajectory line charts.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM predictions
        WHERE student_id = ?
        ORDER BY prediction_date ASC
    """, (str(student_id),)).fetchall()
    conn.close()
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DRIFT LOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_drift(student_id, z_score, drift_detected, alert_message=""):
    """
    Record a drift detection check result.
    
    Parameters:
      drift_detected: 1 if engagement dropped significantly, 0 otherwise
      z_score: how many standard deviations from baseline
      alert_message: human-readable explanation of the drift
    """
    conn = get_connection()
    conn.execute("""
        INSERT INTO drift_log
        (student_id, checked_at, z_score, drift_detected, alert_message)
        VALUES (?, ?, ?, ?, ?)
    """, (str(student_id), datetime.now().isoformat(),
          z_score, 1 if drift_detected else 0, alert_message))
    conn.commit()
    conn.close()


def get_drift_history(student_id):
    """
    Fetch all drift checks for a student.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM drift_log
        WHERE student_id = ?
        ORDER BY checked_at DESC
    """, (str(student_id),)).fetchall()
    conn.close()
    return rows


def get_student_count():
    """Return total number of students in the database."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    conn.close()
    return count


def get_risk_counts():
    """
    Get count of students in each risk category.
    Returns a dict like: {"High Risk": 5, "On Track": 12, ...}
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT risk_level, COUNT(*) as cnt
        FROM students
        GROUP BY risk_level
    """).fetchall()
    conn.close()
    return {row["risk_level"]: row["cnt"] for row in rows}
