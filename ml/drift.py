"""
drift.py — Phase 3: Behavioral Drift Detection for AAPT
========================================================
This is the "Adaptive" part of AAPT. Instead of just giving a
static prediction, we continuously monitor student engagement
and detect sudden drops.

What is "drift"?
  Drift means a student's behavior has changed significantly
  compared to their own baseline. For example:
    - A student who normally clicks 50 times/week suddenly drops to 5
    - This could indicate they're struggling, lost motivation, or
      have personal issues affecting their studies

How we detect it — Z-score method:
  1. Establish a BASELINE from the student's early engagement
     (first 20% of their recorded weeks)
  2. Calculate the baseline mean (E_avg) and standard deviation (std_dev)
  3. For the current week, compute:
       Z = (current_clicks - E_avg) / std_dev
  4. If |Z| > 2.0, the engagement is MORE than 2 standard deviations
     away from normal → flag as drift

Why Z-score?
  - It's relative to each student's OWN baseline (not class average)
  - A student who always clicks 10/week dropping to 5 is more alarming
    than a student who clicks 200/week dropping to 150
  - Z > 2.0 means there's roughly a <5% chance this drop is random
    (assuming normal distribution)

What happens when drift is detected:
  - Student's status changes from "Static" → "Dynamic Risk"
  - An alert can be sent to the teacher and student
  - This overrides the ML model's static prediction
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Z_SCORE_THRESHOLD, BASELINE_PERIOD_FRACTION


def detect_drift(student_id, current_week_clicks, db_module=None):
    """
    Check if a student's engagement has drifted from their baseline.
    
    Parameters:
      student_id: unique identifier for the student
      current_week_clicks: number of VLE clicks this week
      db_module: the database module (imported separately to avoid
                 circular imports). If None, we import it here.
    
    Returns:
      dict with keys:
        - drift_detected: bool (True if engagement dropped significantly)
        - z_score: float (how many std devs from baseline)
        - alert_message: str (human-readable explanation)
        - baseline_mean: float (the student's normal engagement level)
        - baseline_std: float (normal variation in engagement)
    """
    # Import database module if not provided
    if db_module is None:
        from db.database import (get_engagement_history, log_drift,
                                  update_student_risk)
    else:
        get_engagement_history = db_module.get_engagement_history
        log_drift = db_module.log_drift
        update_student_risk = db_module.update_student_risk
    
    # ── Step 1: Fetch engagement history ────────────────────
    # Get all past weekly click records for this student
    history = get_engagement_history(student_id)
    
    # Edge case: not enough history to establish a baseline
    if len(history) < 3:
        result = {
            "drift_detected": False,
            "z_score": 0.0,
            "alert_message": "Insufficient history for drift detection (need at least 3 weeks).",
            "baseline_mean": 0.0,
            "baseline_std": 0.0
        }
        # Still log the check
        log_drift(student_id, 0.0, False, result["alert_message"])
        return result
    
    # ── Step 2: Extract click values from history ───────────
    # Each row in history has a "clicks" column
    click_values = [row["clicks"] for row in history]
    
    # ── Step 3: Compute baseline from first 20% of weeks ────
    # Why first 20%? We assume the student starts with normal behavior.
    # Using early data as baseline avoids contamination from the drift itself.
    baseline_count = max(1, int(len(click_values) * BASELINE_PERIOD_FRACTION))
    baseline_data = click_values[:baseline_count]
    
    # Baseline mean = average engagement in the early period
    E_avg = np.mean(baseline_data)
    
    # Baseline std_dev = how much their engagement normally varies
    std_dev = np.std(baseline_data)
    
    # ── Step 4: Handle edge case — zero standard deviation ──
    # If the student had EXACTLY the same clicks every week in the
    # baseline period, std_dev = 0 and we'd get division by zero.
    # We use a small epsilon to prevent this.
    if std_dev < 1e-6:
        # If std is essentially 0, any change is significant
        # We set it to 1 so that a 1-click change = 1 Z-score
        std_dev = 1.0
    
    # ── Step 5: Compute Z-score ─────────────────────────────
    # Z = (observed - expected) / standard_deviation
    # Negative Z = engagement DROPPED (bad)
    # Positive Z = engagement INCREASED (good, usually)
    # We care about drops, so we check if Z < -threshold
    z_score = (current_week_clicks - E_avg) / std_dev
    
    # ── Step 6: Check if drift is detected ──────────────────
    # |Z| > 2.0 means the current value is far from normal
    # We specifically check for DROPS (negative Z)
    drift_detected = abs(z_score) > Z_SCORE_THRESHOLD
    
    # Build alert message
    if drift_detected and z_score < 0:
        alert_message = (
            f"⚠️ Engagement dropped significantly this week. "
            f"Current clicks: {current_week_clicks}, "
            f"Baseline average: {E_avg:.1f}, "
            f"Z-score: {z_score:.2f} "
            f"(>{Z_SCORE_THRESHOLD} standard deviations below normal)."
        )
    elif drift_detected and z_score > 0:
        alert_message = (
            f"📈 Engagement increased significantly this week. "
            f"Current clicks: {current_week_clicks}, "
            f"Baseline average: {E_avg:.1f}, "
            f"Z-score: {z_score:.2f}."
        )
    else:
        alert_message = (
            f"✅ Engagement is within normal range. "
            f"Z-score: {z_score:.2f}."
        )
    
    # ── Step 7: Log the drift check ─────────────────────────
    log_drift(student_id, z_score, drift_detected, alert_message)
    
    # ── Step 8: Update student status if drift detected ─────
    # If a significant DROP is detected, override the student's
    # status to "Dynamic Risk" regardless of the model prediction
    if drift_detected and z_score < 0:
        update_student_risk(
            student_id,
            risk_level="High Risk",    # Treat as high risk
            confidence_score=0.0,      # Confidence is 0 (drift-based, not model-based)
            status="Dynamic Risk"      # Mark as dynamically flagged
        )
    
    result = {
        "drift_detected": drift_detected,
        "z_score": round(z_score, 4),
        "alert_message": alert_message,
        "baseline_mean": round(E_avg, 2),
        "baseline_std": round(std_dev, 2)
    }
    
    return result


def check_all_students_drift(db_module=None):
    """
    Run drift detection for all students in the database.
    
    This is used for batch checking (e.g., weekly automated check).
    For each student, it uses their most recent engagement entry
    as the "current week" data.
    
    Returns:
      list of dicts with student_id and drift results
    """
    if db_module is None:
        from db.database import get_all_students, get_engagement_history
    else:
        get_all_students = db_module.get_all_students
        get_engagement_history = db_module.get_engagement_history
    
    students = get_all_students()
    results = []
    
    for student in students:
        sid = student["student_id"]
        history = get_engagement_history(sid)
        
        if len(history) > 0:
            # Use the latest week's clicks as "current"
            latest_clicks = history[-1]["clicks"]
            drift_result = detect_drift(sid, latest_clicks, db_module)
            drift_result["student_id"] = sid
            results.append(drift_result)
    
    return results
