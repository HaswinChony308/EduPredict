"""
explain.py — Phase 4: SHAP Explainability for AAPT
===================================================
This module makes our ML model's predictions INTERPRETABLE.

What is SHAP?
  SHAP (SHapley Additive exPlanations) is a method from game theory
  that explains which features contributed most to a prediction.
  
  Imagine a prediction is a team effort — SHAP calculates how much
  each "player" (feature) contributed to the final "score" (prediction).
  
  For example, if a student is predicted "High Risk", SHAP might say:
    - "sum_clicks = 5"      contributed +0.3 toward High Risk
    - "weighted_avg_score = 20" contributed +0.25 toward High Risk
    - "days_active = 2"     contributed +0.15 toward High Risk
  
  This tells the teacher exactly WHY the student is at risk,
  not just THAT they are at risk.

Why TreeExplainer?
  Our model is a tree-based ensemble (Random Forest + Gradient Boosting).
  SHAP's TreeExplainer is specifically optimized for tree models — it's
  fast and gives exact SHAP values (not approximations).
"""

import os
import sys
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PATH, FEATURE_NAMES_PATH, RISK_CLASSES

# Try to import SHAP — it might not be installed yet
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

import joblib


# ──────────────────────────────────────────────────────────
# HUMAN-READABLE SUGGESTIONS
# ──────────────────────────────────────────────────────────
# Maps feature names to actionable advice for teachers/students.
# These are shown alongside SHAP explanations in the dashboard.
FEATURE_SUGGESTIONS = {
    "sum_clicks": {
        "low": "Student has low LMS interaction. Recommend sending course reminder email.",
        "high": "Student is actively engaging with the LMS. Keep up the good work!",
    },
    "weighted_avg_score": {
        "low": "Assessment scores are declining. Schedule a one-on-one session.",
        "high": "Assessment performance is strong. Consider advanced materials.",
    },
    "days_active": {
        "low": "Student has not logged in recently. Send engagement nudge.",
        "high": "Student is consistently active. Positive sign!",
    },
    "click_trend": {
        "low": "Weekly activity is decreasing. Check if student needs support.",
        "high": "Activity trend is positive — student is increasingly engaged.",
    },
    "num_of_prev_attempts": {
        "low": "First attempt at this course.",
        "high": "Student has retaken before. Provide additional learning resources.",
    },
    "avg_clicks_per_day": {
        "low": "Daily engagement is very low. Consider a personalized study plan.",
        "high": "Good daily engagement levels.",
    },
    "last_active_day": {
        "low": "Student stopped engaging early in the course. Urgent follow-up needed.",
        "high": "Student was active until recently.",
    },
    "studied_credits": {
        "low": "Light course load may indicate part-time status.",
        "high": "Heavy course load — check if student is overwhelmed.",
    },
    "gender": {
        "low": "Gender factor — consider if support programs are needed.",
        "high": "Gender factor — consider if support programs are needed.",
    },
    "age_band": {
        "low": "Younger student — may benefit from study skills workshops.",
        "high": "Mature student — may have work/family commitments to consider.",
    },
    "region": {
        "low": "Regional factor detected.",
        "high": "Regional factor detected.",
    },
    "highest_education": {
        "low": "Lower prior education — may need foundational support.",
        "high": "Strong educational background.",
    },
    "imd_band": {
        "low": "Student from a more deprived area. Check if financial support is available.",
        "high": "Socioeconomic factor is positive.",
    },
    "disability": {
        "low": "No disability reported.",
        "high": "Student has a disability. Ensure accessibility accommodations are in place.",
    },
}


def get_suggestion(feature_name, shap_value):
    """
    Get a human-readable suggestion based on the feature and its SHAP value.
    
    Parameters:
      feature_name: name of the feature (e.g., "sum_clicks")
      shap_value: SHAP value (positive = pushes toward risk, negative = protective)
    
    Returns:
      str: actionable suggestion text
    """
    if feature_name in FEATURE_SUGGESTIONS:
        # If SHAP value is positive for a risk class, the feature is a risk factor
        # If negative, it's protective
        direction = "high" if shap_value > 0 else "low"
        return FEATURE_SUGGESTIONS[feature_name].get(
            direction,
            f"Feature '{feature_name}' is influencing the prediction."
        )
    return f"Feature '{feature_name}' is influencing the prediction."


def get_shap_explanation(student_features_df, model=None):
    """
    Generate SHAP explanation for a single student's prediction.
    
    Parameters:
      student_features_df: DataFrame with one row, columns = feature names
      model: trained VotingClassifier (loaded from disk if None)
    
    Returns:
      dict with:
        - top_3_risk_factors: list of (feature_name, shap_value, suggestion)
        - shap_plot_base64: base64-encoded PNG of SHAP bar chart
        - full_shap_values: dict of feature→shap_value
        - predicted_class: the predicted risk level
        - prediction_probabilities: dict of class→probability
    """
    # ── Step 1: Load model if not provided ──────────────────
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return {
                "error": "Model not found. Train the model first.",
                "top_3_risk_factors": [],
                "shap_plot_base64": "",
                "full_shap_values": {},
            }
        model = joblib.load(MODEL_PATH)
    
    # Load feature names
    if os.path.exists(FEATURE_NAMES_PATH):
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    else:
        feature_names = list(student_features_df.columns)
    
    # ── Step 2: Make prediction ─────────────────────────────
    X = student_features_df.values.reshape(1, -1) if len(student_features_df.shape) == 1 \
        else student_features_df.values
    
    predicted_class_idx = model.predict(X)[0]
    predicted_proba = model.predict_proba(X)[0]
    
    # Decode class index to name
    try:
        from config import ENCODERS_PATH
        encoders = joblib.load(ENCODERS_PATH)
        le = encoders["label_encoder"]
        predicted_class = le.inverse_transform([predicted_class_idx])[0]
        class_names = list(le.classes_)
    except Exception:
        predicted_class = RISK_CLASSES[predicted_class_idx] if predicted_class_idx < len(RISK_CLASSES) else str(predicted_class_idx)
        class_names = RISK_CLASSES
    
    # Build probability dict
    prediction_probabilities = {}
    for i, prob in enumerate(predicted_proba):
        if i < len(class_names):
            prediction_probabilities[class_names[i]] = round(float(prob), 4)
    
    # ── Step 3: Compute SHAP values ─────────────────────────
    if not SHAP_AVAILABLE:
        # Return prediction without SHAP if library not available
        return {
            "error": "SHAP library not installed.",
            "top_3_risk_factors": [],
            "shap_plot_base64": "",
            "full_shap_values": {},
            "predicted_class": predicted_class,
            "prediction_probabilities": prediction_probabilities,
        }
    
    try:
        # Access the Random Forest part of the VotingClassifier
        # model.estimators_[0] = the Random Forest we trained
        # We use RF for SHAP because TreeExplainer works best with it
        rf_model = model.estimators_[0]
        
        # TreeExplainer computes EXACT SHAP values for tree models
        # (no approximation needed — this is mathematically precise)
        explainer = shap.TreeExplainer(rf_model)
        
        # Compute SHAP values for this student
        shap_values = explainer.shap_values(X)
        
        # shap_values is a list of arrays (one per class for multiclass)
        # We want the SHAP values for the PREDICTED class
        if isinstance(shap_values, list):
            # For multiclass: shap_values[class_idx] has shape (1, n_features)
            shap_for_predicted = shap_values[predicted_class_idx][0]
        else:
            shap_for_predicted = shap_values[0]
        
    except Exception as e:
        # Fallback: use feature importances from the model instead
        print(f"⚠️ SHAP computation failed: {e}. Using feature importances instead.")
        try:
            rf_model = model.estimators_[0]
            importances = rf_model.feature_importances_
            shap_for_predicted = importances
        except Exception:
            shap_for_predicted = np.zeros(len(feature_names))
    
    # ── Step 4: Build feature→SHAP value mapping ───────────
    full_shap_values = {}
    for i, fname in enumerate(feature_names):
        if i < len(shap_for_predicted):
            full_shap_values[fname] = round(float(shap_for_predicted[i]), 4)
    
    # ── Step 5: Find top 3 risk factors ─────────────────────
    # Sort by absolute SHAP value (highest impact first)
    sorted_features = sorted(
        full_shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    top_3_risk_factors = []
    for fname, sval in sorted_features[:3]:
        suggestion = get_suggestion(fname, sval)
        top_3_risk_factors.append({
            "feature": fname,
            "shap_value": sval,
            "suggestion": suggestion
        })
    
    # ── Step 6: Generate SHAP bar chart ─────────────────────
    shap_plot_base64 = ""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort features by SHAP value for the bar chart
        sorted_names = [f for f, _ in sorted_features]
        sorted_vals = [v for _, v in sorted_features]
        
        # Color bars: red for positive (risk), blue for protective
        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sorted_vals]
        
        ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Feature Impact — Predicted: {predicted_class}")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        
        # Set dark background to match dashboard theme
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        
        plt.tight_layout()
        
        # Convert plot to base64 PNG for embedding in Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#0e1117")
        buf.seek(0)
        shap_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
    except Exception as e:
        print(f"⚠️ Failed to generate SHAP plot: {e}")
    
    return {
        "top_3_risk_factors": top_3_risk_factors,
        "shap_plot_base64": shap_plot_base64,
        "full_shap_values": full_shap_values,
        "predicted_class": predicted_class,
        "prediction_probabilities": prediction_probabilities,
    }
