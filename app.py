"""
app.py — EduPredict Landing Page
=================================
Main entry point with Google-style design, Outfit font, and
premium elevated cards.
"""

import streamlit as st
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH, MODEL_METADATA_PATH
from db.database import init_db, get_student_count, get_risk_counts
from theme import inject_css, get_theme_colors, render_theme_toggle, get_risk_color

st.set_page_config(page_title="EduPredict", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
init_db()
inject_css()
c = get_theme_colors()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 8px;">
        <span class="ep-brand">📊 EduPredict</span>
        <div class="ep-brand-sub">The Insight Engine<br/>Educational Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    render_theme_toggle()
    st.markdown("---")
    st.page_link("app.py", label="🏠 Home", use_container_width=True)
    st.page_link("pages/1_teacher_dashboard.py", label="👨‍🏫 Teacher Dashboard", use_container_width=True)
    st.page_link("pages/2_student_portal.py", label="🎓 Student Portal", use_container_width=True)
    st.page_link("pages/3_batch_upload.py", label="📤 Batch Upload", use_container_width=True)
    st.page_link("pages/4_admin_retrain.py", label="⚙️ Admin Retrain", use_container_width=True)

# ── Hero Section ────────────────────────────────────────────
model_exists = os.path.exists(MODEL_PATH)

st.markdown(f"""
<div class="ep-hero">
    <div class="ep-hero-title">EduPredict</div>
    <div class="ep-hero-desc">
        Predict student performance using machine learning, detect
        engagement drift, and provide explainable insights with
        SHAP analysis — all in one intelligent dashboard.
    </div>
    <div style="margin-top:22px;">
        {"<span class='ep-status ep-status-green'><span>●</span> Model loaded &amp; ready</span>" if model_exists else "<span class='ep-status ep-status-red'><span>●</span> No model found — train first</span>"}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Stats ─────────────────────────────────────────────
total = get_student_count()
rc = get_risk_counts()
stats = [
    ("Total Students", total, c["text"]),
    ("High Risk",      rc.get("High Risk", 0),      c["high_risk"]),
    ("Moderate Risk",  rc.get("Moderate Risk", 0),   c["moderate_risk"]),
    ("On Track",       rc.get("On Track", 0),        c["on_track"]),
    ("Excelling",      rc.get("Excelling", 0),       c["excelling"]),
]
cols = st.columns(5)
for col, (label, value, color) in zip(cols, stats):
    with col:
        bdr = f"border-left:3px solid {color};" if label != "Total Students" else ""
        st.markdown(f"""
        <div class="ep-metric" style="{bdr}">
            <div class="ep-metric-value" style="color:{color};">{value}</div>
            <div class="ep-metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation Cards ────────────────────────────────────────
st.markdown(f"<div class='ep-page-header'>Navigate</div>", unsafe_allow_html=True)
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

nav = [
    ("👨‍🏫", "Teacher Dashboard", "View class-wide analytics, risk distributions, individual student SHAP explanations, and send targeted interventions.", "teacher_dashboard"),
    ("🎓", "Student Portal", "Students can view their own risk level, grade trajectory, top factors affecting their score, and personalized advice.", "student_portal"),
    ("📤", "Batch CSV Upload", "Upload a class CSV to run bulk predictions, detect drift for all students, and download results instantly.", "batch_upload"),
    ("⚙️", "Admin — Retrain", "Password-protected model management. Retrain, compare metrics, and rollback to previous versions.", "admin_retrain"),
]

col1, col2 = st.columns(2)
for i, (icon, title, desc, link) in enumerate(nav):
    with [col1, col2][i % 2]:
        st.markdown(f"""
        <a href="{link}" target="_self" style="text-decoration:none;">
            <div class="ep-nav-card">
                <div class="ep-nav-icon">{icon}</div>
                <div class="ep-nav-title">{title}</div>
                <div class="ep-nav-desc">{desc}</div>
            </div>
        </a>""", unsafe_allow_html=True)
    if i == 1:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

# ── Model Info ──────────────────────────────────────────────
if model_exists and os.path.exists(MODEL_METADATA_PATH):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='ep-page-header'>Current Model</div>", unsafe_allow_html=True)
    with open(MODEL_METADATA_PATH, "r") as f:
        md = json.load(f)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Accuracy", f"{md.get('accuracy',0)*100:.1f}%")
    with mc2:
        st.metric("F1 Score (Macro)", f"{md.get('f1_score',0)*100:.1f}%")
    with mc3:
        st.metric("Training Samples", f"{md.get('num_samples',0):,}")

# ── Footer ──────────────────────────────────────────────────
st.markdown(f"""
<div class="ep-footer">
    <p><strong>Dataset:</strong> Open University Learning Analytics Dataset (OULAD)</p>
    <p><a href="https://analyse.kmi.open.ac.uk/open_dataset" target="_blank">analyse.kmi.open.ac.uk/open_dataset</a> — CC-BY 4.0</p>
    <br>
    <p>EduPredict — Adaptive Academic Performance Tracking | Mini Project</p>
</div>
""", unsafe_allow_html=True)
