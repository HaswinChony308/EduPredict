"""
4_admin_retrain.py — Admin Model Retrain
=========================================
EduPredict — Google-inspired design.
"""

import streamlit as st
import json, os, sys, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.database import init_db, get_student_count
from config import (MODEL_PATH, MODEL_METADATA_PATH, MODEL_BACKUP_PATH,
                    MODEL_PREVIOUS_PATH, ADMIN_PASSWORD, DATA_DIR)
from theme import inject_css, get_theme_colors, render_theme_toggle

init_db()
st.set_page_config(page_title="Admin Retrain — EduPredict", page_icon="⚙️", layout="wide")
inject_css()
c = get_theme_colors()

with st.sidebar:
    st.markdown(f"<span class='ep-brand'>📊 EduPredict</span><div class='ep-brand-sub'>Admin Panel</div>", unsafe_allow_html=True)
    st.markdown("---")
    render_theme_toggle()
    st.markdown("---")

st.markdown(f"<div class='ep-page-header'>⚙️ Admin — Model Retrain</div>", unsafe_allow_html=True)
st.markdown(f"<div class='ep-page-sub'>Manage model training, view metrics, and rollback versions</div>", unsafe_allow_html=True)

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if not st.session_state.admin_logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<div class='ep-card'><div class='ep-card-title'>🔒 Admin Authentication</div><div class='ep-card-desc'>Enter your admin password to access model management</div></div>", unsafe_allow_html=True)
        pw = st.text_input("Admin password", type="password", placeholder="Enter password...", key="admin_pw")
        if st.button("🔑 Authenticate", use_container_width=True, type="primary"):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid password.")
        st.markdown(f"<p style='color:{c['text_secondary']};font-size:0.78rem;margin-top:8px;'>Default password is in config.py</p>", unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown(f"<p style='color:{c['on_track']};font-weight:700;'>✅ Authenticated</p>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()

    model_exists = os.path.exists(MODEL_PATH)
    meta_exists = os.path.exists(MODEL_METADATA_PATH)

    st.markdown(f"<div class='ep-card-title'>🤖 Current Model</div>", unsafe_allow_html=True)
    if model_exists and meta_exists:
        with open(MODEL_METADATA_PATH,"r") as f:
            md = json.load(f)
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">Status</div>
                <div class="ep-metric-value" style="color:{c['on_track']};font-size:1.3rem;">● Active</div></div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">Accuracy</div>
                <div class="ep-metric-value">{md.get('accuracy',0)*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">F1 Score</div>
                <div class="ep-metric-value">{md.get('f1_score',0)*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with cols[3]:
            td = md.get("training_date","Unknown")
            try: td = datetime.fromisoformat(td).strftime("%Y-%m-%d %H:%M")
            except Exception: pass
            st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">Trained</div>
                <div class="ep-metric-value" style="font-size:0.92rem;">{td}</div></div>""", unsafe_allow_html=True)
        with st.expander("📋 Full Training Details"):
            st.json(md)
    elif model_exists:
        st.warning("Model exists but metadata missing.")
    else:
        st.error("❌ No trained model found.")

    st.markdown("---")
    st.markdown(f"<div class='ep-card-title'>📊 Data Status</div>", unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">Students in DB</div>
            <div class="ep-metric-value">{get_student_count()}</div></div>""", unsafe_allow_html=True)
    with dc2:
        oulad = ["studentInfo.csv","studentAssessment.csv","assessments.csv","studentVle.csv"]
        present = sum(1 for f in oulad if os.path.exists(os.path.join(DATA_DIR,f)))
        clr = c["on_track"] if present==4 else c["moderate_risk"]
        st.markdown(f"""<div class="ep-metric"><div class="ep-metric-label">OULAD Files</div>
            <div class="ep-metric-value" style="color:{clr};">{present}/4</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='ep-card-title'>🏋️ Retrain Model</div>", unsafe_allow_html=True)
    has_oulad = all(os.path.exists(os.path.join(DATA_DIR,f)) for f in oulad)
    if not has_oulad:
        st.warning("⚠️ OULAD not found. Run: `python data/download_oulad.py`")

    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("🔄 Retrain with OULAD", use_container_width=True, type="primary", disabled=not has_oulad):
            prog = st.progress(0)
            stat = st.empty()
            try:
                old_acc = md.get("accuracy",0) if meta_exists else 0
                old_f1 = md.get("f1_score",0) if meta_exists else 0
                if model_exists:
                    shutil.copy2(MODEL_PATH, MODEL_PREVIOUS_PATH)
                stat.text("⏳ Preprocessing..."); prog.progress(10)
                from ml.preprocess import preprocess_full_pipeline
                res = preprocess_full_pipeline()
                if res[0] is None: st.error("❌ Preprocessing failed."); st.stop()
                X,y,fn,le = res; prog.progress(40)
                stat.text("⏳ Training..."); prog.progress(50)
                from ml.train import train_model
                tr = train_model(X,y,fn,le)
                if tr is None: st.error("❌ Training failed."); st.stop()
                prog.progress(100); stat.text("✅ Done!")
                mc = st.columns(4)
                with mc[0]: st.metric("Old Accuracy",f"{old_acc*100:.1f}%")
                with mc[1]: st.metric("New Accuracy",f"{tr['accuracy']*100:.1f}%",delta=f"{(tr['accuracy']-old_acc)*100:+.1f}%")
                with mc[2]: st.metric("Old F1",f"{old_f1*100:.1f}%")
                with mc[3]: st.metric("New F1",f"{tr['f1_score']*100:.1f}%",delta=f"{(tr['f1_score']-old_f1)*100:+.1f}%")
                st.success("🎉 Model retrained!")
            except Exception as e:
                st.error(f"❌ {e}"); st.exception(e)
    with rc2:
        prev = os.path.exists(MODEL_PREVIOUS_PATH)
        if st.button("⏪ Rollback", use_container_width=True, disabled=not prev):
            try:
                if os.path.exists(MODEL_PATH): shutil.copy2(MODEL_PATH, MODEL_BACKUP_PATH)
                shutil.copy2(MODEL_PREVIOUS_PATH, MODEL_PATH)
                st.success("✅ Rolled back."); st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
        if not prev: st.info("ℹ️ No previous model available.")

    st.markdown("---")
    with st.expander("🔴 Danger Zone"):
        st.warning("⚠️ These actions are irreversible!")
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("🗑 Delete Model", use_container_width=True):
                if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH); st.success("Deleted."); st.rerun()
        with dc2:
            if st.button("🗑 Clear Database", use_container_width=True):
                dbp = os.path.join(os.path.dirname(__file__),"..","db","aapt.db")
                if os.path.exists(dbp): os.remove(dbp); init_db(); st.success("Cleared."); st.rerun()
