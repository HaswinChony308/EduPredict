"""
1_teacher_dashboard.py — Teacher Dashboard
===========================================
EduPredict — Google-inspired design with Outfit font.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.database import (init_db, get_all_students, get_students_by_teacher,
                          get_prediction_history, get_student)
from config import MODEL_PATH, RISK_CLASSES, EXPECTED_FEATURES
from alerts.email_alert import send_drift_alert
from theme import inject_css, get_theme_colors, render_theme_toggle, get_risk_color

init_db()
st.set_page_config(page_title="Teacher Dashboard — EduPredict", page_icon="👨‍🏫", layout="wide")
inject_css()
c = get_theme_colors()

with st.sidebar:
    st.markdown(f"<span class='ep-brand'>📊 EduPredict</span><div class='ep-brand-sub'>Teacher Dashboard</div>", unsafe_allow_html=True)
    st.markdown("---")
    render_theme_toggle()
    st.markdown("---")

st.markdown(f"<div class='ep-page-header'>👨‍🏫 Teacher Dashboard</div>", unsafe_allow_html=True)
st.markdown(f"<div class='ep-page-sub'>Monitor your class, view risk analysis, and take action</div>", unsafe_allow_html=True)

if "teacher_logged_in" not in st.session_state:
    st.session_state.teacher_logged_in = False

if not st.session_state.teacher_logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<div class='ep-card'><div class='ep-card-title'>Teacher Login</div><div class='ep-card-desc'>Enter your email or view all students</div></div>", unsafe_allow_html=True)
        email = st.text_input("Teacher email", placeholder="prof.smith@university.edu", key="t_email")
        lc1, lc2 = st.columns(2)
        with lc1:
            if st.button("🔑 Login", use_container_width=True, type="primary"):
                if email:
                    st.session_state.teacher_email = email
                    st.session_state.teacher_logged_in = True
                    st.rerun()
                else:
                    st.error("Please enter your email.")
        with lc2:
            if st.button("👁 View All Students", use_container_width=True):
                st.session_state.teacher_email = "__all__"
                st.session_state.teacher_logged_in = True
                st.rerun()
else:
    with st.sidebar:
        st.markdown(f"<p style='color:{c['text_secondary']};font-size:0.78rem;font-weight:500;'>Logged in as:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{c['text']};font-weight:700;font-size:0.88rem;'>{st.session_state.teacher_email}</p>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.teacher_logged_in = False
            st.rerun()
        st.markdown("---")
        risk_filter = st.selectbox("Filter by risk level", ["All"] + RISK_CLASSES + ["Dynamic Risk"])

    students = get_all_students() if st.session_state.teacher_email == "__all__" else get_students_by_teacher(st.session_state.teacher_email)
    if not students:
        students = get_all_students()
    if not students:
        st.warning("No students found. Upload data via Batch Upload page.")
        st.stop()

    sdf = pd.DataFrame([dict(s) for s in students])
    if risk_filter != "All":
        sdf = sdf[sdf["status"] == "Dynamic Risk"] if risk_filter == "Dynamic Risk" else sdf[sdf["risk_level"] == risk_filter]

    total = len(sdf)
    hr = len(sdf[sdf["risk_level"] == "High Risk"])
    mr = len(sdf[sdf["risk_level"] == "Moderate Risk"])
    ot = len(sdf[sdf["risk_level"] == "On Track"])
    ex = len(sdf[sdf["risk_level"] == "Excelling"])

    cols = st.columns(5)
    for col, (label, val, clr) in zip(cols, [("Total", total, c["text"]), ("High Risk", hr, c["high_risk"]),
        ("Moderate", mr, c["moderate_risk"]), ("On Track", ot, c["on_track"]), ("Excelling", ex, c["excelling"])]):
        with col:
            bdr = f"border-left:3px solid {clr};" if label != "Total" else ""
            st.markdown(f"""<div class="ep-metric" style="{bdr}">
                <div class="ep-metric-value" style="color:{clr};">{val}</div>
                <div class="ep-metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch_col, tb_col = st.columns([1, 2])
    with ch_col:
        st.markdown(f"<div class='ep-card-title'>📊 Risk Distribution</div>", unsafe_allow_html=True)
        rd = sdf["risk_level"].value_counts().reset_index()
        rd.columns = ["Risk Level", "Count"]
        cm = {rl: get_risk_color(rl, c) for rl in rd["Risk Level"]}
        fig = px.bar(rd, x="Risk Level", y="Count", color="Risk Level", color_discrete_map=cm, template=c["chart_template"])
        fig.update_layout(showlegend=False, plot_bgcolor=c["chart_bg"], paper_bgcolor=c["chart_paper"],
                          margin=dict(l=20,r=20,t=10,b=20), height=280, font=dict(color=c["text"], family="Inter"))
        fig.update_xaxes(gridcolor=c["chart_grid"])
        fig.update_yaxes(gridcolor=c["chart_grid"])
        st.plotly_chart(fig, use_container_width=True)
    with tb_col:
        st.markdown(f"<div class='ep-card-title'>📋 Class Overview</div>", unsafe_allow_html=True)
        ddf = sdf[["student_id","name","risk_level","confidence_score","status","last_updated"]].copy()
        ddf["confidence_score"] = ddf["confidence_score"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
        ddf.columns = ["ID","Name","Risk Level","Confidence","Status","Updated"]
        st.dataframe(ddf, use_container_width=True, hide_index=True, height=280)

    st.download_button("📥 Download Class Report", data=sdf.to_csv(index=False),
                       file_name="edupredict_class_report.csv", mime="text/csv", type="primary")
    st.markdown("---")
    st.markdown(f"<div class='ep-card-title'>🔍 Individual Student Analysis</div>", unsafe_allow_html=True)
    opts = {f"{r['student_id']} — {r['name']} ({r['risk_level']})": r['student_id'] for _, r in sdf.iterrows()}
    sel = st.selectbox("Choose a student", list(opts.keys()))
    if sel:
        sid = opts[sel]
        student = get_student(sid)
        if student:
            sd = dict(student)
            rc = get_risk_color(sd.get("risk_level",""), c)
            st.markdown(f"""<div class="ep-card" style="border-left:4px solid {rc};">
                <div style="font-family:Outfit,sans-serif;font-size:1.25rem;font-weight:800;color:{c['text']};">{sd.get('name',sid)}</div>
                <p style="color:{c['text_secondary']};font-size:0.85rem;margin:6px 0;">ID: {sid} &nbsp;|&nbsp; Course: {sd.get('course_id','N/A')}</p>
                <span class="ep-badge" style="background:{rc}15;color:{rc};border:1.5px solid {rc};">{sd.get('risk_level','Unknown')}</span>
                <span style="color:{c['text_secondary']};margin-left:12px;font-weight:500;">Confidence: <strong>{sd.get('confidence_score',0):.1%}</strong></span>
            </div>""", unsafe_allow_html=True)

            dc1, dc2 = st.columns(2)
            pred_h = get_prediction_history(sid)
            with dc1:
                st.markdown(f"<div class='ep-card-title'>🧠 SHAP Top Factors</div>", unsafe_allow_html=True)
                if pred_h:
                    latest = dict(pred_h[-1])
                    factors = []
                    for i in range(1,4):
                        f = latest.get(f"top_factor_{i}","")
                        if f:
                            parts = f.split(":",1)
                            factors.append({"feature":parts[0].strip(),"suggestion":parts[1].strip() if len(parts)>1 else f,"impact":1.0/i})
                    if factors:
                        fig_s = go.Figure(go.Bar(x=[f["impact"] for f in factors], y=[f["feature"] for f in factors],
                            orientation='h', marker_color=[c["high_risk"],c["moderate_risk"],c["excelling"]]))
                        fig_s.update_layout(template=c["chart_template"],plot_bgcolor=c["chart_bg"],paper_bgcolor=c["chart_paper"],
                            margin=dict(l=10,r=10,t=10,b=10),height=180,showlegend=False,font=dict(color=c["text"],family="Inter"))
                        st.plotly_chart(fig_s, use_container_width=True)
                        for f in factors:
                            st.markdown(f"- **{f['feature']}**: {f['suggestion']}")
                    else:
                        st.info("No SHAP factors recorded.")
                else:
                    st.info("No predictions yet.")
            with dc2:
                st.markdown(f"<div class='ep-card-title'>📈 Grade Trajectory</div>", unsafe_allow_html=True)
                if pred_h:
                    hdf = pd.DataFrame([dict(p) for p in pred_h])
                    r2n = {"High Risk":1,"Moderate Risk":2,"On Track":3,"Excelling":4}
                    hdf["risk_numeric"] = hdf["risk_level"].map(r2n)
                    hdf["prediction_date"] = pd.to_datetime(hdf["prediction_date"])
                    fig_t = go.Figure(go.Scatter(x=hdf["prediction_date"],y=hdf["risk_numeric"],mode="lines+markers",
                        line=dict(color=c["primary"],width=2),marker=dict(size=8),text=hdf["risk_level"],
                        hovertemplate="Date: %{x}<br>Level: %{text}<extra></extra>"))
                    fig_t.update_layout(template=c["chart_template"],plot_bgcolor=c["chart_bg"],paper_bgcolor=c["chart_paper"],
                        margin=dict(l=10,r=10,t=10,b=10),height=180,yaxis=dict(tickmode="array",tickvals=[1,2,3,4],
                        ticktext=["High Risk","Moderate","On Track","Excelling"]),font=dict(color=c["text"],family="Inter"))
                    fig_t.update_xaxes(gridcolor=c["chart_grid"])
                    fig_t.update_yaxes(gridcolor=c["chart_grid"])
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.info("No prediction history.")

            if st.button(f"📧 Send Alert for {sd.get('name',sid)}", type="primary"):
                result = send_drift_alert(sd.get("name",sid), sd.get("email",""), st.session_state.teacher_email, 0.0, [])
                st.success(f"✅ {result['message']}") if result["success"] else st.error(f"❌ {result['message']}")
