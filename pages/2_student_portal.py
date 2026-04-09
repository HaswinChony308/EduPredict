"""
2_student_portal.py — Student Portal
======================================
EduPredict — Google-inspired design.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.database import init_db, get_student, get_prediction_history, get_drift_history
from theme import inject_css, get_theme_colors, render_theme_toggle, get_risk_color

init_db()
st.set_page_config(page_title="Student Portal — EduPredict", page_icon="🎓", layout="wide")
inject_css()
c = get_theme_colors()

MOTIVATIONAL = {
    "Excelling":     ("🌟","Outstanding work! You're performing exceptionally. Keep this momentum — your dedication is truly inspiring. Consider helping peers who might need support!"),
    "On Track":      ("👍","You're doing great! You're on the right path. Keep maintaining your consistent effort — small improvements each day lead to big results!"),
    "Moderate Risk": ("💪","You've got potential — let's unlock it! A bit more focus can make a big difference. Reach out to your teacher or peers for support. You can turn this around!"),
    "High Risk":     ("🤝","We're here for you. Everyone faces tough times, and what matters is taking that first step forward. Talk to your teacher, visit office hours, or connect with a study group. You CAN do this!"),
}

with st.sidebar:
    st.markdown(f"<span class='ep-brand'>📊 EduPredict</span><div class='ep-brand-sub'>Student Portal</div>", unsafe_allow_html=True)
    st.markdown("---")
    render_theme_toggle()
    st.markdown("---")

st.markdown(f"<div class='ep-page-header'>🎓 Student Portal</div>", unsafe_allow_html=True)
st.markdown(f"<div class='ep-page-sub'>View your performance analysis and personalized recommendations</div>", unsafe_allow_html=True)

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if not st.session_state.student_logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<div class='ep-card'><div class='ep-card-title'>Student Login</div><div class='ep-card-desc'>Enter your Student ID to view your performance dashboard</div></div>", unsafe_allow_html=True)
        sid = st.text_input("Student ID", placeholder="e.g., STU001", key="s_id")
        if st.button("🔑 Login", use_container_width=True, type="primary"):
            if sid:
                student = get_student(sid)
                if student:
                    st.session_state.student_id = sid
                    st.session_state.student_logged_in = True
                    st.rerun()
                else:
                    st.error(f"Student ID '{sid}' not found.")
            else:
                st.error("Please enter your Student ID.")
else:
    with st.sidebar:
        st.markdown(f"<p style='color:{c['text_secondary']};font-size:0.78rem;font-weight:500;'>Student ID:</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{c['text']};font-weight:700;'>{st.session_state.student_id}</p>", unsafe_allow_html=True)
        if st.button("🚪 Logout"):
            st.session_state.student_logged_in = False
            st.rerun()

    student = get_student(st.session_state.student_id)
    if not student:
        st.error("Student record not found.")
        st.stop()

    sd = dict(student)
    risk = sd.get("risk_level","Unknown")
    conf = sd.get("confidence_score",0)
    status = sd.get("status","Static")
    rc = get_risk_color(risk, c)

    if status == "Dynamic Risk":
        st.warning("⚠️ **Engagement drop detected.** Our system noticed a significant change in your activity. Please reach out to your teacher for support.")

    ic1, ic2 = st.columns([2, 1])
    with ic1:
        st.markdown(f"""<div class="ep-info-card">
            <div style="font-family:Outfit,sans-serif;font-size:1.6rem;font-weight:800;color:{c['text']};margin-bottom:10px;">
                Welcome back, {sd.get('name', st.session_state.student_id)}!
            </div>
            <p style="color:{c['text_secondary']};font-size:0.88rem;">
                Course: {sd.get('course_id','N/A')} &nbsp;|&nbsp; Last Updated: {sd.get('last_updated','N/A')}
            </p>
            <span class="ep-badge" style="background:{rc}18;color:{rc};border:1.5px solid {rc};padding:8px 22px;font-size:1rem;">
                {risk}
            </span>
            <span style="display:inline-block;padding:8px 16px;border-radius:50px;font-size:0.82rem;margin-left:10px;
                         background:{c['surface']};color:{c['text_secondary']};border:1px solid {c['border']};">
                Status: {status}
            </span>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""<div class="ep-info-card" style="text-align:center;">
            <div class="ep-info-label">Confidence Score</div>
        </div>""", unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, conf)))
        st.markdown(f"<p style='text-align:center;color:{rc};font-family:Outfit,sans-serif;font-size:2.2rem;font-weight:800;'>{conf:.1%}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='ep-card-title'>📈 Your Grade Trajectory</div>", unsafe_allow_html=True)
    pred_h = get_prediction_history(st.session_state.student_id)
    if pred_h:
        hdf = pd.DataFrame([dict(p) for p in pred_h])
        r2n = {"High Risk":1,"Moderate Risk":2,"On Track":3,"Excelling":4}
        hdf["risk_numeric"] = hdf["risk_level"].map(r2n)
        hdf["prediction_date"] = pd.to_datetime(hdf["prediction_date"])
        try:
            fill = f"rgba{tuple(list(int(rc.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.08])}"
        except Exception:
            fill = "rgba(0,0,0,0.05)"
        fig = go.Figure(go.Scatter(x=hdf["prediction_date"],y=hdf["risk_numeric"],fill="tozeroy",fillcolor=fill,
            line=dict(color=rc,width=3),mode="lines+markers",marker=dict(size=10,color=rc),text=hdf["risk_level"],
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Level: %{text}<extra></extra>"))
        fig.update_layout(template=c["chart_template"],plot_bgcolor=c["chart_bg"],paper_bgcolor=c["chart_paper"],
            margin=dict(l=20,r=20,t=10,b=20),height=280,yaxis=dict(tickmode="array",tickvals=[1,2,3,4],
            ticktext=["High Risk","Moderate Risk","On Track","Excelling"],range=[0.5,4.5]),showlegend=False,
            font=dict(color=c["text"],family="Inter"))
        fig.update_xaxes(gridcolor=c["chart_grid"])
        fig.update_yaxes(gridcolor=c["chart_grid"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Your grade trajectory will appear here after predictions are made.")

    st.markdown("---")
    st.markdown(f"<div class='ep-card-title'>🔍 Top 3 Factors Affecting Your Score</div>", unsafe_allow_html=True)
    if pred_h:
        latest = dict(pred_h[-1])
        factors = []
        for i in range(1,4):
            f = latest.get(f"top_factor_{i}","")
            if f:
                parts = f.split(":",1)
                factors.append({"feature":parts[0].strip(),"suggestion":parts[1].strip() if len(parts)>1 else f,"impact":1.0/i})
        if factors:
            fig_f = go.Figure(go.Bar(x=[f["impact"] for f in factors],y=[f["feature"] for f in factors],
                orientation="h",marker=dict(color=[c["high_risk"],c["moderate_risk"],c["excelling"]])))
            fig_f.update_layout(template=c["chart_template"],plot_bgcolor=c["chart_bg"],paper_bgcolor=c["chart_paper"],
                margin=dict(l=10,r=10,t=10,b=10),height=180,xaxis_title="Impact",showlegend=False,
                font=dict(color=c["text"],family="Inter"))
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("#### 💡 Suggestions")
            for i, f in enumerate(factors, 1):
                icons = ["🔴","🟡","🔵"]
                st.markdown(f"{icons[i-1]} **{f['feature']}**: {f['suggestion']}")
        else:
            st.info("Factor analysis available after predictions are generated.")
    else:
        st.info("💡 Your personalized factors will appear after predictions.")

    st.markdown("---")
    emoji, msg = MOTIVATIONAL.get(risk, MOTIVATIONAL["On Track"])
    st.markdown(f"""<div class="ep-motivation">
        <div class="ep-motivation-emoji">{emoji}</div>
        <div class="ep-motivation-text">{msg}</div>
    </div>""", unsafe_allow_html=True)

    drift_h = get_drift_history(st.session_state.student_id)
    if drift_h:
        with st.expander("📋 Engagement Check History"):
            ddf = pd.DataFrame([dict(d) for d in drift_h])
            ddf = ddf[["checked_at","z_score","drift_detected","alert_message"]].copy()
            ddf["drift_detected"] = ddf["drift_detected"].map({1:"⚠️ Yes",0:"✅ No"})
            ddf.columns = ["Date","Z-Score","Drift?","Details"]
            st.dataframe(ddf, use_container_width=True, hide_index=True)
