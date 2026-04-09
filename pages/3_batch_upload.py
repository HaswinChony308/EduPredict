"""
3_batch_upload.py — Batch CSV Upload
======================================
EduPredict — Google-inspired design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.database import (init_db, insert_student, log_prediction, log_engagement)
from config import MODEL_PATH, EXPECTED_FEATURES, RISK_CLASSES, SAMPLE_CSV_PATH, ENCODERS_PATH
from ml.preprocess import preprocess_batch
from ml.drift import detect_drift
from alerts.email_alert import send_drift_alert
from theme import inject_css, get_theme_colors, render_theme_toggle, get_risk_color

init_db()
st.set_page_config(page_title="Batch Upload — EduPredict", page_icon="📤", layout="wide")
inject_css()
c = get_theme_colors()

with st.sidebar:
    st.markdown(f"<span class='ep-brand'>📊 EduPredict</span><div class='ep-brand-sub'>Batch CSV Upload</div>", unsafe_allow_html=True)
    st.markdown("---")
    render_theme_toggle()
    st.markdown("---")

st.markdown(f"<div class='ep-page-header'>📤 Batch CSV Upload</div>", unsafe_allow_html=True)
st.markdown(f"<div class='ep-page-sub'>Upload a class CSV to run bulk predictions and detect engagement drift</div>", unsafe_allow_html=True)

with st.expander("📋 Required CSV Format", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Required columns:**")
        for col in ["student_id","name","email"] + EXPECTED_FEATURES:
            st.markdown(f"- `{col}`")
    with c2:
        st.markdown("**Optional:** `teacher_email`, `course_id`")
        st.markdown("**Notes:** `gender`: 0=M,1=F | `sum_clicks`: total VLE clicks")
    if os.path.exists(SAMPLE_CSV_PATH):
        with open(SAMPLE_CSV_PATH,"r") as f:
            st.download_button("📥 Download Sample Template", data=f.read(), file_name="sample_students.csv", mime="text/csv")

st.markdown(f"<div class='ep-card-title'>📁 Upload Your CSV</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_uploader")
use_sample = st.checkbox("🧪 Use sample data instead (for testing)")

if uploaded is not None or use_sample:
    try:
        if use_sample:
            if os.path.exists(SAMPLE_CSV_PATH):
                df = pd.read_csv(SAMPLE_CSV_PATH)
                st.info(f"Using sample data: {len(df)} students")
            else:
                st.error("Sample CSV not found.")
                st.stop()
        else:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Uploaded: {len(df)} rows, {len(df.columns)} columns")

        missing = [col for col in ["student_id"]+EXPECTED_FEATURES if col not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            st.stop()
        st.success("✅ All required columns present!")

        with st.expander("👁 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        st.markdown(f"<div class='ep-card-title'>🤖 Running Predictions</div>", unsafe_allow_html=True)
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model not found! Run: `python -X utf8 ml/train_sample.py`")
            st.stop()

        import joblib
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODERS_PATH)["label_encoder"] if os.path.exists(ENCODERS_PATH) else None

        prog = st.progress(0)
        stat = st.empty()
        stat.text("⏳ Preprocessing...")
        prog.progress(10)
        X = preprocess_batch(df)
        prog.progress(30)
        stat.text("⏳ Predicting...")
        preds = model.predict(X)
        probs = model.predict_proba(X)
        prog.progress(50)

        df["risk_level"] = le.inverse_transform(preds) if le else [RISK_CLASSES[p] if p<len(RISK_CLASSES) else "Unknown" for p in preds]
        df["confidence_score"] = np.max(probs, axis=1)
        df["status"] = "Static"
        prog.progress(60)

        stat.text("⏳ Computing explanations...")
        top_factors_list = []
        try:
            from ml.explain import get_shap_explanation
            for idx in range(len(df)):
                try:
                    sf = pd.DataFrame([X[idx]], columns=EXPECTED_FEATURES)
                    expl = get_shap_explanation(sf, model)
                    top_factors_list.append([f"{f['feature']}: {f.get('suggestion','')}" for f in expl.get("top_3_risk_factors",[])[:3]])
                except Exception:
                    top_factors_list.append([])
                prog.progress(min(90, 60+int(30*(idx+1)/len(df))))
        except Exception:
            top_factors_list = [[] for _ in range(len(df))]

        prog.progress(90)
        stat.text("⏳ Saving & drift detection...")
        drift_count = 0
        for idx, row in df.iterrows():
            sid = str(row["student_id"])
            insert_student(sid, row.get("name",sid), row.get("email",""), row.get("teacher_email",""),
                          row.get("course_id",""), df.at[idx,"risk_level"], float(df.at[idx,"confidence_score"]), "Static")
            log_engagement(sid, 1, int(row.get("sum_clicks",0)), int(row.get("days_active",0)))
            facs = top_factors_list[idx] if idx<len(top_factors_list) else []
            log_prediction(sid, df.at[idx,"risk_level"], float(df.at[idx,"confidence_score"]),
                          facs[0] if len(facs)>0 else "", facs[1] if len(facs)>1 else "", facs[2] if len(facs)>2 else "")
            try:
                dr = detect_drift(sid, int(row.get("sum_clicks",0)))
                if dr["drift_detected"] and dr["z_score"]<0:
                    drift_count += 1
                    df.at[idx,"status"] = "Dynamic Risk"
            except Exception:
                pass

        prog.progress(100)
        stat.text("✅ Complete!")

        st.markdown("---")
        st.markdown(f"<div class='ep-card-title'>📊 Results</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for col,(lbl,v,clr) in zip(cols,[("Total",len(df),c["text"]),("High Risk",len(df[df["risk_level"]=="High Risk"]),c["high_risk"]),
            ("Moderate",len(df[df["risk_level"]=="Moderate Risk"]),c["moderate_risk"]),("On Track",len(df[df["risk_level"]=="On Track"]),c["on_track"]),
            ("Excelling",len(df[df["risk_level"]=="Excelling"]),c["excelling"])]):
            with col:
                bdr = f"border-left:3px solid {clr};" if lbl!="Total" else ""
                st.markdown(f"""<div class="ep-metric" style="{bdr}">
                    <div class="ep-metric-value" style="color:{clr};">{v}</div>
                    <div class="ep-metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if drift_count>0:
            st.warning(f"⚠️ **{drift_count} drift alert(s)** triggered!")
        ddf = df[["student_id","name","risk_level","confidence_score","status"]].copy()
        ddf["confidence_score"] = ddf["confidence_score"].apply(lambda x: f"{x:.1%}")
        ddf.columns = ["ID","Name","Risk Level","Confidence","Status"]
        st.dataframe(ddf, use_container_width=True, hide_index=True, height=350)

        st.markdown("---")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button("📥 Download Predictions", data=df.to_csv(index=False),
                              file_name="edupredict_predictions.csv", mime="text/csv", type="primary", use_container_width=True)
        with dc2:
            hr = df[df["risk_level"]=="High Risk"]
            if st.button(f"📧 Send Alerts ({len(hr)} High-Risk)", use_container_width=True, disabled=len(hr)==0):
                sent = sum(1 for _,r in hr.iterrows() if send_drift_alert(r.get("name",""),r.get("email",""),r.get("teacher_email",""),0.0,[])["success"])
                st.success(f"✅ {sent} alert(s) sent!") if sent else st.info("Check SMTP config.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
else:
    st.markdown(f"""<div class="ep-dropzone">
        <p style="font-size:2.5rem;margin:0;">📁</p>
        <p style="color:{c['text_secondary']};font-size:1rem;font-weight:500;">Drag & drop your CSV file here or click "Browse files"</p>
        <p style="color:{c['text_secondary']};font-size:0.8rem;opacity:0.6;">Supported: .csv | Max recommended: 10,000 rows</p>
    </div>""", unsafe_allow_html=True)
