# EduPredict — Educational Analytics Insight Engine

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)

A machine learning-powered web application that **predicts student academic performance**, **detects engagement drift**, **explains predictions using SHAP**, and **alerts teachers via email** — all through a modern, premium Google-inspired Streamlit dashboard.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [OULAD Dataset](#-oulad-dataset)
- [Usage Guide](#-usage-guide)
- [Email Alerts Setup](#-email-alerts-setup)
- [How It Works](#-how-it-works)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Prediction** | Hybrid ensemble (Random Forest + Gradient Boosting) predicts 4 risk levels |
| 📉 **Drift Detection** | Z-score based monitoring detects sudden engagement drops |
| 🧠 **SHAP Explainability** | Explains WHY a student is at risk with actionable suggestions |
| 📧 **Email Alerts** | Automated HTML emails to teachers and students via Gmail |
| 👨‍🏫 **Teacher Dashboard** | Class overview, risk distribution, individual student analysis |
| 🎓 **Student Portal** | Personal risk level, grade trajectory, personalized advice |
| 📤 **Batch Upload** | Upload CSV → instant predictions for the entire class |
| ⚙️ **Admin Retrain** | Retrain model, compare metrics, rollback versions |

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (multipage dashboard with Google-inspired design system)
- **ML Models:** Scikit-learn (RandomForest + GradientBoosting VotingClassifier)
- **Explainability:** SHAP (TreeExplainer)
- **Data:** Pandas, NumPy
- **Database:** SQLite
- **Charts:** Plotly
- **Email:** smtplib + email.mime (Gmail SMTP_SSL)
- **Serialization:** Joblib

---

## 📁 Project Structure

```
aapt/
├── app.py                        # Main Streamlit entry point (EduPredict Landing)
├── config.py                     # All configuration constants
├── theme.py                      # Premium UI Design System CSS
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
│
├── pages/
│   ├── 1_teacher_dashboard.py    # Teacher analytics & student management
│   ├── 2_student_portal.py       # Student self-service portal
│   ├── 3_batch_upload.py         # Bulk CSV prediction & drift detection
│   └── 4_admin_retrain.py        # Model management (password-protected)
│
├── ml/
│   ├── preprocess.py             # OULAD data cleaning & feature engineering
│   ├── train.py                  # Full model training & evaluation 
│   ├── train_sample.py           # Quick testing model trainer
│   ├── drift.py                  # Behavioral drift detection (Z-score)
│   └── explain.py                # SHAP explanations & dynamic suggestions
│
├── db/
│   └── database.py               # SQLite schema & CRUD operations
│
├── alerts/
│   └── email_alert.py            # Email notification system using SMTP
│
├── data/
│   ├── sample_students.csv       # 10 fake rows for testing
│   └── download_oulad.py         # Script to download OULAD dataset
│
└── models/                       # (trained models saved here after training)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "Mini Project/aapt"
pip install -r requirements.txt
```

### 2. (Option A) Test with Sample Data

You can immediately test the app using the included sample data:

```bash
python -X utf8 ml/train_sample.py
streamlit run app.py
```

Then go to **Batch Upload** → check **"Use sample data"** → predictions will run.

### 3. (Option B) Train with OULAD Dataset

```bash
# Download the OULAD dataset (~60MB)
python data/download_oulad.py

# Preprocess and train the model
python -X utf8 ml/train.py

# Launch the dashboard
streamlit run app.py
```

---

## 📊 OULAD Dataset

The **Open University Learning Analytics Dataset** (OULAD) contains data about 32,593 students from the Open University (UK).

- **Download:** [https://analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- **License:** CC-BY 4.0
- **Citation:** Kuzilek, J., Hlosta, M. & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.

### Automatic Download

```bash
python data/download_oulad.py
```

---

## 📖 Usage Guide

### Page 1: Teacher Dashboard 👨‍🏫

1. Enter your teacher email (or click "View All Students")
2. View summary metrics and risk distribution
3. Select a student to see SHAP explanation and grade trajectory
4. Click "Send Alert Email" to notify at-risk students

### Page 2: Student Portal 🎓

1. Enter your Student ID (e.g., STU001)
2. View your risk level, confidence score, and status
3. Check your grade trajectory over time
4. Read personalized suggestions for improvement

### Page 3: Batch Upload 📤

1. Upload a CSV file matching the expected format
2. Or check "Use sample data" for testing
3. View predictions with color-coded risk levels
4. Download results as CSV
5. Send bulk email alerts for high-risk students

### Page 4: Admin Retrain ⚙️

1. Enter admin password (default: `admin123`)
2. View current model metrics
3. Click "Retrain Model" to train on OULAD data
4. Compare old vs new accuracy
5. Rollback to previous model if needed

---

## 📧 Email Alerts Setup

To enable email notifications:

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password:**
   - Go to: [https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
   - Select "Mail" → "Other" → enter "EduPredict"
   - Copy the 16-character password
3. **Update `config.py`:**
   ```python
   SMTP_SENDER = "your_actual_email@gmail.com"
   SMTP_PASSWORD = "your_16_char_app_password"
   ```

---

## 🔬 How It Works

### ML Pipeline

1. **Preprocessing:** OULAD CSVs → merged master table → feature engineering (14 features) → StandardScaler + LabelEncoder
2. **Training:** RandomForest(200 trees) + GradientBoosting(100 rounds) → VotingClassifier(soft)
3. **Prediction:** 4 risk classes — Excelling, On Track, Moderate Risk, High Risk
4. **Explainability:** SHAP TreeExplainer → top 3 risk factors → human-readable suggestions

### Drift Detection

- Uses **Z-score analysis** comparing current engagement to student's own baseline
- Baseline = mean + std of first 20% of recorded weeks
- `|Z| > 2.0` → drift detected → status changes to "Dynamic Risk"
- Triggers email alerts to teacher and student

### Risk Classes

| Class | OULAD Label | Color | Meaning |
|-------|-------------|-------|---------|
| Excelling | Distinction | 🔵 Blue | Top performer |
| On Track | Pass | 🟢 Green | Satisfactory progress |
| Moderate Risk | Withdrawn | 🟡 Amber | Withdrew from course |
| High Risk | Fail | 🔴 Red | Failed the course |

---

*Built as a 3rd Year CSE Mini Project — EduPredict System*
