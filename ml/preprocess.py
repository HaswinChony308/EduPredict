"""
preprocess.py — Phase 1: Data Preprocessing for AAPT
=====================================================
This module handles:
  1. Loading OULAD (Open University Learning Analytics Dataset) CSVs
  2. Merging them into a single master dataframe per student
  3. Engineering features (demographics, academic, engagement)
  4. Encoding categorical variables and scaling numerical ones
  5. Saving the fitted scaler/encoders for reuse during prediction

OULAD Dataset Files Used:
  - studentInfo.csv     → demographics + final_result
  - studentAssessment.csv → assessment scores
  - assessments.csv     → assessment weights
  - studentVle.csv      → VLE (Virtual Learning Environment) clicks

Why we merge all tables:
  Each CSV captures a different dimension of student behavior.
  ML models work best when all information is in one flat table
  with one row per student and many columns (features).
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, SCALER_PATH, ENCODERS_PATH, EXPECTED_FEATURES, BASE_DIR


def load_oulad_data():
    """
    Load the individual OULAD CSV files.
    
    We use chunked reading where possible to keep memory usage
    under 4GB. For the smaller files (studentInfo, assessments),
    we load them fully. For larger files (studentVle which can
    be ~1GB), we process in chunks.
    
    Returns: dict of DataFrames keyed by table name
    """
    print("📂 Loading OULAD dataset files...")
    
    data = {}
    
    # ── studentInfo.csv ─────────────────────────────────────
    # Contains: student demographics + final_result (our target)
    # Columns: code_module, code_presentation, id_student, gender,
    #          region, highest_education, imd_band, age_band,
    #          num_of_prev_attempts, studied_credits, disability, final_result
    info_path = os.path.join(DATA_DIR, "studentInfo.csv")
    if os.path.exists(info_path):
        data["studentInfo"] = pd.read_csv(info_path)
        print(f"  ✓ studentInfo: {len(data['studentInfo'])} rows")
    else:
        print(f"  ✗ studentInfo.csv not found at {info_path}")
        return None
    
    # ── assessments.csv ─────────────────────────────────────
    # Contains: assessment metadata including weight (importance)
    # We need the weight column to compute weighted average scores
    assess_path = os.path.join(DATA_DIR, "assessments.csv")
    if os.path.exists(assess_path):
        data["assessments"] = pd.read_csv(assess_path)
        print(f"  ✓ assessments: {len(data['assessments'])} rows")
    
    # ── studentAssessment.csv ───────────────────────────────
    # Contains: individual student scores on each assessment
    # We merge with assessments.csv to get the weight
    sa_path = os.path.join(DATA_DIR, "studentAssessment.csv")
    if os.path.exists(sa_path):
        data["studentAssessment"] = pd.read_csv(sa_path)
        print(f"  ✓ studentAssessment: {len(data['studentAssessment'])} rows")
    
    # ── studentVle.csv ──────────────────────────────────────
    # Contains: VLE click data (can be very large ~10M rows)
    # We aggregate this per student to avoid memory issues
    vle_path = os.path.join(DATA_DIR, "studentVle.csv")
    if os.path.exists(vle_path):
        # Process in chunks of 100,000 rows to conserve memory
        print("  ⏳ Processing studentVle (large file, chunked reading)...")
        vle_chunks = []
        chunk_size = 100_000
        for chunk in pd.read_csv(vle_path, chunksize=chunk_size):
            # Aggregate clicks per student per module/presentation in each chunk
            agg = chunk.groupby(
                ["code_module", "code_presentation", "id_student"]
            ).agg(
                sum_clicks=("sum_click", "sum"),
                days_active=("date", "nunique"),
                last_active_day=("date", "max")
            ).reset_index()
            vle_chunks.append(agg)
        
        # Combine all chunks and re-aggregate
        # (a student might span multiple chunks)
        vle_combined = pd.concat(vle_chunks, ignore_index=True)
        data["studentVle"] = vle_combined.groupby(
            ["code_module", "code_presentation", "id_student"]
        ).agg(
            sum_clicks=("sum_clicks", "sum"),
            days_active=("days_active", "sum"),
            last_active_day=("last_active_day", "max")
        ).reset_index()
        print(f"  ✓ studentVle aggregated: {len(data['studentVle'])} student records")
    
    return data


def compute_weighted_avg_score(data):
    """
    Calculate each student's weighted average assessment score.
    
    Why weighted average instead of simple mean?
      - Not all assessments are equally important
      - A final exam worth 50% should count more than a quiz worth 5%
      - The 'weight' column in assessments.csv tells us the importance
    
    Formula: weighted_avg = Σ(score × weight) / Σ(weight)
    """
    if "studentAssessment" not in data or "assessments" not in data:
        return pd.DataFrame()
    
    # Merge student scores with assessment weights
    sa = data["studentAssessment"].copy()
    assessments = data["assessments"][["id_assessment", "code_module",
                                       "code_presentation", "weight"]].copy()
    
    # Join on assessment ID to get the weight for each score
    merged = sa.merge(assessments, on="id_assessment", how="left")
    
    # Some scores might be missing (student didn't submit)
    # We fill those with 0 — treating no submission as a zero score
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0)
    merged["weight"] = merged["weight"].fillna(0)
    
    # Compute weighted score for each row
    merged["weighted_score"] = merged["score"] * merged["weight"]
    
    # Aggregate per student: sum of weighted scores / sum of weights
    result = merged.groupby(
        ["id_student", "code_module", "code_presentation"]
    ).agg(
        total_weighted_score=("weighted_score", "sum"),
        total_weight=("weight", "sum")
    ).reset_index()
    
    # Avoid division by zero (if a student has no assessments)
    result["weighted_avg_score"] = np.where(
        result["total_weight"] > 0,
        result["total_weighted_score"] / result["total_weight"],
        0.0
    )
    
    return result[["id_student", "code_module", "code_presentation",
                    "weighted_avg_score"]]


def compute_click_trend(data):
    """
    Calculate the trend (slope) of weekly VLE clicks for each student.
    
    Why click_trend matters:
      - A student with 100 total clicks could be:
        a) Steady at ~10 clicks/week (stable engagement) → neutral slope
        b) Started at 20, now at 2 (declining) → negative slope
        c) Started at 2, now at 20 (improving) → positive slope
      - The slope tells us the DIRECTION of engagement change
    
    We use numpy polyfit (degree=1) which fits a straight line
    y = slope*x + intercept to the weekly click data. We only
    care about the slope.
    """
    if "studentVle" not in data:
        return pd.DataFrame()
    
    # For this calculation we need the raw VLE data with dates
    vle_path = os.path.join(DATA_DIR, "studentVle.csv")
    if not os.path.exists(vle_path):
        return pd.DataFrame()
    
    print("  ⏳ Computing click trends (weekly slopes)...")
    trends = []
    chunk_size = 100_000
    
    # Accumulate weekly clicks per student across chunks
    weekly_data = {}
    
    for chunk in pd.read_csv(vle_path, chunksize=chunk_size):
        # Convert date (days from course start) to week number
        # OULAD uses days relative to course start (can be negative)
        chunk["week"] = chunk["date"] // 7
        
        # Sum clicks per student per week
        weekly = chunk.groupby(
            ["code_module", "code_presentation", "id_student", "week"]
        )["sum_click"].sum().reset_index()
        
        for _, row in weekly.iterrows():
            key = (row["code_module"], row["code_presentation"], row["id_student"])
            if key not in weekly_data:
                weekly_data[key] = {}
            week = row["week"]
            if week in weekly_data[key]:
                weekly_data[key][week] += row["sum_click"]
            else:
                weekly_data[key][week] = row["sum_click"]
    
    # Now compute slope for each student
    for (module, pres, student), week_clicks in weekly_data.items():
        weeks = sorted(week_clicks.keys())
        clicks = [week_clicks[w] for w in weeks]
        
        if len(weeks) >= 2:
            # polyfit with degree 1 returns [slope, intercept]
            slope, _ = np.polyfit(weeks, clicks, 1)
        else:
            # Not enough data points for a trend
            slope = 0.0
        
        trends.append({
            "code_module": module,
            "code_presentation": pres,
            "id_student": student,
            "click_trend": slope
        })
    
    result = pd.DataFrame(trends)
    print(f"  ✓ Click trends computed for {len(result)} students")
    return result


def encode_demographics(df):
    """
    Encode categorical demographic features into numbers.
    
    Why encode?
      - ML models (Random Forest, Gradient Boosting) work with numbers
      - "Male"/"Female" must become 0/1
      - Ordinal categories (age bands, education levels) should be
        encoded in a meaningful order so the model can learn
        "higher education → different outcome patterns"
    
    We use manual mappings (not LabelEncoder) for ordinal features
    because we want to control the order.
    """
    df = df.copy()
    
    # ── Gender ──────────────────────────────────────────────
    # Simple binary: M=0, F=1
    gender_map = {"M": 0, "F": 1}
    df["gender"] = df["gender"].map(gender_map).fillna(0).astype(int)
    
    # ── Age band ────────────────────────────────────────────
    # Ordinal encoding: younger → lower number
    age_map = {"0-35": 0, "35-55": 1, "55<=": 2}
    df["age_band"] = df["age_band"].map(age_map).fillna(0).astype(int)
    
    # ── Region ──────────────────────────────────────────────
    # Nominal (no natural order), so we just assign unique integers
    if "region" in df.columns:
        regions = df["region"].unique()
        region_map = {r: i for i, r in enumerate(regions)}
        df["region"] = df["region"].map(region_map).fillna(0).astype(int)
    
    # ── Highest education ───────────────────────────────────
    # Ordinal: lower education → lower number
    edu_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }
    df["highest_education"] = df["highest_education"].map(edu_map).fillna(1).astype(int)
    
    # ── IMD band (Index of Multiple Deprivation) ────────────
    # Socioeconomic measure: "0-10%" (most deprived) to "90-100%" (least)
    imd_map = {
        "0-10%": 0, "10-20": 1, "10-20%": 1, "20-30%": 2, "30-40%": 3,
        "40-50%": 4, "50-60%": 5, "60-70%": 6, "70-80%": 7,
        "80-90%": 8, "90-100%": 9
    }
    df["imd_band"] = df["imd_band"].map(imd_map).fillna(5).astype(int)
    
    # ── Disability ──────────────────────────────────────────
    # Binary: Y=1 (has disability), N=0 (no disability)
    disability_map = {"Y": 1, "N": 0}
    df["disability"] = df["disability"].map(disability_map).fillna(0).astype(int)
    
    return df


def build_master_dataframe(data):
    """
    Merge all OULAD tables into one master dataframe.
    
    The result has one row per student per course, with columns:
      - Demographics: gender, age_band, region, etc.
      - Academic: num_of_prev_attempts, studied_credits, weighted_avg_score
      - Engagement: sum_clicks, avg_clicks_per_day, days_active,
                    last_active_day, click_trend
      - Target: final_result
    """
    print("\n🔀 Building master dataframe...")
    
    # Start with student info (demographics + target variable)
    master = data["studentInfo"].copy()
    
    # ── Merge weighted average score ────────────────────────
    wt_scores = compute_weighted_avg_score(data)
    if not wt_scores.empty:
        master = master.merge(
            wt_scores,
            left_on=["code_module", "code_presentation", "id_student"],
            right_on=["code_module", "code_presentation", "id_student"],
            how="left"
        )
    else:
        master["weighted_avg_score"] = 0.0
    
    # ── Merge VLE engagement data ───────────────────────────
    if "studentVle" in data:
        master = master.merge(
            data["studentVle"],
            left_on=["code_module", "code_presentation", "id_student"],
            right_on=["code_module", "code_presentation", "id_student"],
            how="left"
        )
    else:
        master["sum_clicks"] = 0
        master["days_active"] = 0
        master["last_active_day"] = 0
    
    # ── Compute avg_clicks_per_day ──────────────────────────
    # Total clicks divided by days active (with zero-division protection)
    master["avg_clicks_per_day"] = np.where(
        master["days_active"] > 0,
        master["sum_clicks"] / master["days_active"],
        0.0
    )
    
    # ── Merge click trend ───────────────────────────────────
    click_trends = compute_click_trend(data)
    if not click_trends.empty:
        master = master.merge(
            click_trends,
            left_on=["code_module", "code_presentation", "id_student"],
            right_on=["code_module", "code_presentation", "id_student"],
            how="left"
        )
    else:
        master["click_trend"] = 0.0
    
    # ── Fill missing values ─────────────────────────────────
    # Students who never accessed the VLE will have NaN for click features
    fill_cols = ["weighted_avg_score", "sum_clicks", "avg_clicks_per_day",
                 "days_active", "last_active_day", "click_trend"]
    for col in fill_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)
    
    print(f"  ✓ Master dataframe: {master.shape[0]} rows, {master.shape[1]} columns")
    return master


def encode_target(df):
    """
    Map the OULAD final_result to our 4 risk classes.
    
    OULAD original labels → Our labels:
      "Distinction" → "Excelling"    (top performers)
      "Pass"        → "On Track"     (satisfactory)
      "Withdrawn"   → "Moderate Risk" (left the course — concerning)
      "Fail"        → "High Risk"    (failed — needs intervention)
    
    Why these mappings?
      - Distinction and Pass are positive outcomes
      - Withdrawn means the student left mid-course (often due to struggles)
      - Fail means they completed but didn't pass
    """
    target_map = {
        "Distinction": "Excelling",
        "Pass": "On Track",
        "Withdrawn": "Moderate Risk",
        "Fail": "High Risk"
    }
    df["final_result"] = df["final_result"].map(target_map)
    
    # Drop any rows where final_result didn't match our mapping
    df = df.dropna(subset=["final_result"])
    
    # Print class distribution so we can check for imbalance
    print("\n📊 Target class distribution:")
    dist = df["final_result"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return df


def preprocess_full_pipeline():
    """
    Run the complete preprocessing pipeline on OULAD data.
    
    Steps:
      1. Load all CSVs
      2. Merge into master dataframe
      3. Encode demographics
      4. Encode target variable
      5. Scale numerical features
      6. Save scaler and encoders for later reuse
    
    Returns: (X, y, feature_names) — ready for model training
    """
    # Step 1: Load data
    data = load_oulad_data()
    if data is None:
        print("❌ Failed to load OULAD data. Make sure CSVs are in data/ folder.")
        return None, None, None
    
    # Step 2: Build master dataframe
    master = build_master_dataframe(data)
    
    # Step 3: Encode demographics
    master = encode_demographics(master)
    
    # Step 4: Encode target
    master = encode_target(master)
    
    # Step 5: Select features and target
    feature_cols = EXPECTED_FEATURES
    
    # Ensure all expected features exist
    for col in feature_cols:
        if col not in master.columns:
            master[col] = 0
    
    X = master[feature_cols].copy()
    y = master["final_result"].copy()
    
    # Step 6: Scale numerical features
    # StandardScaler transforms each feature to have mean=0, std=1
    # This helps some models learn faster and treats all features equally
    numerical_cols = ["num_of_prev_attempts", "studied_credits",
                      "weighted_avg_score", "sum_clicks",
                      "avg_clicks_per_day", "days_active",
                      "last_active_day", "click_trend"]
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Step 7: Encode target labels to integers for sklearn
    label_encoder = LabelEncoder()
    # Fit on our known classes to ensure consistent ordering
    label_encoder.fit(["Excelling", "On Track", "Moderate Risk", "High Risk"])
    y_encoded = label_encoder.transform(y)
    
    # Step 8: Save the fitted scaler and label encoder for reuse
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"label_encoder": label_encoder, "numerical_cols": numerical_cols},
                ENCODERS_PATH)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"  Features shape: {X.shape}")
    print(f"  Scaler saved to: {SCALER_PATH}")
    print(f"  Encoders saved to: {ENCODERS_PATH}")
    
    return X, y_encoded, feature_cols, label_encoder


def preprocess_batch(df):
    """
    Preprocess a batch upload CSV for prediction.
    
    This function is called from the Batch Upload page (page 3).
    It applies the same transformations as the training pipeline
    but uses the SAVED scaler (not fitting a new one).
    
    Parameters:
      df: DataFrame with columns matching EXPECTED_FEATURES
    
    Returns:
      X_scaled: numpy array ready for model.predict()
    """
    df = df.copy()
    
    # Load the saved scaler
    if not os.path.exists(SCALER_PATH):
        print("⚠️ Scaler not found. Using raw features (model may perform poorly).")
        return df[EXPECTED_FEATURES].values
    
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    numerical_cols = encoders["numerical_cols"]
    
    # Ensure all expected features exist
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    # Apply scaling to numerical columns only
    X = df[EXPECTED_FEATURES].copy()
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X.values


# ──────────────────────────────────────────────────────────
# Run as script: python ml/preprocess.py
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AAPT — Phase 1: Data Preprocessing")
    print("=" * 60)
    result = preprocess_full_pipeline()
    if result[0] is not None:
        print("\n🎉 Data is ready for model training!")
        print("   Next step: python ml/train.py")
