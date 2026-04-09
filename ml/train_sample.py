"""
train_sample.py — Train a model using sample/synthetic data
============================================================
Use this script to train a working model WITHOUT downloading the
full OULAD dataset. It generates synthetic training data that mimics
the OULAD distribution, trains the ensemble model, and saves it.

This is useful for:
  1. Testing the app immediately after setup
  2. Demonstrating the full pipeline in a presentation
  3. Development and debugging

Usage:  python ml/train_sample.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (MODEL_PATH, FEATURE_NAMES_PATH, MODEL_METADATA_PATH,
                    SCALER_PATH, ENCODERS_PATH, EXPECTED_FEATURES,
                    RF_N_ESTIMATORS, RF_CRITERION, RF_RANDOM_STATE,
                    GBT_N_ESTIMATORS, GBT_LEARNING_RATE, GBT_MAX_DEPTH,
                    GBT_RANDOM_STATE, TEST_SPLIT_SIZE, RISK_CLASSES)


def generate_synthetic_data(n_samples=2000, random_state=42):
    """
    Generate synthetic student data that mimics OULAD patterns.
    
    We create 4 clusters corresponding to our risk classes:
      - Excelling: high clicks, high scores, positive trends
      - On Track: moderate clicks, decent scores
      - Moderate Risk: low-moderate clicks, lower scores
      - High Risk: very low clicks, poor scores, negative trends
    
    This gives the model realistic patterns to learn from.
    """
    np.random.seed(random_state)
    
    # How many samples per class (slightly imbalanced like real data)
    class_sizes = {
        "Excelling": int(n_samples * 0.15),       # 15% — rarest
        "On Track": int(n_samples * 0.35),         # 35% — most common
        "Moderate Risk": int(n_samples * 0.25),    # 25%
        "High Risk": int(n_samples * 0.25),        # 25%
    }
    
    all_data = []
    
    for risk_class, size in class_sizes.items():
        if risk_class == "Excelling":
            # High engagement, high scores
            data = {
                "gender": np.random.choice([0, 1], size),
                "age_band": np.random.choice([0, 1, 2], size, p=[0.6, 0.3, 0.1]),
                "region": np.random.randint(0, 13, size),
                "highest_education": np.random.choice([2, 3, 4], size, p=[0.3, 0.4, 0.3]),
                "imd_band": np.random.randint(4, 10, size),
                "disability": np.random.choice([0, 1], size, p=[0.9, 0.1]),
                "num_of_prev_attempts": np.random.choice([0, 0, 0, 1], size),
                "studied_credits": np.random.choice([60, 90, 120], size),
                "weighted_avg_score": np.random.normal(82, 8, size).clip(60, 100),
                "sum_clicks": np.random.normal(4000, 800, size).clip(2000, 7000),
                "avg_clicks_per_day": np.random.normal(20, 4, size).clip(10, 35),
                "days_active": np.random.normal(200, 15, size).clip(150, 260),
                "last_active_day": np.random.normal(250, 10, size).clip(220, 270),
                "click_trend": np.random.normal(2.5, 1.0, size).clip(0, 5),
            }
        elif risk_class == "On Track":
            # Moderate engagement, decent scores
            data = {
                "gender": np.random.choice([0, 1], size),
                "age_band": np.random.choice([0, 1, 2], size, p=[0.5, 0.35, 0.15]),
                "region": np.random.randint(0, 13, size),
                "highest_education": np.random.choice([1, 2, 3], size, p=[0.3, 0.4, 0.3]),
                "imd_band": np.random.randint(3, 8, size),
                "disability": np.random.choice([0, 1], size, p=[0.85, 0.15]),
                "num_of_prev_attempts": np.random.choice([0, 0, 1, 1], size),
                "studied_credits": np.random.choice([60, 90, 120], size),
                "weighted_avg_score": np.random.normal(62, 10, size).clip(40, 85),
                "sum_clicks": np.random.normal(2000, 600, size).clip(800, 4000),
                "avg_clicks_per_day": np.random.normal(10, 3, size).clip(4, 20),
                "days_active": np.random.normal(170, 20, size).clip(120, 220),
                "last_active_day": np.random.normal(230, 15, size).clip(180, 260),
                "click_trend": np.random.normal(0.5, 0.8, size).clip(-1, 3),
            }
        elif risk_class == "Moderate Risk":
            # Low engagement, concerning patterns
            data = {
                "gender": np.random.choice([0, 1], size),
                "age_band": np.random.choice([0, 1, 2], size, p=[0.4, 0.35, 0.25]),
                "region": np.random.randint(0, 13, size),
                "highest_education": np.random.choice([0, 1, 2], size, p=[0.3, 0.4, 0.3]),
                "imd_band": np.random.randint(1, 6, size),
                "disability": np.random.choice([0, 1], size, p=[0.8, 0.2]),
                "num_of_prev_attempts": np.random.choice([0, 1, 1, 2], size),
                "studied_credits": np.random.choice([60, 90, 120, 150], size),
                "weighted_avg_score": np.random.normal(40, 12, size).clip(10, 65),
                "sum_clicks": np.random.normal(800, 400, size).clip(100, 2000),
                "avg_clicks_per_day": np.random.normal(5, 2, size).clip(1, 12),
                "days_active": np.random.normal(120, 30, size).clip(50, 180),
                "last_active_day": np.random.normal(160, 40, size).clip(60, 240),
                "click_trend": np.random.normal(-1.0, 1.0, size).clip(-4, 1),
            }
        else:  # High Risk
            # Very low engagement, poor performance
            data = {
                "gender": np.random.choice([0, 1], size),
                "age_band": np.random.choice([0, 1, 2], size, p=[0.4, 0.3, 0.3]),
                "region": np.random.randint(0, 13, size),
                "highest_education": np.random.choice([0, 1, 2], size, p=[0.5, 0.35, 0.15]),
                "imd_band": np.random.randint(0, 5, size),
                "disability": np.random.choice([0, 1], size, p=[0.75, 0.25]),
                "num_of_prev_attempts": np.random.choice([0, 1, 2, 3], size),
                "studied_credits": np.random.choice([60, 120, 150], size),
                "weighted_avg_score": np.random.normal(25, 10, size).clip(0, 45),
                "sum_clicks": np.random.normal(300, 200, size).clip(10, 800),
                "avg_clicks_per_day": np.random.normal(2, 1.5, size).clip(0.1, 6),
                "days_active": np.random.normal(70, 30, size).clip(10, 140),
                "last_active_day": np.random.normal(100, 40, size).clip(10, 180),
                "click_trend": np.random.normal(-2.5, 1.2, size).clip(-5, 0),
            }
        
        df = pd.DataFrame(data)
        df["final_result"] = risk_class
        all_data.append(df)
    
    master = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the data
    master = master.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"✅ Generated {len(master)} synthetic samples")
    print(f"\n📊 Class distribution:")
    for cls, count in master["final_result"].value_counts().items():
        print(f"  {cls}: {count} ({count/len(master)*100:.1f}%)")
    
    return master


def train_with_sample_data():
    """
    Full training pipeline using synthetic data.
    Produces a working model that can be used immediately.
    """
    print("=" * 60)
    print("AAPT — Training with Synthetic Sample Data")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n📐 Generating synthetic training data...")
    master = generate_synthetic_data(n_samples=2000)
    
    # Step 2: Prepare features and target
    feature_cols = EXPECTED_FEATURES
    X = master[feature_cols].copy()
    y_raw = master["final_result"].copy()
    
    # Step 3: Scale numerical features
    numerical_cols = ["num_of_prev_attempts", "studied_credits",
                      "weighted_avg_score", "sum_clicks",
                      "avg_clicks_per_day", "days_active",
                      "last_active_day", "click_trend"]
    
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Step 4: Encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(RISK_CLASSES)
    y = label_encoder.transform(y_raw)
    
    # Step 5: Save scaler and encoders
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"label_encoder": label_encoder, "numerical_cols": numerical_cols},
                ENCODERS_PATH)
    print(f"\n✅ Scaler saved to: {SCALER_PATH}")
    print(f"✅ Encoders saved to: {ENCODERS_PATH}")
    
    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=y
    )
    print(f"\n📊 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Step 7: Build ensemble model
    print("\n🏋️ Training ensemble model...")
    
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, criterion=RF_CRITERION,
        random_state=RF_RANDOM_STATE, class_weight="balanced", n_jobs=-1
    )
    gbt = GradientBoostingClassifier(
        n_estimators=GBT_N_ESTIMATORS, learning_rate=GBT_LEARNING_RATE,
        max_depth=GBT_MAX_DEPTH, random_state=GBT_RANDOM_STATE
    )
    ensemble = VotingClassifier(
        estimators=[("random_forest", rf), ("gradient_boosting", gbt)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    # Step 8: Evaluate
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    target_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:         {accuracy*100:.1f}%")
    print(f"  F1-score (macro): {f1*100:.1f}%")
    print(f"\n{report}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Step 9: Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(list(feature_cols), FEATURE_NAMES_PATH)
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "num_samples": 2000,
        "num_features": len(feature_cols),
        "feature_names": list(feature_cols),
        "class_names": list(target_names),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "data_source": "synthetic_sample"
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Metadata saved to: {MODEL_METADATA_PATH}")
    print(f"\n🎉 Training complete! You can now run: streamlit run app.py")
    
    return metadata


if __name__ == "__main__":
    train_with_sample_data()
