"""
train.py — Phase 2: Hybrid Model Training for AAPT
====================================================
This module trains TWO machine learning models and combines them:

1. Random Forest Classifier
   - An "ensemble" of many decision trees (200 trees)
   - Each tree sees a random subset of data and features
   - Final prediction = majority vote across all trees
   - WHY: Very robust, handles noisy data well, rarely overfits
   - Uses entropy criterion (measures information gain) for splits

2. Gradient Boosting Classifier
   - Builds trees SEQUENTIALLY — each new tree tries to fix
     the mistakes of the previous ones
   - WHY: Often achieves higher accuracy than Random Forest alone
   - learning_rate=0.05 means each tree contributes cautiously
   - max_depth=4 prevents individual trees from being too complex

3. Voting Classifier (Ensemble)
   - Combines both models using 'soft voting'
   - Soft voting = average the probability estimates from both models
   - WHY WE COMBINE: Different models capture different patterns.
     Combining them gives more reliable predictions than either alone.
     Think of it like asking two experts and averaging their opinions.
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
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (MODEL_PATH, FEATURE_NAMES_PATH, MODEL_METADATA_PATH,
                    MODEL_BACKUP_PATH, RF_N_ESTIMATORS, RF_CRITERION,
                    RF_RANDOM_STATE, GBT_N_ESTIMATORS, GBT_LEARNING_RATE,
                    GBT_MAX_DEPTH, GBT_RANDOM_STATE, TEST_SPLIT_SIZE,
                    RISK_CLASSES)
from ml.preprocess import preprocess_full_pipeline


def train_model(X=None, y=None, feature_names=None, label_encoder=None):
    """
    Train the hybrid ensemble model on preprocessed data.
    
    Parameters:
      X: feature matrix (numpy array or DataFrame)
      y: target labels (encoded as integers)
      feature_names: list of feature column names
      label_encoder: fitted LabelEncoder to decode predictions
    
    If X and y are None, runs the full preprocessing pipeline first.
    
    Returns:
      dict with model, metrics, and file paths
    """
    # ── Step 1: Get data ────────────────────────────────────
    if X is None or y is None:
        print("=" * 60)
        print("AAPT — Phase 2: Model Training")
        print("=" * 60)
        result = preprocess_full_pipeline()
        if result[0] is None:
            print("❌ Cannot train: preprocessing failed.")
            return None
        X, y, feature_names, label_encoder = result
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    
    # ── Step 2: Train-test split ────────────────────────────
    # stratify=y ensures each risk category appears in both train
    # and test sets proportionally. Without this, rare categories
    # (like "Excelling") might be missing from the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT_SIZE,
        random_state=42,
        stratify=y
    )
    print(f"\n📊 Data split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    # ── Step 3: Create individual models ────────────────────
    
    # Random Forest: 200 trees, each built independently
    # class_weight='balanced' automatically adjusts weights to handle
    # class imbalance (gives more importance to underrepresented classes)
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,   # 200 trees
        criterion=RF_CRITERION,          # "entropy" — information gain
        random_state=RF_RANDOM_STATE,    # reproducibility
        class_weight="balanced",         # handle class imbalance
        n_jobs=-1                        # use all CPU cores
    )
    
    # Gradient Boosting: 100 rounds of sequential tree building
    # Unlike RF, each tree here focuses on correcting previous errors
    gbt_model = GradientBoostingClassifier(
        n_estimators=GBT_N_ESTIMATORS,     # 100 boosting rounds
        learning_rate=GBT_LEARNING_RATE,   # 0.05 — small steps
        max_depth=GBT_MAX_DEPTH,           # limit tree complexity
        random_state=GBT_RANDOM_STATE
    )
    
    # ── Step 4: Create ensemble (Voting Classifier) ─────────
    # VotingClassifier combines both models
    # voting='soft' means it averages probability estimates:
    #   p_final = (p_rf + p_gbt) / 2
    # This is better than 'hard' voting (simple majority) because
    # it considers HOW CONFIDENT each model is, not just its guess.
    ensemble = VotingClassifier(
        estimators=[
            ("random_forest", rf_model),
            ("gradient_boosting", gbt_model)
        ],
        voting="soft"
    )
    
    # ── Step 5: Train the ensemble ──────────────────────────
    print("\n🏋️ Training models...")
    print("  Training Random Forest (200 trees)...")
    print("  Training Gradient Boosting (100 rounds)...")
    ensemble.fit(X_train, y_train)
    print("  ✓ Training complete!")
    
    # ── Step 6: Evaluate ────────────────────────────────────
    y_pred = ensemble.predict(X_test)
    
    # Accuracy: % of correct predictions (simple but can be misleading
    # with imbalanced classes)
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1-score (macro): harmonic mean of precision and recall,
    # averaged equally across all classes. "macro" means each class
    # counts equally regardless of how many samples it has.
    f1 = f1_score(y_test, y_pred, average="macro")
    
    # Decode labels back to readable names for the report
    target_names = label_encoder.classes_
    
    print(f"\n{'='*60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:       {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1-score (macro): {f1:.4f}")
    
    # Classification report: precision, recall, f1 per class
    # Precision = of all students predicted "High Risk", how many actually were?
    # Recall = of all actual "High Risk" students, how many did we catch?
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Confusion matrix: rows = actual, columns = predicted
    # Diagonal values = correct predictions
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {cm}")
    
    # ── Step 7: Evaluate individual models ──────────────────
    # Let's also see how each model does alone
    for name, model in ensemble.named_estimators_.items():
        y_pred_single = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_single)
        f1_single = f1_score(y_test, y_pred_single, average="macro")
        print(f"\n  {name}: Accuracy={acc:.4f}, F1={f1_single:.4f}")
    
    # ── Step 8: Save model and metadata ─────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # If an existing model exists, back it up before overwriting
    if os.path.exists(MODEL_PATH):
        print(f"\n  Backing up existing model...")
        joblib.dump(joblib.load(MODEL_PATH), MODEL_BACKUP_PATH)
    
    # Save the ensemble model
    joblib.dump(ensemble, MODEL_PATH)
    print(f"  ✓ Model saved to: {MODEL_PATH}")
    
    # Save feature names (needed for SHAP and batch prediction)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"  ✓ Feature names saved to: {FEATURE_NAMES_PATH}")
    
    # Save metadata (displayed on admin retrain page)
    metadata = {
        "training_date": datetime.now().isoformat(),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "feature_names": list(feature_names) if feature_names else [],
        "class_names": list(target_names),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to: {MODEL_METADATA_PATH}")
    
    print(f"\n🎉 Model training complete!")
    print(f"   Next step: streamlit run app.py")
    
    return {
        "model": ensemble,
        "accuracy": accuracy,
        "f1_score": f1,
        "metadata": metadata
    }


# ──────────────────────────────────────────────────────────
# Run as script: python ml/train.py
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
