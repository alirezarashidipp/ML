# -*- coding: utf-8 -*-
# Simple XGBoost training script
# Requires: xgboost>=2.1.*, scikit-learn, pandas, numpy, joblib

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ========= Config =========
CSV_PATH   = "STEP_10_FOR_HUMAN_LABELLING.csv"
LABEL_COL  = "HUMAN_LABEL"
FEATURES = [
    "avg_sentence_length_tokens","sentence_length_cv","clause_density_finite_per_sentence",
    "subordination_ratio","coordination_index_cconj_per_clause","avg_dependency_distance",
    "np_mean_length_tokens","nominalization_density","noun_verb_ratio",
    "passive_voice_percent_finite_verbs","mean_zipf_lemma","lexical_diversity_mattr",
    "technical_jargon_density","negation_density_per_sentence","idea_density_predicates_per_10w",
    "FRE_norm","FKGL_norm","Fog_norm","LW_norm","ARI_norm","SMOG_norm","CLI_norm",
    "DC_norm","LIX_norm","SPACHE_norm"
]

# ========= Load Data =========
df = pd.read_csv(CSV_PATH)

# Keep only the features + label
X = df[FEATURES].copy()
y = df[LABEL_COL].copy()

# Replace missing values with column mean (simple imputation)
X = X.fillna(X.mean())

# ========= Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= Train XGBoost =========
model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6
)

model.fit(X_train, y_train)

# ========= Evaluate =========
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ========= Save model =========
os.makedirs("runs_train", exist_ok=True)
joblib.dump(model, "runs_train/xgb_simple_model.joblib")
print("Model saved to runs_train/xgb_simple_model.joblib")
