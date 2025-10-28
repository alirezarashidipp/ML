# -*- coding: utf-8 -*-
# Ordinal XGBoost – Method A (multi:softprob + calibration + class weighting)
# Requires: xgboost>=2.1.*, scikit-learn, pandas, numpy, joblib

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
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
X = df[FEATURES].fillna(df[FEATURES].mean())
y = df[LABEL_COL].astype(int)

# ========= Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========= Class Weights (handling imbalance) =========
cls, cnt = np.unique(y_train, return_counts=True)
freq = dict(zip(cls, cnt))

alpha = 1.0   # 0.5=light, 1.0=stronger, 1.5=very strong reweighting
w_per_class = {c: (1.0 / (freq[c] ** alpha)) for c in freq}

# normalize so average weight ≈ 1
mean_w = np.mean(list(w_per_class.values()))
w_per_class = {c: w_per_class[c] / mean_w for c in w_per_class}

sample_w_train = y_train.map(w_per_class).astype(float).values

# ========= Model =========
xgb_params = dict(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

base = XGBClassifier(**xgb_params)

# ========= Calibrate (with weights) =========
cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
cal.fit(X_train, y_train, sample_weight=sample_w_train)

# ========= Predict Probabilities =========
proba = cal.predict_proba(X_test)  # shape (n,3)

# ========= Continuous Score =========
score_map = np.array([0.0, 0.5, 1.0])  # ordinal weights poor/acceptable/good
score = (proba * score_map).sum(axis=1)
score_0_100 = np.clip(score * 100, 1, 99)

# ========= Discrete prediction (for reporting only) =========
y_pred = proba.argmax(axis=1)

# ========= Metrics =========
acc  = accuracy_score(y_test, y_pred)
f1_w = f1_score(y_test, y_pred, average="weighted")
qwk  = cohen_kappa_score(y_test, y_pred, weights="quadratic")
report = classification_report(y_test, y_pred, digits=3)
cm = confusion_matrix(y_test, y_pred)

summary_text = (
    f"=== Method A Summary ===\n"
    f"Accuracy: {acc:.3f}\nF1-weighted: {f1_w:.3f}\nQuadratic Weighted Kappa: {qwk:.3f}\n"
    f"Mean continuous score: {score_0_100.mean():.2f}\n"
    f"Class weights: {w_per_class}\n"
)
print(summary_text)
print(report)

# ========= Save Outputs =========
os.makedirs("runs_train_methodA_weighted", exist_ok=True)
joblib.dump(cal, "runs_train_methodA_weighted/xgb_softprob_calibrated_weighted.joblib")

pd.DataFrame({
    "true_label": y_test,
    "pred_label": y_pred,
    "score_0_100": score_0_100
}).to_csv("runs_train_methodA_weighted/predictions.csv", index=False)

metrics_dict = {
    "accuracy": acc,
    "f1_weighted": f1_w,
    "quadratic_weighted_kappa": qwk,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "mean_score": float(score_0_100.mean()),
    "class_weights": w_per_class,
    "management_summary": summary_text
}
with open("runs_train_methodA_weighted/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

print("Model, metrics, and predictions saved to runs_train_methodA_weighted/")

# ========= Inference Helper =========
def predict_score_0_100(model, X_new):
    """Return continuous readability score 0–100 for new data."""
    proba = model.predict_proba(X_new)
    score_map = np.array([0.0, 0.5, 1.0])
    s = (proba * score_map).sum(axis=1)
    return np.clip(s * 100, 1, 99)
