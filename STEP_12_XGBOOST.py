# -*- coding: utf-8 -*-
# Ordinal XGBoost training script (advanced, saves all outputs)
# Requires: xgboost>=2.1.*, scikit-learn, pandas, numpy, joblib
# CV + Hyperparameter tuning
# Class imbalance handling
# Early stopping
# ALL METRICS
# Feature importance

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, confusion_matrix, classification_report)
from sklearn.metrics import cohen_kappa_score
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
X = df[FEATURES].copy()
y = df[LABEL_COL].copy()
X = X.fillna(X.mean())

# ========= Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= Build cumulative binary targets =========
y_train_task1 = (y_train <= 0).astype(int)
y_test_task1  = (y_test  <= 0).astype(int)
y_train_task2 = (y_train <= 1).astype(int)
y_test_task2  = (y_test  <= 1).astype(int)

# ========= Helper =========
def calc_spw(y):
    n_pos = np.sum(y==1)
    n_neg = np.sum(y==0)
    return n_neg / n_pos if n_pos > 0 else 1.0

base_params = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6
)

param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300]
}

def train_task(X_train, y_train, spw):
    model = XGBClassifier(**base_params, scale_pos_weight=spw)
    grid = GridSearchCV(model, param_grid, cv=3, scoring="f1")
    grid.fit(X_train, y_train, verbose=False)
    return grid.best_estimator_

# ========= Train models =========
spw1 = calc_spw(y_train_task1)
spw2 = calc_spw(y_train_task2)

model1 = train_task(X_train, y_train_task1, spw1)
model2 = train_task(X_train, y_train_task2, spw2)

# ========= Predict cumulative =========
p1 = model1.predict_proba(X_test)[:,1]
p2 = model2.predict_proba(X_test)[:,1]

y_pred = []
for prob1, prob2 in zip(p1, p2):
    if prob1 >= 0.5:
        y_pred.append(0)
    elif prob2 >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(2)
y_pred = np.array(y_pred)

# ========= Metrics =========
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
mcc = matthews_corrcoef(y_test, y_pred)
qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

cls_report = classification_report(y_test, y_pred, digits=3)
conf_mat = confusion_matrix(y_test, y_pred)

imp1 = pd.Series(model1.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp2 = pd.Series(model2.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ========= Prepare outputs =========
os.makedirs("runs_train", exist_ok=True)

# JSON (structured metrics)
metrics_dict = {
    "accuracy": acc,
    "precision_macro": prec,
    "recall_macro": rec,
    "f1_macro": f1,
    "precision_weighted": prec_w,
    "recall_weighted": rec_w,
    "f1_weighted": f1_w,
    "mcc": mcc,
    "quadratic_weighted_kappa": qwk,
    "confusion_matrix": conf_mat.tolist(),
    "top_features_task1": imp1.head(10).to_dict(),
    "top_features_task2": imp2.head(10).to_dict()
}
with open("runs_train/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

# TXT (human-readable full report)
with open("runs_train/eval_report.txt", "w") as f:
    f.write("=== Metrics on Test Set ===\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"Precision (macro): {prec:.3f}, Recall (macro): {rec:.3f}, F1 (macro): {f1:.3f}\n")
    f.write(f"Precision (weighted): {prec_w:.3f}, Recall (weighted): {rec_w:.3f}, F1 (weighted): {f1_w:.3f}\n")
    f.write(f"MCC: {mcc:.3f}\n")
    f.write(f"Quadratic Weighted Kappa: {qwk:.3f}\n\n")

    f.write("=== Classification Report ===\n")
    f.write(cls_report + "\n\n")

    f.write("=== Confusion Matrix ===\n")
    f.write(str(conf_mat) + "\n\n")

    f.write("=== Top Features (Task1 - Poor vs others) ===\n")
    f.write(str(imp1.head(10)) + "\n\n")

    f.write("=== Top Features (Task2 - Poor+Acceptable vs Good) ===\n")
    f.write(str(imp2.head(10)) + "\n\n")

# ========= Save models =========
joblib.dump((model1, model2), "runs_train/xgb_ordinal_advanced.joblib")
print("Models and reports saved to runs_train/")
