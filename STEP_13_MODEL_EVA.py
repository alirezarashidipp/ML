# -*- coding: utf-8 -*-
# STEP_15_COMPARE_MODELS.py
# Champion–Challenger comparison: baseline vs new model
# Requires: scikit-learn, pandas, numpy, joblib, xgboost

import warnings, os, json, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, matthews_corrcoef, classification_report
)
from sklearn.metrics import cohen_kappa_score

# ========= Config =========
BASELINE_MODEL = "runs_train/BS_model.joblib"
NEW_MODEL      = "runs_train/v1_model.joblib"
TEST_CSV       = "STEP_10_FOR_HUMAN_LABELLING_TEST.csv"

OUTDIR = "runs_compare"
os.makedirs(OUTDIR, exist_ok=True)

FEATURES = [
    "avg_sentence_length_tokens","sentence_length_cv","clause_density_finite_per_sentence",
    "subordination_ratio","coordination_index_cconj_per_clause","avg_dependency_distance",
    "np_mean_length_tokens","nominalization_density","noun_verb_ratio",
    "passive_voice_percent_finite_verbs","mean_zipf_lemma","lexical_diversity_mattr",
    "technical_jargon_density","negation_density_per_sentence","idea_density_predicates_per_10w",
    "FRE_norm","FKGL_norm","Fog_norm","LW_norm","ARI_norm","SMOG_norm","CLI_norm",
    "DC_norm","LIX_norm","SPACHE_norm"
]
LABEL_COL = "HUMAN_LABEL"

# ========= Load Data =========
df = pd.read_csv(TEST_CSV)
X_test = df[FEATURES].copy().fillna(df[FEATURES].mean())
y_test = df[LABEL_COL].copy()

# ========= Load Models =========
baseline_model = joblib.load(BASELINE_MODEL)
new_model      = joblib.load(NEW_MODEL)

# ========= Predict =========
y_pred_base = baseline_model.predict(X_test)
y_pred_new  = new_model.predict(X_test)

# ========= Metrics Helper =========
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    conf = confusion_matrix(y_true, y_pred).tolist()
    cls_report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "mcc": mcc,
        "qwk": qwk,
        "confusion_matrix": conf,
        "per_class": cls_report
    }

metrics_base = evaluate(y_test, y_pred_base)
metrics_new  = evaluate(y_test, y_pred_new)

# ========= Δ (differences) =========
delta = {}
for k in ["accuracy","f1_macro","f1_weighted","mcc","qwk"]:
    delta[k] = metrics_new[k] - metrics_base[k]

# ========= Save JSON =========
results = {
    "baseline": metrics_base,
    "new_model": metrics_new,
    "delta": delta
}
with open(os.path.join(OUTDIR, "compare_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# ========= Save TXT (management summary) =========
with open(os.path.join(OUTDIR, "compare_report.txt"), "w") as f:
    f.write("=== Model Comparison Report (Champion–Challenger) ===\n\n")

    f.write("Overall Metrics:\n")
    f.write(f"Baseline: acc={metrics_base['accuracy']:.3f}, f1_macro={metrics_base['f1_macro']:.3f}, qwk={metrics_base['qwk']:.3f}\n")
    f.write(f"New:      acc={metrics_new['accuracy']:.3f}, f1_macro={metrics_new['f1_macro']:.3f}, qwk={metrics_new['qwk']:.3f}\n\n")

    f.write("Δ (New - Baseline):\n")
    for k,v in delta.items():
        f.write(f"{k}: {v:+.3f}\n")
    f.write("\n")

    f.write("Confusion Matrix (Baseline):\n")
    f.write(str(metrics_base["confusion_matrix"]) + "\n\n")
    f.write("Confusion Matrix (New):\n")
    f.write(str(metrics_new["confusion_matrix"]) + "\n\n")

    f.write("Per-Class F1 (Baseline vs New):\n")
    labels = [str(c) for c in sorted(set(y_test))]
    for lbl in labels:
        f1_b = metrics_base["per_class"][lbl]["f1-score"]
        f1_n = metrics_new["per_class"][lbl]["f1-score"]
        f.write(f"Class {lbl}: baseline={f1_b:.3f}, new={f1_n:.3f}, Δ={f1_n-f1_b:+.3f}\n")

print(f"Comparison report saved to {OUTDIR}/compare_report.txt and compare_metrics.json")
