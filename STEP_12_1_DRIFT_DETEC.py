# -*- coding: utf-8 -*-
# STEP_14_DRIFT_REPORT.py
# Check Data/Label/Performance Drift between baseline and new dataset
# Requires: pandas, numpy, scipy, scikit-learn, joblib, xgboost

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# ========= Config =========
BASELINE_CSV = "STEP_10_FOR_HUMAN_LABELLING_BASELINE.csv"
NEW_CSV      = "STEP_10_FOR_HUMAN_LABELLING_NEW.csv"

BASELINE_MODEL = "runs_train/baseline/model.joblib"
NEW_MODEL      = "runs_train/v1/model.joblib"

OUTDIR = "runs_drift"
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
df_base = pd.read_csv(BASELINE_CSV)
df_new  = pd.read_csv(NEW_CSV)

X_base, y_base = df_base[FEATURES].fillna(df_base[FEATURES].mean()), df_base[LABEL_COL]
X_new,  y_new  = df_new[FEATURES].fillna(df_new[FEATURES].mean()), df_new[LABEL_COL]

# ========= 1. Data Drift (KS test for numeric features) =========
data_drift = {}
for feat in FEATURES:
    try:
        stat, pval = ks_2samp(X_base[feat], X_new[feat])
        data_drift[feat] = {"ks_stat": float(stat), "p_value": float(pval)}
    except Exception as e:
        data_drift[feat] = {"error": str(e)}

# ========= 2. Label Drift (Chi-square test on distribution) =========
labels_base = y_base.value_counts().reindex([0,1,2], fill_value=0)
labels_new  = y_new.value_counts().reindex([0,1,2], fill_value=0)
cont_table = np.array([labels_base.values, labels_new.values])

chi2, pval, _, _ = chi2_contingency(cont_table)
label_drift = {
    "baseline_dist": labels_base.to_dict(),
    "new_dist": labels_new.to_dict(),
    "chi2": float(chi2),
    "p_value": float(pval)
}

# ========= 3. Performance Drift =========
# Evaluate baseline model on both baseline and new data
base_model = joblib.load(BASELINE_MODEL)
new_model  = joblib.load(NEW_MODEL)

y_pred_base_on_base = base_model.predict(X_base)
y_pred_base_on_new  = base_model.predict(X_new)
y_pred_new_on_new   = new_model.predict(X_new)

perf_drift = {
    "baseline_on_baseline": {
        "acc": accuracy_score(y_base, y_pred_base_on_base),
        "f1_macro": f1_score(y_base, y_pred_base_on_base, average="macro")
    },
    "baseline_on_new": {
        "acc": accuracy_score(y_new, y_pred_base_on_new),
        "f1_macro": f1_score(y_new, y_pred_base_on_new, average="macro")
    },
    "new_on_new": {
        "acc": accuracy_score(y_new, y_pred_new_on_new),
        "f1_macro": f1_score(y_new, y_pred_new_on_new, average="macro")
    }
}

# ========= Save JSON =========
results = {
    "data_drift": data_drift,
    "label_drift": label_drift,
    "performance_drift": perf_drift
}
with open(os.path.join(OUTDIR, "drift_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# ========= Save TXT (management summary) =========
with open(os.path.join(OUTDIR, "drift_report.txt"), "w") as f:
    f.write("=== Drift Report ===\n\n")

    # Label drift
    f.write("Label Distribution (Baseline vs New):\n")
    f.write(str(label_drift["baseline_dist"]) + " vs " + str(label_drift["new_dist"]) + "\n")
    f.write(f"Chi2={label_drift['chi2']:.3f}, p={label_drift['p_value']:.3f}\n\n")

    # Performance drift
    f.write("Performance Comparison:\n")
    for k,v in perf_drift.items():
        f.write(f"{k}: acc={v['acc']:.3f}, f1_macro={v['f1_macro']:.3f}\n")
    f.write("\n")

    # Data drift summary (top 5 features with biggest KS stat)
    drift_sorted = sorted(data_drift.items(), key=lambda x: x[1].get("ks_stat",0), reverse=True)
    f.write("Top 5 drifting features:\n")
    for feat, vals in drift_sorted[:5]:
        f.write(f"{feat}: KS={vals.get('ks_stat',0):.3f}, p={vals.get('p_value',1):.3f}\n")

print(f"Drift report saved to {OUTDIR}/drift_report.txt and drift_metrics.json")
