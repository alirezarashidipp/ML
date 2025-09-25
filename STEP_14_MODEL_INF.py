# -*- coding: utf-8 -*-
# STEP_13_INFERENCE_ORDINAL.py
# Inference with Ordinal XGBoost models + SHAP explanations
# Requires: xgboost>=2.1.*, shap>=0.44, pandas, numpy, joblib

import warnings, os, shap, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ========= Config =========
UNLABELLED_CSV = "STEP_11_UNLABELLED.csv"  # input: must have Key + FEATURES
MODEL_PATH     = "runs_train/xgb_ordinal_advanced.joblib"
OUTDIR         = "runs_inference"

FEATURES = [
    "avg_sentence_length_tokens","sentence_length_cv","clause_density_finite_per_sentence",
    "subordination_ratio","coordination_index_cconj_per_clause","avg_dependency_distance",
    "np_mean_length_tokens","nominalization_density","noun_verb_ratio",
    "passive_voice_percent_finite_verbs","mean_zipf_lemma","lexical_diversity_mattr",
    "technical_jargon_density","negation_density_per_sentence","idea_density_predicates_per_10w",
    "FRE_norm","FKGL_norm","Fog_norm","LW_norm","ARI_norm","SMOG_norm","CLI_norm",
    "DC_norm","LIX_norm","SPACHE_norm"
]

LABEL_MAP = {0: "Poor", 1: "Acceptable", 2: "Good"}
CONF_THRESHOLD = 0.8  # active learning cutoff

# ========= Load =========
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(UNLABELLED_CSV)
X = df[FEATURES].copy().fillna(df[FEATURES].mean())

model1, model2 = joblib.load(MODEL_PATH)

# ========= Predict cumulative =========
p1 = model1.predict_proba(X)[:,1]  # P(y <= 0)
p2 = model2.predict_proba(X)[:,1]  # P(y <= 1)

# Build probabilities for 3 classes
P0 = p1
P1 = np.maximum(0, p2 - p1)
P2 = 1 - p2
probs = np.vstack([P0, P1, P2]).T

y_pred = np.argmax(probs, axis=1)
final_labels = [LABEL_MAP[i] for i in y_pred]
confidences = probs.max(axis=1)

# ========= SHAP explanations =========
expl1 = shap.TreeExplainer(model1)
expl2 = shap.TreeExplainer(model2)

shap_values1 = expl1.shap_values(X)
shap_values2 = expl2.shap_values(X)

top_features_list = []
for i in range(len(X)):
    cls = y_pred[i]
    if cls == 0:   # Poor → model1 relevant
        shap_vals = shap_values1[i]
    elif cls == 1: # Acceptable → model2 relevant
        shap_vals = shap_values2[i]
    else:          # Good → model2 relevant
        shap_vals = shap_values2[i]

    feat_imp = pd.Series(shap_vals, index=FEATURES).abs().sort_values(ascending=False).head(5)
    top_feats = "; ".join([f"{f}: {shap_vals[list(FEATURES).index(f)]:+.3f}" for f in feat_imp.index])
    top_features_list.append(top_feats)

# ========= Save predictions =========
pred_df = pd.DataFrame({
    "Key": df["Key"],
    "Prediction": y_pred,
    "Final_Label": final_labels,
    "P_class0": P0,
    "P_class1": P1,
    "P_class2": P2,
    "Confidence": confidences,
    "Top5_SHAP_Features": top_features_list
})

pred_path = os.path.join(OUTDIR, "predictions.csv")
pred_df.to_csv(pred_path, index=False)

# ========= Save low-confidence samples for Active Learning =========
lowconf_df = pred_df[pred_df["Confidence"] < CONF_THRESHOLD]
lowconf_path = os.path.join(OUTDIR, "predictions_lowconf.csv")
lowconf_df.to_csv(lowconf_path, index=False)

# ========= Summary =========
summary_path = os.path.join(OUTDIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Inference Summary ===\n")
    f.write(f"Total records: {len(df)}\n\n")

    dist = pred_df["Final_Label"].value_counts(normalize=True) * 100
    f.write("Prediction distribution (%):\n")
    for lbl, pct in dist.items():
        f.write(f"{lbl}: {pct:.1f}%\n")
    f.write("\n")

    f.write(f"Average confidence: {confidences.mean():.3f}\n\n")

    f.write(f"Low-confidence threshold: {CONF_THRESHOLD:.2f}\n")
    f.write(f"Total low-confidence samples: {len(lowconf_df)}\n\n")

    low_conf = pred_df.nsmallest(10, "Confidence")
    f.write("10 samples with lowest confidence:\n")
    f.write(low_conf[["Key","Final_Label","Confidence"]].to_string(index=False))
    f.write("\n")

print(f"Inference completed. Results saved to:\n- {pred_path}\n- {lowconf_path}\n- {summary_path}")
