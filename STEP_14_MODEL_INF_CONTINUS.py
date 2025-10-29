# -*- coding: utf-8 -*-
# STEP_13_INFERENCE_ORDINAL.py  (revised for unified multiclass softprob)
# Inference with XGBoost multiclass (softprob) + SHAP explanations
# Requires: xgboost>=2.1.*, shap>=0.44, pandas, numpy, joblib

# method 2

import warnings, os, shap, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ========= Config =========
UNLABELLED_CSV = "STEP_11_UNLABELLED.csv"
MODEL_PATH     = "runs_train/xgb_softprob_model.joblib"
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
CONF_THRESHOLD = 0.8

# ========= Load =========
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(UNLABELLED_CSV)
X = df[FEATURES].copy().fillna(df[FEATURES].mean())

model = joblib.load(MODEL_PATH)

# ========= Predict (multiclass softprob) =========
probs = model.predict_proba(X)
row_sum = probs.sum(axis=1)
if not np.all((row_sum > 0.999 - 1e-6) & (row_sum < 1.001 + 1e-6)):
    s = row_sum.reshape(-1, 1)
    mask = s > 0
    probs[mask[:,0]] = probs[mask[:,0]] / s[mask]
    probs[~mask[:,0]] = 1.0 / 3

y_pred = np.argmax(probs, axis=1)
final_labels = [LABEL_MAP[i] for i in y_pred]
confidences = probs.max(axis=1)

# ========= Continuous score (Expected Value) =========
expected_score = probs @ np.array([0, 1, 2], dtype=float)
# expected_score_100 = (expected_score / 2.0) * 100.0
expected_score_100 = 5 + (expected_score / 2.0) * (92 - 5)

# ========= SHAP explanations =========
expl = shap.TreeExplainer(model)
shap_values = expl.shap_values(X)

def get_shap_row_for_class(shap_vals, i, cls):
    if isinstance(shap_vals, (list, tuple)) and len(shap_vals) > 0:
        arr = shap_vals[cls]
        return arr[i]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        return shap_vals[i, :, cls]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        return shap_vals[i]
    return np.zeros(len(FEATURES), dtype=float)

top_features_list = []
for i in range(len(X)):
    cls = int(y_pred[i])
    sv = get_shap_row_for_class(shap_values, i, cls)
    feat_imp = pd.Series(np.abs(sv), index=FEATURES).sort_values(ascending=False).head(5)
    signed_pairs = []
    for f in feat_imp.index:
        j = list(FEATURES).index(f)
        signed_pairs.append(f"{f}: {sv[j]:+.3f}")
    top_feats = "; ".join(signed_pairs)
    top_features_list.append(top_feats)

# ========= Save predictions =========
pred_df = pd.DataFrame({
    "Key": df["Key"],
    "Prediction": y_pred,
    "Final_Label": final_labels,
    "P_class0": probs[:, 0],
    "P_class1": probs[:, 1],
    "P_class2": probs[:, 2],
    "Confidence": confidences,
    "Expected_Score_0_2": expected_score,
    "Expected_Score_0_100": expected_score_100,
    "Top5_SHAP_Features": top_features_list
})

pred_path = os.path.join(OUTDIR, "predictions.csv")
pred_df.to_csv(pred_path, index=False)

# ========= Save low-confidence samples =========
lowconf_df = pred_df[pred_df["Confidence"] < CONF_THRESHOLD]
lowconf_path = os.path.join(OUTDIR, "predictions_lowconf.csv")
lowconf_df.to_csv(lowconf_path, index=False)

# ========= Split for Active Learning (keys only) vs Main (exclude them) =========
al_keys_path     = os.path.join(OUTDIR, "active_learning_keys.csv")
main_wo_al_path  = os.path.join(OUTDIR, "predictions_main_without_AL.csv")


lowconf_df[["Key"]].drop_duplicates().to_csv(al_keys_path, index=False)


pred_main_wo_al = pred_df[~pred_df["Key"].isin(lowconf_df["Key"])].copy()
pred_main_wo_al.to_csv(main_wo_al_path, index=False)

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
    cols = ["Key","Final_Label","Confidence","Expected_Score_0_100"]
    f.write("10 samples with lowest confidence:\n")
    f.write(low_conf[cols].to_string(index=False))
    f.write("\n")

  
    f.write(f"\nActive-Learning keys saved to: {al_keys_path}\n")
    f.write(f"Main predictions excluding AL saved to: {main_wo_al_path}\n")

print(f"Inference completed. Results saved to:\n- {pred_path}\n- {lowconf_path}\n- {al_keys_path}\n- {main_wo_al_path}\n- {summary_path}")
