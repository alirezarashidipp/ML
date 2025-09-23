# -*- coding: utf-8 -*-
# Inference + Active Learning split + SHAP for confident tickets
# Requires: xgboost==2.1.*, shap>=0.44, scikit-learn, pandas, numpy, matplotlib, joblib

import warnings, os, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap

# ========= Config =========
UNLABELED_CSV = "STEP_10_UNLABELLED_DATA.csv"
MODEL_JOBLIB  = "runs_train/xgb_train_xxxx_model.joblib"  
OUTDIR        = "runs_infer"; os.makedirs(OUTDIR, exist_ok=True)
SEED = 42

# Uncertainty thresholds
UNC_MAX_PROBA = 0.60
UNC_MARGIN    = 0.20

# ========= Load model =========
artifacts = joblib.load(MODEL_JOBLIB)
model = artifacts["model"]
le = artifacts["label_encoder"]
FEATURES = artifacts["features"]
classes = list(le.classes_)

# ========= Load data =========
df = pd.read_csv(UNLABELED_CSV)
X = df[FEATURES].copy()

# ========= Predict =========
proba = model.predict_proba(X)
pred_idx = proba.argmax(axis=1)
pred_label = le.inverse_transform(pred_idx)

df["PRED_LABEL"] = pred_label
for i, c in enumerate(classes):
    df[f"proba_{c}"] = proba[:, i]

# ========= Uncertainty metrics =========
def entropy_row(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def margin_row(p):
    s = np.sort(p)[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else float(s[0])

maxp    = proba.max(axis=1)
margin  = np.apply_along_axis(margin_row, 1, proba)
entropy = np.apply_along_axis(entropy_row, 1, proba)

uncertain_mask = (maxp < UNC_MAX_PROBA) | (margin < UNC_MARGIN)

df["confidence"] = maxp
df["margin"] = margin
df["entropy"] = entropy
df["uncertain_flag"] = uncertain_mask.astype(int)

# ========= Split outputs =========
confident_df = df[df["uncertain_flag"] == 0].copy()
uncertain_df = df[df["uncertain_flag"] == 1].copy()

# --- 1) Confident samples → add SHAP explanations ---
if len(confident_df) > 0:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_vals = explainer.shap_values(confident_df[FEATURES])

    fnames = FEATURES
    shap_top_feats, shap_top_vals = [], []
    for r in range(len(confident_df)):
        c = pred_idx[confident_df.index[r]]
        vec = shap_vals[c][r]
        order = np.argsort(np.abs(vec))[::-1][:5]  # top-5 features
        shap_top_feats.append("|".join(fnames[i] for i in order))
        shap_top_vals.append("|".join(f"{vec[i]:+0.6f}" for i in order))

    confident_df["shap_top5_features"] = shap_top_feats
    confident_df["shap_top5_values"] = shap_top_vals

# ========= Save =========
ts = time.strftime("%Y%m%d_%H%M%S")
base = os.path.join(OUTDIR, f"infer_{ts}")
os.makedirs(base, exist_ok=True)

confident_path = os.path.join(base, "confident_predictions.csv")
confident_df.to_csv(confident_path, index=False, encoding="utf-8")

uncertain_path = os.path.join(base, "uncertain_for_labeling.csv")
uncertain_df.to_csv(uncertain_path, index=False, encoding="utf-8")

print(f"[done] confident -> {confident_path} ({len(confident_df)})")
print(f"[done] uncertain -> {uncertain_path} ({len(uncertain_df)})")
