# -*- coding: utf-8 -*-
# Inference + Feature Importance + SHAP + Active Learning shortlist
# Requires: xgboost==2.1.*, shap>=0.44, scikit-learn, pandas, numpy, matplotlib, joblib

import warnings, os, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# ========= Config =========
UNLABELED_CSV = "STEP_10_UNLABELLED_DATA.csv"         # Input CSV
MODEL_JOBLIB  = "runs_train/xgb_train_xxxx_model.joblib"  
OUTDIR        = "runs_infer"; os.makedirs(OUTDIR, exist_ok=True)
SEED = 42

# Uncertainty thresholds (Active Learning)
UNC_MAX_PROBA = 0.60
UNC_MARGIN    = 0.20

# ========= Load model =========
artifacts = joblib.load(MODEL_JOBLIB)
model = artifacts["model"]
le = artifacts["label_encoder"]
FEATURES = artifacts["features"]
classes = list(le.classes_)

# ========= Load new data =========
df_unlabeled = pd.read_csv(UNLABELED_CSV)
X_new = df_unlabeled[FEATURES].copy()

# ========= Predict =========
y_proba = model.predict_proba(X_new)
y_pred = y_proba.argmax(axis=1)
y_label = le.inverse_transform(y_pred)

df_unlabeled["PRED_LABEL"] = y_label
for i, c in enumerate(classes):
    df_unlabeled[f"proba_{c}"] = y_proba[:, i]

# ========= Active Learning signals =========
def entropy_row(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def margin_row(p):
    s = np.sort(p)[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else float(s[0])

maxp    = y_proba.max(axis=1)
margin  = np.apply_along_axis(margin_row, 1, y_proba)
entropy = np.apply_along_axis(entropy_row, 1, y_proba)

uncertain_mask = (maxp < UNC_MAX_PROBA) | (margin < UNC_MARGIN)

df_unlabeled["confidence"] = maxp
df_unlabeled["margin"] = margin
df_unlabeled["entropy"] = entropy
df_unlabeled["uncertain_flag"] = uncertain_mask.astype(int)

# Save predictions
pred_path = os.path.join(OUTDIR, "inference_predictions.csv")
df_unlabeled.to_csv(pred_path, index=False, encoding="utf-8")

# Save uncertain subset for labeling
uncertain_df = df_unlabeled[df_unlabeled["uncertain_flag"] == 1].copy()
uncertain_df = uncertain_df.sort_values(by=["margin", "confidence", "entropy"],
                                        ascending=[True, True, False])
uncertain_df.to_csv(os.path.join(OUTDIR, "uncertain_for_labeling.csv"),
                    index=False, encoding="utf-8")

print(f"[done] predictions -> {pred_path}")
print(f"[done] uncertain  -> uncertain_for_labeling.csv ({len(uncertain_df)})")

# ========= Feature Importance =========
booster = model.get_booster()
fscore = booster.get_score(importance_type="gain")
imp_vals = [fscore.get(feat, 0.0) for feat in FEATURES]

fig = plt.figure()
plt.barh(range(len(imp_vals)), imp_vals)
plt.yticks(range(len(imp_vals)), FEATURES)
plt.xlabel("Gain")
plt.title("Feature Importance (gain)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "feature_importance.png"), dpi=160)
plt.close(fig)

# ========= SHAP (subset) =========
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
bg_idx = np.random.RandomState(SEED).choice(len(X_new), size=min(512, len(X_new)), replace=False)
bg = X_new.iloc[bg_idx]

k = min(100, len(X_new))
X_slice = X_new.iloc[:k]
shap_values = explainer.shap_values(X_slice)

for c_idx, c_name in enumerate(classes):
    try:
        fig = plt.figure()
        shap.summary_plot(shap_values[c_idx], X_slice, show=False)
        plt.title(f"SHAP Summary - class: {c_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"shap_summary_class_{c_name}.png"), dpi=160)
        plt.close(fig)
    except Exception:
        pass

print("Feature importance + SHAP plots saved in:", OUTDIR)
