# -*- coding: utf-8 -*-
# Inference on Unlabeled Data + SHAP explanations
# Requires: xgboost==2.1.*, shap>=0.44, scikit-learn, pandas, numpy, matplotlib, joblib

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# ========= Config =========
UNLABELED_CSV = "UNLABELED.csv"               # Input CSV with features but no labels
MODEL_JOBLIB  = "runs_train/xgb_train_XXXX_model.joblib"  # Path to trained model artifacts
OUTDIR        = "runs_infer"; os.makedirs(OUTDIR, exist_ok=True)

TOPK_SHAP     = 5       # Number of top SHAP features to keep per sample
N_WATERFALL   = 20      # Save waterfall plots for N least confident samples
UNC_MAX_PROBA = 0.60    # Threshold on max probability for uncertainty
UNC_MARGIN    = 0.20    # Threshold on margin (p1 - p2) for uncertainty
ID_COL_FALLBACK = None  # If no "Key" column, specify another ID column here; else uses first column

# ========= Load model & assets =========
pack = joblib.load(MODEL_JOBLIB)
model    = pack["model"]
le       = pack["label_encoder"]
FEATURES = pack["features"]
classes  = list(le.classes_)
n_classes = len(classes)

# ========= Load unlabeled =========
df = pd.read_csv(UNLABELED_CSV)

# Select ID column
if "Key" in df.columns:
    id_col = "Key"
elif ID_COL_FALLBACK and ID_COL_FALLBACK in df.columns:
    id_col = ID_COL_FALLBACK
else:
    id_col = df.columns[0]

# Check for required features
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")

X_df = df[FEATURES].copy()
if X_df.isna().any().any():
    X_df = X_df.fillna(X_df.median(numeric_only=True))

# ========= Predict =========
proba = model.predict_proba(X_df.values)   # (N, C)
pred_idx = proba.argmax(axis=1)
pred_label = [classes[i] for i in pred_idx]

# ========= Uncertainty signals =========
def entropy_row(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def margin_row(p):
    s = np.sort(p)[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else float(s[0])

maxp   = proba.max(axis=1)
margin = np.apply_along_axis(margin_row, 1, proba)
entropy = np.apply_along_axis(entropy_row, 1, proba)
uncertain = (maxp < UNC_MAX_PROBA) | (margin < UNC_MARGIN)

# ========= SHAP (per-ticket Top-K) =========
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_df), size=min(512, len(X_df)), replace=False)
background = X_df.iloc[bg_idx]

explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", data=background)

def get_shap_3d(explainer, X_batch):
    """Normalize SHAP output to shape (N, D, C)."""
    vals = explainer.shap_values(X_batch)
    if isinstance(vals, list):
        C = len(vals); n, d = vals[0].shape
        out = np.zeros((n, d, C), dtype=np.float32)
        for c in range(C): out[:, :, c] = vals[c]
        return out
    vals = np.asarray(vals)
    if vals.ndim == 3:  # (N, D, C)
        return vals
    if vals.ndim == 2:  # (N, D)
        n, d = vals.shape
        return vals.reshape(n, d, 1)
    raise RuntimeError(f"Unexpected SHAP shape: {vals.shape}")

def topk_for_pred_class(shap_3d_row, x_row, pred_c, feature_names, k=TOPK_SHAP):
    """Return top-k SHAP features for predicted class of one sample."""
    vec = shap_3d_row[:, pred_c]
    order = np.argsort(np.abs(vec))[::-1][:k]
    feats = [feature_names[i] for i in order]
    svals = [float(vec[i]) for i in order]
    rvals = [float(x_row[i]) for i in order]
    abs_sum = np.abs(vec).sum() or 1.0
    parts = [f"{feats[j]}({svals[j]:+0.3f},{(abs(vec[order[j]])/abs_sum)*100:0.1f}%)" for j in range(len(order))]
    return "|".join(feats), "|".join(f"{v:+0.6f}" for v in svals), "|".join(f"{v:0.6f}" for v in rvals), "; ".join(parts)

# Expected values per class
try:
    exp_vals = explainer.expected_value
    if np.isscalar(exp_vals): exp_vals = np.array([exp_vals] * n_classes)
    elif isinstance(exp_vals, list): exp_vals = np.array(exp_vals)
except Exception:
    exp_vals = np.zeros((n_classes,), dtype=float)

def save_waterfall(idx, shap_vec_c, row_values, feature_names, base_title):
    try:
        expl = shap.Explanation(
            values=shap_vec_c,
            base_values=float(exp_vals[pred_idx[idx]]) if len(exp_vals) > pred_idx[idx] else 0.0,
            data=row_values,
            feature_names=feature_names
        )
        fig = plt.figure()
        shap.plots.waterfall(expl, show=False, max_display=15)
        plt.title(base_title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"wf_idx{idx}.png"), dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] waterfall failed for {idx}: {e}")

# Batch SHAP
B = 1024
topk_feats_txt, topk_vals_txt, topk_raws_txt, topk_human_txt = [], [], [], []
wf_indices = np.argsort(maxp)[:max(0, N_WATERFALL)].tolist() if N_WATERFALL > 0 else []

for start in range(0, len(X_df), B):
    stop = min(start + B, len(X_df))
    Xb = X_df.iloc[start:stop]
    shap_3d = get_shap_3d(explainer, Xb)  # (b, d, C)
    for i in range(len(Xb)):
        gi = start + i
        c = pred_idx[gi]
        feats_txt, vals_txt, raws_txt, human_txt = topk_for_pred_class(
            shap_3d[i], Xb.iloc[i].values, c, X_df.columns, TOPK_SHAP
        )
        topk_feats_txt.append(feats_txt)
        topk_vals_txt.append(vals_txt)
        topk_raws_txt.append(raws_txt)
        topk_human_txt.append(human_txt)
        if gi in wf_indices:
            save_waterfall(gi, shap_3d[i, :, c], Xb.iloc[i].values, X_df.columns.tolist(),
                           base_title=f"idx {gi} – pred={classes[c]} (conf={maxp[gi]:.2f})")

# ========= Build outputs =========
out = pd.DataFrame({
    id_col: df[id_col].values,
    "pred_label": pred_label,
    "pred_class_idx": pred_idx,
    "confidence": np.round(maxp, 6),
    "margin": np.round(margin, 6),
    "entropy": np.round(entropy, 6),
    "uncertain_flag": uncertain.astype(int),
})

# Per-class probabilities
for j, cls in enumerate(classes):
    out[f"proba_{cls}"] = np.round(proba[:, j], 6)

# SHAP top-k outputs
out[f"shap_top{TOPK_SHAP}_features"] = topk_feats_txt
out[f"shap_top{TOPK_SHAP}_values"]   = topk_vals_txt
out[f"shap_top{TOPK_SHAP}_raw_x"]    = topk_raws_txt
out[f"shap_top{TOPK_SHAP}_human"]    = topk_human_txt

pred_path = os.path.join(OUTDIR, "predictions_with_shap.csv")
out.to_csv(pred_path, index=False, encoding="utf-8")

# Save uncertain subset
unc = out[out["uncertain_flag"] == 1].copy().sort_values(
    by=["margin", "confidence", "entropy"], ascending=[True, True, False]
)
unc_path = os.path.join(OUTDIR, "uncertain_for_labeling_unlabeled.csv")
unc.to_csv(unc_path, index=False, encoding="utf-8")

# Save meta
meta = {
    "model_joblib": MODEL_JOBLIB,
    "num_samples": int(len(df)),
    "classes": classes,
    "topk_shap": TOPK_SHAP,
    "waterfalls_saved": int(len([f for f in os.listdir(OUTDIR) if f.startswith('wf_idx') and f.endswith('.png')])),
    "uncertainty_thresholds": {"max_proba": UNC_MAX_PROBA, "margin": UNC_MARGIN}
}
with open(os.path.join(OUTDIR, "inference_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"[done] Saved: {pred_path}")
print(f"[done] Saved: {unc_path} (for human labeling)")
