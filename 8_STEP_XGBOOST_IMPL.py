import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import shap

# -------- Config --------
UNLABELED_CSV = "NEW_UNLABELED.csv"   # ← مسیر فایل بدون برچسب
ARTIFACTS     = "xgb_readability_artifacts.joblib"
UNC_MAX_PROBA = 0.50
UNC_MARGIN    = 0.10
TOPK_EXPLAIN  = 5
ID_COL_FALLBACK = None  # اگر ستونی به نام "Key" ندارید، اینجا نام شناسه را بگذارید؛ وگرنه از ستون اول استفاده می‌شود.

# -------- Load artifacts --------
art = joblib.load(ARTIFACTS)
model     = art["model"]
imputer   = art["imputer"]
le        = art["label_encoder"]
FEATURES  = art["features"]
explainer = art.get("explainer", None)  # ممکن است None نباشد

classes = le.classes_
n_classes = len(classes)

# -------- Load unlabeled data --------
df = pd.read_csv(UNLABELED_CSV)

# sanity: check features
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")

X = df[FEATURES].copy()
X_imp = imputer.transform(X)

# -------- Predict probs/labels --------
proba = model.predict_proba(X_imp)
pred_idx = proba.argmax(axis=1)
pred_label = le.inverse_transform(pred_idx)

# -------- Uncertainty signals --------
p_sorted = np.sort(proba, axis=1)[:, ::-1]
maxp   = p_sorted[:, 0]
margin = p_sorted[:, 0] - p_sorted[:, 1]
entropy = (-proba * np.log(np.clip(proba, 1e-12, 1))).sum(axis=1)
uncertain = (maxp < UNC_MAX_PROBA) | (margin < UNC_MARGIN)

# -------- SHAP per-predicted-class (robust) --------
def shap_pick_pred_class(shap_vals, yhat, n_samples, n_features, n_classes):
    if isinstance(shap_vals, list):
        out = np.zeros((n_samples, n_features))
        for c, sv in enumerate(shap_vals):
            m = (yhat == c)
            if np.any(m): out[m, :] = sv[m, :]
        return out
    arr = np.asarray(shap_vals)
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == n_classes and arr.shape[1] == n_samples:  # (C,N,F)
            out = np.zeros((n_samples, arr.shape[2]))
            for i in range(n_samples): out[i, :] = arr[yhat[i], i, :]
            return out
        if arr.shape[0] == n_samples and arr.shape[2] == n_classes:  # (N,F,C)
            out = np.zeros((n_samples, arr.shape[1]))
            for i in range(n_samples): out[i, :] = arr[i, :, yhat[i]]
            return out
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

def build_topk_strings(shap_pred, k=TOPK_EXPLAIN):
    out = []
    for r in range(shap_pred.shape[0]):
        contrib = shap_pred[r]
        abs_c = np.abs(contrib); s = abs_c.sum()
        pct = abs_c / s if s != 0 else np.zeros_like(abs_c)
        order = np.argsort(-pct)
        parts = [f"{FEATURES[i]}({contrib[i]:+0.3f},{pct[i]*100:0.1f}%)" for i in order[:k]]
        out.append("; ".join(parts))
    return out

# compute SHAP explanations if explainer available; else skip gracefully
try:
    if explainer is None:
        explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_imp)
    shap_pred = shap_pick_pred_class(shap_vals, pred_idx, X_imp.shape[0], X_imp.shape[1], n_classes)
    topk_explain = build_topk_strings(shap_pred, TOPK_EXPLAIN)
except Exception as e:
    topk_explain = [f"(SHAP unavailable: {e})"] * X_imp.shape[0]

# -------- Build outputs --------
# pick ID column
if "Key" in df.columns:
    id_col = "Key"
elif ID_COL_FALLBACK and ID_COL_FALLBACK in df.columns:
    id_col = ID_COL_FALLBACK
else:
    id_col = df.columns[0]  # fallback to first column

scored = pd.DataFrame({
    id_col: df[id_col].values,
    "pred_label": pred_label,
    "max_proba": maxp,
    "margin": margin,
    "entropy": entropy,
    "uncertain": uncertain,
    "topk_explain": topk_explain
})

# save all predictions
scored.to_csv("unlabeled_scored.csv", index=False)

# export uncertain-only for human labeling (most borderline first)
unc = scored[scored["uncertain"]].copy().sort_values(
    ["margin", "max_proba", "entropy"], ascending=[True, True, False]
)
unc.to_csv("uncertain_for_labeling_unlabeled.csv", index=False)

print("Saved: unlabeled_scored.csv")
print("Saved: uncertain_for_labeling_unlabeled.csv  ← برای human label")
