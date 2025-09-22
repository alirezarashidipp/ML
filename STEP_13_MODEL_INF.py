# -*- coding: utf-8 -*-
# Simple, aligned inference for XGBoost multiclass + Active-Learning shortlist
# Works with the training artifact produced earlier: {"model","label_encoder","features"}
# Optional SHAP (only on uncertain samples)

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# ========== Config ==========
UNLABELED_CSV  = "UNLABELED.csv"                               # ورودی بدون لیبل
MODEL_JOBLIB   = "runs_train/xgb_train_XXXX_model.joblib"      # آرتیفکت ترِین
OUTDIR         = "runs_infer"; os.makedirs(OUTDIR, exist_ok=True)

# آستانه‌های عدم‌قطعیت (Active Learning)
UNC_MAX_PROBA  = 0.60    # اگر max p < این، نامطمئن
UNC_MARGIN     = 0.20    # اگر (p1 - p2) < این، نامطمئن

# SHAP (اختیاری و ساده)
DO_SHAP        = False   # برای سادگی خاموش؛ اگر True شد فقط روی نامطمئن‌ها
TOPK_SHAP      = 5
MAX_SHAP_SAMPLES = 200   # حداکثر تعداد نمونه برای SHAP (نامطمئن‌ترین‌ها)

ID_COL_FALLBACK = None   # اگر "Key" نبود، این را ست کن؛ وگرنه می‌افتد روی ستون اول

# ========== Load model & assets ==========
pack = joblib.load(MODEL_JOBLIB)
model     = pack["model"]
le        = pack["label_encoder"]
FEATURES  = pack["features"]
classes   = list(le.classes_)
n_classes = len(classes)

best_ntree = getattr(model, "best_ntree_limit", None)
it_range = (0, best_ntree) if best_ntree else None

# اگر در آینده خواستی میانهٔ ویژگی‌ها را هم ذخیره کنی، اینجا مصرف می‌شود:
feature_medians = pack.get("feature_medians", None)

# ========== Load unlabeled ==========
df = pd.read_csv(UNLABELED_CSV)

# انتخاب ستون ID
if "Key" in df.columns:
    id_col = "Key"
elif ID_COL_FALLBACK and ID_COL_FALLBACK in df.columns:
    id_col = ID_COL_FALLBACK
else:
    id_col = df.columns[0]

# چکِ فیچرها و چینش دقیق
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")
X_df = df.reindex(columns=FEATURES).copy()

# ناس‌ها: اگر میانهٔ ترِین داری مصرف کن؛ وگرنه میانهٔ همین ورودی
if feature_medians is not None:
    med = pd.Series(feature_medians)
    X_df = X_df.fillna(med)
else:
    # ساده و محافظه‌کار
    X_df = X_df.fillna(X_df.median(numeric_only=True))

# ========== Predict ==========
proba = model.predict_proba(X_df, iteration_range=it_range)  # (N, C)
pred_idx = proba.argmax(axis=1)
pred_label = le.inverse_transform(pred_idx)

# ========== Uncertainty signals ==========
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
uncertain_idx  = np.where(uncertain_mask)[0]

# ========== Build outputs (preds + probs + uncertainty) ==========
out = pd.DataFrame({
    id_col: df[id_col].values,
    "pred_label": pred_label,
    "pred_class_idx": pred_idx,
    "confidence": np.round(maxp, 6),
    "margin": np.round(margin, 6),
    "entropy": np.round(entropy, 6),
    "uncertain_flag": uncertain_mask.astype(int),
})

# ستون‌های احتمال برای هر کلاس
for j, cls in enumerate(classes):
    out[f"proba_{cls}"] = np.round(proba[:, j], 6)

# ========== Optional SHAP on uncertain subset (simple & small) ==========
if DO_SHAP and len(uncertain_idx) > 0:
    import shap
    # فقط روی نامطمئن‌ترین‌ها (بر اساس کمترین margin)
    ord_idx = uncertain_idx[np.argsort(margin[uncertain_idx])]
    take = ord_idx[:min(MAX_SHAP_SAMPLES, len(ord_idx))]
    X_sel = X_df.iloc[take]

    # توضیح: برای اسکیکت-API چندکلاسه معمولاً لیست برمی‌گرداند
    explainer = shap.TreeExplainer(model)  # ساده
    vals = explainer.shap_values(X_sel)    # list[C] از (n,d)

    # بردارهای SHAP مطابق کلاس پیش‌بینی‌شدهٔ هر ردیف
    topk_feats, topk_vals, topk_raw = [], [], []
    fnames = X_df.columns.tolist()

    # اگر لیست است (چندکلاسه)
    if isinstance(vals, list):
        # vals[c].shape ==> (n, d)
        for r, gi in enumerate(take):
            c = pred_idx[gi]
            vec = vals[c][r]  # (d,)
            order = np.argsort(np.abs(vec))[::-1][:TOPK_SHAP]
            topk_feats.append("|".join(fnames[i] for i in order))
            topk_vals.append("|".join(f"{vec[i]:+0.6f}" for i in order))
            topk_raw.append("|".join(f"{X_df.iloc[gi, i]:0.6f}" for i in order))
    else:
        # تک‌کلاسه (بعید در این کار)، نگاشت ساده
        arr = np.asarray(vals)  # (n, d)
        for r, gi in enumerate(take):
            vec = arr[r]
            order = np.argsort(np.abs(vec))[::-1][:TOPK_SHAP]
            topk_feats.append("|".join(fnames[i] for i in order))
            topk_vals.append("|".join(f"{vec[i]:+0.6f}" for i in order))
            topk_raw.append("|".join(f"{X_df.iloc[gi, i]:0.6f}" for i in order))

    shap_df = pd.DataFrame({
        id_col: df.loc[take, id_col].values,
        f"shap_top{TOPK_SHAP}_features": topk_feats,
        f"shap_top{TOPK_SHAP}_values": topk_vals,
        f"shap_top{TOPK_SHAP}_raw_x": topk_raw,
    })
    out = out.merge(shap_df, on=id_col, how="left")

# ========== Save ==========
# پوشهٔ مخصوص این ران
ts = __import__("time").strftime("%Y%m%d_%H%M%S")
base = os.path.join(OUTDIR, f"infer_{ts}")
os.makedirs(base, exist_ok=True)

pred_path = os.path.join(base, "predictions.csv")
out.to_csv(pred_path, index=False, encoding="utf-8")

# خروجی Active Learning (فقط نامطمئن‌ها، مرتب‌شده)
unc = out[out["uncertain_flag"] == 1].copy().sort_values(
    by=["margin", "confidence", "entropy"], ascending=[True, True, False]
)
unc_path = os.path.join(base, "uncertain_for_labeling.csv")
unc_cols_min = [id_col, "pred_label", "confidence", "margin", "entropy"]
unc[unc_cols_min].to_csv(unc_path, index=False, encoding="utf-8")

# یک لیست ID ساده برای تسهیل لیبل‌گذاری دستی
with open(os.path.join(base, "active_learning_ids.txt"), "w", encoding="utf-8") as f:
    for v in unc[id_col].tolist():
        f.write(str(v) + "\n")

# متا
meta = {
    "model_joblib": MODEL_JOBLIB,
    "num_samples": int(len(df)),
    "num_uncertain": int(len(unc)),
    "classes": classes,
    "thresholds": {"max_proba": UNC_MAX_PROBA, "margin": UNC_MARGIN},
    "best_ntree_limit_used": int(best_ntree) if best_ntree else None,
    "do_shap": DO_SHAP,
}
with open(os.path.join(base, "inference_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"[done] predictions  -> {pred_path}")
print(f"[done] to_label    -> {unc_path}  (Active Learning queue: {len(unc)})")
