# -*- coding: utf-8 -*-
# XGBoost (multiclass) + Simple K-Fold CV (with inner early stopping) + Holdout Train/Dev/Test
# MCC + ConfMat + Learning Curves + SHAP
# xgboost==2.1.*, shap>=0.44

import warnings, os, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, classification_report,
                             matthews_corrcoef)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
import shap
import joblib

# ========= Config =========
CSV_PATH   = "4_STEP_MERGE.csv"
LABEL_COL  = "HUMAN_LABEL"
FEATURES = [
    "avg_sentence_length_tokens","sentence_length_cv","clause_density_finite_per_sentence",
    "subordination_ratio","coordination_index_cconj_per_clause","avg_dependency_distance",
    "np_mean_length_tokens","nominalization_density","noun_verb_ratio",
    "passive_voice_percent_finite_verbs","mean_zipf_lemma","lexical_diversity_mattr",
    "technical_jargon_density","negation_density_per_sentence","idea_density_predicates_per_10w",
    "FRE_norm","FKGL_norm","Fog_norm","LW_norm","ARI_norm","SMOG_norm","CLI_norm",
    "DC_norm","LIX_norm","SPACHE_norm",
]
SEED   = 42
OUTDIR = "runs_train"; os.makedirs(OUTDIR, exist_ok=True)

# switches
DO_CV = True           # ← اگر فقط می‌خواهی هولدآوت باشد، این را False کن
CV_FOLDS = 5
CV_VAL_SIZE = 0.15     # سهم validation داخل هر فولد برای early stopping

# ========= Utils =========
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def plot_confmat(cm, classes, title, path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if cm.dtype!=int else f"{val:d}"
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)

def plot_learning_curve_generic(model, X, y, scoring, title, path_png, path_csv):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    sizes = np.linspace(0.1, 1.0, 8)
    ts, tr, va = learning_curve(model, X, y, cv=cv, train_sizes=sizes,
                                scoring=scoring, n_jobs=-1)
    tr_mean, va_mean = tr.mean(axis=1), va.mean(axis=1)
    fig = plt.figure()
    plt.plot(ts, tr_mean, 'o-', label=f"Train ({scoring})")
    plt.plot(ts, va_mean, 'o-', label=f"CV ({scoring})")
    plt.xlabel("Number of training samples")
    plt.ylabel(scoring)
    plt.title(title)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path_png, dpi=160); plt.close(fig)
    pd.DataFrame({"train_size": ts, "train_score": tr_mean, "cv_score": va_mean}) \
      .to_csv(path_csv, index=False, encoding="utf-8")

def compute_metrics(y_true, y_pred, y_proba, num_classes):
    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
    except Exception:
        auc_macro = np.nan
    cm_raw = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_m, 4),
        "recall_macro": round(rec_m, 4),
        "f1_macro": round(f1_m, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted": round(rec_w, 4),
        "f1_weighted": round(f1_w, 4),
        "mcc": round(mcc, 4),
        "roc_auc_ovo_macro": None if np.isnan(auc_macro) else round(float(auc_macro), 4),
    }, cm_raw, cm_norm

def new_model(num_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric=["mlogloss","merror"],
        random_state=SEED,
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        reg_lambda=1.0,
        early_stopping_rounds=100,
    )

# ========= Load =========
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[LABEL_COL])
X_df = df[FEATURES].copy()
y_raw = df[LABEL_COL].astype(str)

le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)
classes = list(le.classes_)
num_classes = len(classes)

# ========= Cross-Validation =========
ts = int(time.time())
base = os.path.join(OUTDIR, f"xgb_train_{ts}")

if DO_CV:
    cv_dir = base + "_cv"
    os.makedirs(cv_dir, exist_ok=True)

    kfold = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    all_cm_raw = np.zeros((num_classes, num_classes), dtype=float)

    for fold_idx, (tr_idx, te_idx) in enumerate(kfold.split(X_df, y), start=1):
        X_tr_full, X_te = X_df.iloc[tr_idx], X_df.iloc[te_idx]
        y_tr_full, y_te = y[tr_idx], y[te_idx]

        # inner split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=CV_VAL_SIZE,
            stratify=y_tr_full, random_state=SEED
        )

        # class weights per fold (on training partition)
        cw = compute_class_weight(class_weight="balanced",
                                  classes=np.unique(y_tr), y=y_tr)
        cw_map = {cls: w for cls, w in zip(np.unique(y_tr), cw)}
        sw_tr = np.array([cw_map[yy] for yy in y_tr])

        model_cv = new_model(num_classes)
        model_cv.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False
        )
        best_ntree = getattr(model_cv, "best_ntree_limit", None)

        it_range = (0, best_ntree) if best_ntree else None
        y_pred = model_cv.predict(X_te, iteration_range=it_range)
        y_proba = model_cv.predict_proba(X_te, iteration_range=it_range)

        mets, cm_raw, cm_norm = compute_metrics(y_te, y_pred, y_proba, num_classes)
        all_cm_raw += cm_raw

        # save per-fold confmats
        plot_confmat(cm_raw, classes, f"CV Fold {fold_idx} - ConfMat (raw)",
                     os.path.join(cv_dir, f"cv_fold{fold_idx}_confmat_raw.png"))
        plot_confmat(cm_norm, classes, f"CV Fold {fold_idx} - ConfMat (norm)",
                     os.path.join(cv_dir, f"cv_fold{fold_idx}_confmat_norm.png"))

        row = {"fold": fold_idx,
               "n_train_fold": int(len(X_tr)),
               "n_val_fold": int(len(X_val)),
               "n_test_fold": int(len(X_te)),
               "best_ntree_limit": int(best_ntree) if best_ntree else None}
        row.update(mets)
        fold_rows.append(row)

    # aggregate & save
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(cv_dir, "cv_fold_metrics.csv"), index=False, encoding="utf-8")

    summary = {"k_folds": CV_FOLDS}
    for col in ["accuracy","precision_macro","recall_macro","f1_macro",
                "precision_weighted","recall_weighted","f1_weighted","mcc"]:
        summary[f"{col}_mean"] = round(float(df_folds[col].mean()), 4)
        summary[f"{col}_std"]  = round(float(df_folds[col].std(ddof=1)), 4)
    if "roc_auc_ovo_macro" in df_folds.columns:
        vals = df_folds["roc_auc_ovo_macro"].dropna()
        summary["roc_auc_ovo_macro_mean"] = None if vals.empty else round(float(vals.mean()), 4)
        summary["roc_auc_ovo_macro_std"]  = None if vals.empty else round(float(vals.std(ddof=1)), 4)

    save_json(summary, os.path.join(cv_dir, "cv_summary.json"))

    # average confusion matrix across folds (raw counts)
    avg_cm = all_cm_raw / CV_FOLDS
    plot_confmat(avg_cm, classes, "CV Average ConfMat (raw counts)",
                 os.path.join(cv_dir, "cv_avg_confmat_raw.png"))

# ========= Holdout Split (original flow) =========
X_tmp, X_test_df, y_tmp, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train_df, X_dev_df, y_train, y_dev = train_test_split(
    X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED
)

# ========= Class weights (holdout) =========
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_train), y=y_train)
cw_map = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
sample_weight = np.array([cw_map[yy] for yy in y_train])

# ========= Model (holdout) =========
model = new_model(num_classes)

# ========= Train (holdout) =========
model.fit(
    X_train_df, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_train_df, y_train), (X_dev_df, y_dev)],
    verbose=False
)

best_ntree = model.best_ntree_limit if hasattr(model, "best_ntree_limit") else None

# ========= Evaluate (holdout) =========
it_range = (0, best_ntree) if best_ntree else None
y_pred = model.predict(X_test_df, iteration_range=it_range)
y_proba = model.predict_proba(X_test_df, iteration_range=it_range)

acc = accuracy_score(y_test, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

try:
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")
except Exception:
    auc_macro = np.nan

cm_raw = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

cls_rep = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

# ========= Save (holdout) =========
model.save_model(base + "_model.json")
joblib.dump({"model": model, "label_encoder": le, "features": FEATURES}, base + "_model.joblib")

metrics = {
    "n_train": int(len(X_train_df)),
    "n_dev": int(len(X_dev_df)),
    "n_test": int(len(X_test_df)),
    "best_ntree_limit": int(best_ntree) if best_ntree else None,
    "classes": classes,
    "accuracy": round(acc, 4),
    "precision_macro": round(prec_m, 4),
    "recall_macro": round(rec_m, 4),
    "f1_macro": round(f1_m, 4),
    "precision_weighted": round(prec_w, 4),
    "recall_weighted": round(rec_w, 4),
    "f1_weighted": round(f1_w, 4),
    "mcc": round(mcc, 4),
    "roc_auc_ovo_macro": None if np.isnan(auc_macro) else round(float(auc_macro), 4),
    "classification_report": cls_rep,
    "seed": SEED,
    "params": model.get_params()
}
save_json(metrics, base + "_metrics.json")
pd.DataFrame([metrics]).to_csv(base + "_metrics.csv", index=False, encoding="utf-8")

plot_confmat(cm_raw, classes, "Confusion Matrix (Raw counts)", base + "_confmat_raw.png")
plot_confmat(cm_norm, classes, "Confusion Matrix (Normalized by true class)", base + "_confmat_norm.png")

# ========= Learning Curves (quick) =========
lc_model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    random_state=SEED,
    n_estimators=400, max_depth=6, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.9, tree_method="hist"
)
plot_learning_curve_generic(
    lc_model, X_train_df, y_train,
    scoring="accuracy",
    title="Learning Curve - Accuracy",
    path_png=base + "_lc_accuracy.png",
    path_csv=base + "_lc_accuracy.csv"
)
plot_learning_curve_generic(
    lc_model, X_train_df, y_train,
    scoring="f1_macro",
    title="Learning Curve - F1-macro",
    path_png=base + "_lc_f1macro.png",
    path_csv=base + "_lc_f1macro.csv"
)

# ========= Feature Importance =========
booster = model.get_booster()
fscore = booster.get_score(importance_type="gain")
imp_vals = [fscore.get(feat, 0.0) for feat in X_train_df.columns]

fig = plt.figure()
plt.barh(range(len(imp_vals)), imp_vals)
plt.yticks(range(len(imp_vals)), X_train_df.columns)
plt.xlabel("Gain")
plt.title("Feature Importance (gain)")
plt.tight_layout()
plt.savefig(base + "_feat_importance.png", dpi=160)
plt.close(fig)

# ========= SHAP (subset) =========
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
bg_idx = np.random.RandomState(SEED).choice(len(X_train_df), size=min(512, len(X_train_df)), replace=False)
bg = X_train_df.iloc[bg_idx]

k = min(100, len(X_test_df))
X_test_slice = X_test_df.iloc[:k]
shap_values = explainer.shap_values(X_test_slice)

for c_idx, c_name in enumerate(classes):
    try:
        fig = plt.figure()
        shap.summary_plot(shap_values[c_idx], X_test_slice, show=False)
        plt.title(f"SHAP Summary - class: {c_name}")
        plt.tight_layout()
        plt.savefig(f"{base}_shap_summary_class_{c_name}.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass

try:
    i = 0
    force = shap.force_plot(explainer.expected_value[c_idx], shap_values[c_idx][i,:], X_test_slice.iloc[i,:], matplotlib=False)
    shap.save_html(f"{base}_shap_force_example.html", force)
except Exception:
    pass

print("\n=== Saved under:", OUTDIR)
print("Base:", base)
print(json.dumps({k: v for k, v in metrics.items() if k not in ["classification_report","params"]}, indent=2))

# ========= ExplainerDashboard =========
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
explainer_dash = ClassifierExplainer(model, X_test_df, y_test, labels=classes)
dashboard = ExplainerDashboard(explainer_dash, title="XGBoost Readability Dashboard")
dashboard.run()
