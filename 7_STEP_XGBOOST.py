# -*- coding: utf-8 -*-
# XGBoost (multiclass) Training + Dev EarlyStopping + MCC + Normalized ConfMat
# + Learning Curves (Acc & F1-macro) + SHAP
# Works with xgboost==2.1.*, shap>=0.44

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
            plt.text(j, i, format(cm[i, j], '.2f' if cm.dtype!=int else 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
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

# ========= Load =========
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[LABEL_COL])
X_df = df[FEATURES].copy()
y_raw = df[LABEL_COL].astype(str)

le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)
classes = list(le.classes_)
num_classes = len(classes)

# ========= Split =========
X_tmp, X_test_df, y_tmp, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train_df, X_dev_df, y_train, y_dev = train_test_split(
    X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED
)

# ========= Class weights =========
class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y_train),
                                     y=y_train)
cw_map = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
sample_weight = np.array([cw_map[yy] for yy in y_train])

# ========= Model =========
model = XGBClassifier(
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
)

# ========= Train =========
model.fit(
    X_train_df, y_train,
    sample_weight=sample_weight,
    eval_set=[(X_train_df, y_train), (X_dev_df, y_dev)],
    early_stopping_rounds=100,
    verbose=False
)

best_ntree = model.best_ntree_limit if hasattr(model, "best_ntree_limit") else None

# ========= Evaluate =========
y_pred = model.predict(X_test_df, iteration_range=(0, best_ntree) if best_ntree else None)
y_proba = model.predict_proba(X_test_df, iteration_range=(0, best_ntree) if best_ntree else None)

acc = accuracy_score(y_test, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

try:
    auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")
except Exception:
    auc_macro = np.nan

cm_raw = confusion_matrix(y_test, y_pred)
cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

cls_rep = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

# ========= Save =========
ts = int(time.time())
base = os.path.join(OUTDIR, f"xgb_train_{ts}")

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
