# -*- coding: utf-8 -*-
# Ordinal XGBoost training script (revised: unified multiclass softprob)
# Author: Ali R.
# Requires: xgboost>=2.1.*, scikit-learn, pandas, numpy, joblib

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix, classification_report,
    cohen_kappa_score, make_scorer   # <-- ADDED: make_scorer
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # <-- ADDED
from xgboost import XGBClassifier
import joblib

# ========= Config =========
CSV_PATH   = "STEP_10_FOR_HUMAN_LABELLING.csv"
LABEL_COL  = "HUMAN_LABEL"
FEATURES = [
    "avg_sentence_length_tokens","sentence_length_cv","clause_density_finite_per_sentence",
    "subordination_ratio","coordination_index_cconj_per_clause","avg_dependency_distance",
    "np_mean_length_tokens","nominalization_density","noun_verb_ratio",
    "passive_voice_percent_finite_verbs","mean_zipf_lemma","lexical_diversity_mattr",
    "technical_jargon_density","negation_density_per_sentence","idea_density_predicates_per_10w",
    "FRE_norm","FKGL_norm","Fog_norm","LW_norm","ARI_norm","SMOG_norm","CLI_norm",
    "DC_norm","LIX_norm","SPACHE_norm"
]

# ========= Load Data =========
df = pd.read_csv(CSV_PATH)
X = df[FEATURES].copy()
y = df[LABEL_COL].copy()
X = X.fillna(X.mean())

# ========= Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= Model =========
base_params = dict(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6
)

param_grid = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300]
}

model = XGBClassifier(**base_params)
grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_macro")
grid.fit(X_train, y_train, verbose=False)
best_model = grid.best_estimator_

# ========= Predict =========
y_pred = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)

# ========= Metrics =========
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
mcc = matthews_corrcoef(y_test, y_pred)
qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

cls_report = classification_report(y_test, y_pred, digits=3)
conf_mat = confusion_matrix(y_test, y_pred)

# Feature importance
imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ========= Management Summary =========
errors = conf_mat.copy()
np.fill_diagonal(errors, 0)
max_error_idx = np.unravel_index(np.argmax(errors), errors.shape)
error_summary = f"Most frequent error: True class {max_error_idx[0]} predicted as {max_error_idx[1]} ({errors[max_error_idx]} times)."
top_feat = imp.head(1).index[0]

summary_text = (
    f"=== Management Summary ===\n"
    f"Current accuracy is about {acc:.2f}.\n"
    f"{error_summary}\n"
    f"Most influential feature: {top_feat}.\n\n"
)
print(summary_text)

# ========= Cross-Validation (ADDED) =========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
qwk_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
mcc_scorer = make_scorer(matthews_corrcoef)

est_for_cv = XGBClassifier(**best_model.get_params())
cv_scores = cross_validate(
    est_for_cv, X, y, cv=cv, n_jobs=-1,
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "qwk": qwk_scorer,
        "mcc": mcc_scorer
    },
    return_train_score=False
)

cv_summary = {
    "cv_n_splits": 5,
    "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
    "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
    "cv_f1_macro_mean": float(np.mean(cv_scores["test_f1_macro"])),
    "cv_f1_macro_std": float(np.std(cv_scores["test_f1_macro"])),
    "cv_qwk_mean": float(np.mean(cv_scores["test_qwk"])),
    "cv_qwk_std": float(np.std(cv_scores["test_qwk"])),
    "cv_mcc_mean": float(np.mean(cv_scores["test_mcc"])),
    "cv_mcc_std": float(np.std(cv_scores["test_mcc"]))
}

# ========= Prepare outputs =========
os.makedirs("runs_train", exist_ok=True)

metrics_dict = {
    "accuracy": acc,
    "precision_macro": prec,
    "recall_macro": rec,
    "f1_macro": f1,
    "precision_weighted": prec_w,
    "recall_weighted": rec_w,
    "f1_weighted": f1_w,
    "mcc": mcc,
    "quadratic_weighted_kappa": qwk,
    "confusion_matrix": conf_mat.tolist(),
    "top_features": imp.head(10).to_dict(),
    "management_summary": summary_text,
    "cross_validation": cv_summary   # <-- ADDED
}

with open("runs_train/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

with open("runs_train/eval_report.txt", "w") as f:
    f.write(summary_text)
    f.write("=== Metrics on Test Set ===\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"Precision (macro): {prec:.3f}, Recall (macro): {rec:.3f}, F1 (macro): {f1:.3f}\n")
    f.write(f"Precision (weighted): {prec_w:.3f}, Recall (weighted): {rec_w:.3f}, F1 (weighted): {f1_w:.3f}\n")
    f.write(f"MCC: {mcc:.3f}\n")
    f.write(f"Quadratic Weighted Kappa: {qwk:.3f}\n\n")
    f.write("=== Classification Report ===\n")
    f.write(cls_report + "\n\n")
    f.write("=== Confusion Matrix ===\n")
    f.write(str(conf_mat) + "\n\n")
    f.write("=== Top Features ===\n")
    f.write(str(imp.head(10)) + "\n\n")
    f.write("=== 5-Fold Stratified Cross-Validation ===\n")  # <-- ADDED
    f.write(json.dumps(cv_summary, indent=2) + "\n")        # <-- ADDED

# ========= Save model =========
joblib.dump(best_model, "runs_train/xgb_softprob_model.joblib")
print("Model and reports saved to runs_train/")


try:
    from explainerdashboard import ClassifierExplainer, ExplainerDashboard

    # Label names (safe for both numeric and string labels)
    class_labels = [str(c) for c in sorted(pd.unique(y_train))]

    # Build explainer on test set (you can change to X_train if you prefer)
    explainer = ClassifierExplainer(
        best_model,
        X_test,
        y_test,
        labels=class_labels,
        model_output="probability"   # aligns with multi:softprob
    )

    # Optional: persist explainer (can be large because it may cache SHAP etc.)
    joblib.dump(explainer, "runs_train/xgb_explainer.joblib")

    # Create dashboard object + save config
    dashboard = ExplainerDashboard(
        explainer,
        title="XGBoost 3-class (softprob) Explainer",
        shap_interaction=False
    )
    dashboard.to_yaml("runs_train/explainer_dashboard.yaml")

    # Run only if explicitly requested (avoids blocking training script by default)
    if os.environ.get("RUN_DASHBOARD", "0") == "1":
        print("Starting ExplainerDashboard at http://127.0.0.1:8050")
        dashboard.run(port=8050)
    else:
        print("ExplainerDashboard config saved: runs_train/explainer_dashboard.yaml (set RUN_DASHBOARD=1 to run)")

except Exception as e:
    print("ExplainerDashboard not available or failed to initialize:", repr(e))

