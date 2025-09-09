import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# ---- Config ----
INPUT_CSV   = "STEP_9_MERGE.csv"
GO_OUT_CSV  = "STEP_9_FOR_HUMAN_LABELLING.csv"
UNLAB_OUT   = "STEP_9_UNLABELLED_DATA.csv"
N_CLUSTERS  = 5
TOP_N       = 10
ID_COL      = "Key"
EXCLUDE_COLS = [ID_COL, "ISSUE_DESC_STR_CLEANED"]  # non-numeric/text columns to exclude

if __name__ == "__main__":
    # 1) Load
    df = pd.read_csv(INPUT_CSV)

    # 2) Numeric feature columns (exclude IDs/text)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in EXCLUDE_COLS]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    # 3) Clean NaN/Inf and drop rows with all-NaN features
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    keep_mask = X.notna().any(axis=1)
    df = df.loc[keep_mask].reset_index(drop=True)
    X = X.loc[keep_mask].reset_index(drop=True)

    # 4) Impute -> Scale (Robust to outliers)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = RobustScaler().fit_transform(X)

    # 5) KMeans (stable init)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # 6) Distances to own centroid
    dists = np.linalg.norm(X - centers[labels], axis=1)

    # 7) Pick TOP_N closest per cluster with a deterministic tie-break on ID_COL
    meta = pd.DataFrame({"cluster": labels, "dist": dists})
    # tie-break: rank by ID (string), stable ordering inside equal distances
    meta["tie"] = df[ID_COL].astype(str).rank(method="first")
    meta_sorted = meta.sort_values(["cluster", "dist", "tie"])

    core_idx = (
        meta_sorted.groupby("cluster", as_index=False, group_keys=False)
        .head(TOP_N)
        .index
    )

    # Optional: warn on small clusters
    counts = meta["cluster"].value_counts().sort_index()
    small = counts[counts < TOP_N]
    if not small.empty:
        print("Warning: small clusters (size < TOP_N):", small.to_dict())

    # 8) Outputs
    go_df = df.loc[core_idx].copy()
    go_df["HUMAN_LABEL"] = ""  # to be filled by human
    go_df.to_csv(GO_OUT_CSV, index=False)

    rest_df = df.drop(index=core_idx).copy()
    rest_df.to_csv(UNLAB_OUT, index=False)

    print(f"Saved {GO_OUT_CSV} (top {TOP_N} per cluster)")
    print(f"Saved {UNLAB_OUT} (remaining pool)")
    print("Selected per cluster:", counts.to_dict())
