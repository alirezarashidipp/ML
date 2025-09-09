import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------- Config --------
INPUT_CSV   = "merged_features.csv"          # your input file
GO_OUT_CSV  = "GO_GET_HUMAN_LABEL.csv"       # core samples for human labeling
UNLAB_OUT   = "UNLABELLED_DATA.csv"          # the rest (unlabeled pool)
N_CLUSTERS  = 5
TOP_N       = 10
ID_COL      = "Key"                          # your ID column name
EXCLUDE_COLS = [ID_COL, "ISSUE_DESC_STR_CLEANED"]  # non-numeric/text columns to exclude from clustering

# -------- 1) Load --------
df = pd.read_csv(INPUT_CSV)

# -------- 2) Numeric features only (exclude ID/text) --------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c not in EXCLUDE_COLS]
if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found for clustering.")

# -------- 3) Clean NaN/Inf --------
X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# report missing columns counts (optional)
missing = X_raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    print("Columns with missing values (before imputation):")
    print(missing.to_string())

# drop rows with all-NaN features (optional but safe)
all_nan_mask = X_raw.isna().all(axis=1)
if all_nan_mask.any():
    print(f"Dropping {all_nan_mask.sum()} rows with all-NaN features.")
    df = df.loc[~all_nan_mask].reset_index(drop=True)
    X_raw = X_raw.loc[~all_nan_mask].reset_index(drop=True)

# -------- 4) Impute -> Scale --------
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X_raw)
X = scaler.fit_transform(X_imputed)

# -------- 5) KMeans --------
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# -------- 6) Distances to centroid --------
dists = np.linalg.norm(X - centers[labels], axis=1)

# meta table to help selection (keep original index)
meta = pd.DataFrame({
    "cluster": labels,
    "dist_to_centroid": dists
}, index=df.index)

# -------- 7) Pick TOP_N nearest-to-centroid per cluster --------
core_idx = (
    meta.sort_values(["cluster", "dist_to_centroid"], ascending=[True, True])
        .groupby("cluster", as_index=False, group_keys=False)
        .head(TOP_N)
        .index
)

# -------- 8) Build GO_GET_HUMAN_LABEL.csv (all cols + empty HUMAN_LABEL) --------
go_df = df.loc[core_idx].copy()
go_df["HUMAN_LABEL"] = ""   # empty column for human labeling
go_df.to_csv(GO_OUT_CSV, index=False)

# -------- 9) Build UNLABELLED_DATA.csv (exclude GO ids; has all original cols) --------
rest_df = df.drop(index=core_idx).copy()
rest_df.to_csv(UNLAB_OUT, index=False)

# -------- 10) Logs --------
counts = meta.loc[core_idx, "cluster"].value_counts().sort_index()
print("Core samples selected per cluster:")
print(counts.to_string())
print(f"Saved: {GO_OUT_CSV}  (central {TOP_N} per cluster, with HUMAN_LABEL empty)")
print(f"Saved: {UNLAB_OUT}   (remaining rows, no HUMAN_LABEL)")



