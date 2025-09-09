# -*- coding: utf-8 -*-
# Ultra-minimal sampling pipeline (clean & compact)

import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# ---- Config ----
INPUT_CSV      = "STEP_9_MERGE.csv"
GO_OUT_CSV     = "STEP_10_FOR_HUMAN_LABELLING.csv"
UNLAB_OUT_CSV  = "STEP_10_UNLABELLED_DATA.csv"
ID_COL         = "Key"
EXCLUDE_COLS   = [ID_COL, "ISSUE_DESC_STR_CLEANED"]  # non-numeric/text cols
PCT_PER_CLUST  = 0.06   # 3% near + 3% far
MIN_PER_CLUST  = 5
MAX_PER_CLUST  = 30
MISS_ROW_MAX   = 0.60
K_MIN, K_MAX   = 2, 12
RSEED          = 42

# ---- Load & feature selection ----
df0 = pd.read_csv(INPUT_CSV)
num_cols = df0.select_dtypes(include=[np.number]).columns.tolist()
feat = [c for c in num_cols if c not in EXCLUDE_COLS]
if not feat: raise ValueError("No numeric feature columns found.")

# ---- Gentle row filter ----
miss = df0[feat].isna().mean(axis=1)
df1 = df0.loc[miss <= MISS_ROW_MAX].copy()                # drop rows >60% missing
Xr  = df1[feat].replace([np.inf, -np.inf], np.nan)
df  = df1.loc[Xr.notna().any(axis=1)].reset_index(drop=True)  # drop all-NaN rows
Xr  = Xr.loc[Xr.notna().any(axis=1)].reset_index(drop=True)

# ---- Impute + Scale ----
imp = SimpleImputer(strategy="median").fit(Xr)
Xs  = imp.transform(Xr)
sc  = RobustScaler().fit(Xs)
X   = sc.transform(Xs)

# ---- Choose K (sqrt rule) ----
n = X.shape[0]
if n < K_MIN: raise ValueError(f"Not enough rows after filtering: {n}")
K = max(K_MIN, min(int(round(math.sqrt(n/2))), K_MAX))

# ---- KMeans + distances ----
km = KMeans(n_clusters=K, random_state=RSEED, n_init=10).fit(X)
lab = km.labels_
cen = km.cluster_centers_
dist = np.linalg.norm(X - cen[lab], axis=1)
meta = pd.DataFrame({"i": np.arange(n), "c": lab, "d": dist, "t": df[ID_COL].astype(str)})

# ---- Per-cluster near + far ----
sel = []
for c in sorted(meta.c.unique()):
    part = meta[meta.c == c].sort_values(["d", "t"], ascending=[True, True])
    m = len(part)
    q = max(MIN_PER_CLUST, min(int(round(PCT_PER_CLUST*m)), MAX_PER_CLUST, m))
    nc = (q + 1)//2
    nf = q - nc if q > 1 else (1 if m >= 2 else 0)
    near = part.head(nc).i.tolist()
    far  = part.sort_values(["d","t"], ascending=[False,True]).head(nf).i.tolist()
    sel += near + [i for i in far if i not in near]

sel = list(dict.fromkeys(sel))               # keep order, make unique
go_df = df.iloc[sel].copy()
go_df["HUMAN_LABEL"] = ""                    # empty label
go_df.to_csv(GO_OUT_CSV, index=False)

# ---- UNLAB = rest ----
rest_idx = [i for i in range(len(df)) if i not in set(go_df.index)]
df.iloc[rest_idx].to_csv(UNLAB_OUT_CSV, index=False)

print(f"Rows in: {len(df0)} | clustered: {n} | K={K} | GO: {len(go_df)} | UNLAB: {len(rest_idx)}")
