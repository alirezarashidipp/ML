# -*- coding: utf-8 -*-
"""
Minimal-but-solid sampling for human labeling:
- Auto-K (silhouette) over a small K range
- Per-cluster sampling: 3% closest + 3% farthest with MIN/MAX caps
- Mild missingness filter before clustering (row-level)
- GO file keeps only rows with no NULL in feature_cols (with backfill)
- Persist imputer/scaler params + feature_cols for consistent inference
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

# ---------------- Config ----------------
INPUT_CSV      = "STEP_9_MERGE.csv"
GO_OUT_CSV     = "STEP_9_FOR_HUMAN_LABELLING.csv"
UNLAB_OUT_CSV  = "STEP_9_UNLABELLED_DATA.csv"

# columns
ID_COL         = "Key"
EXCLUDE_COLS   = [ID_COL, "ISSUE_DESC_STR_CLEANED"]  # non-numeric/text columns to exclude from features

# sampling policy
PCT_PER_CLUSTER = 0.06      # 3% near + 3% far
MIN_PER_CLUSTER = 5
MAX_PER_CLUSTER = 30

# missingness policy
MISSING_ROW_MAX_RATIO = 0.60  # drop rows with >60% missing over feature_cols

# auto-K policy
K_MIN = 2
K_MAX_CAP = 12
SILH_DELTA_TIE = 0.02         # prefer smaller K if within this margin
SILH_SAMPLE_MAX = 10000       # max rows for silhouette sampling to keep it light

# persistence
ARTIFACT_DIR = Path("sampling_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


# --------------- Helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

def select_numeric_feature_cols(df: pd.DataFrame, exclude):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

def filter_rows_by_missing_ratio(df: pd.DataFrame, feature_cols, max_ratio: float):
    # ratio of NaN across feature_cols
    miss_ratio = df[feature_cols].isna().mean(axis=1)
    keep_mask = miss_ratio <= max_ratio
    return df.loc[keep_mask].reset_index(drop=True), keep_mask

def auto_choose_k(X: np.ndarray, n: int, kmin: int, kmax_cap: int,
                  min_per_clust: int, silh_delta_tie: float, sample_max: int):
    # derive K_max from data size
    kmax = min(kmax_cap, max(kmin, int(round(math.sqrt(n / 2)))) )
    candidates = list(range(kmin, max(kmin + 1, kmax) + 1))
    if len(candidates) == 1:
        return candidates[0], {candidates[0]: 0.0}

    # sample for silhouette to keep it cheap
    if n > sample_max:
        rs = np.random.RandomState(42)
        idx = rs.choice(n, size=sample_max, replace=False)
        X_silh = X[idx]
    else:
        X_silh = X

    scores = {}
    penalties = {}  # penalize if any cluster size < 2*MIN_PER_CLUSTER
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # penalty check
        unique, counts = np.unique(labels, return_counts=True)
        too_small = (counts < (2 * min_per_clust)).any()
        penalties[k] = 0.05 if too_small else 0.0
        # silhouette (on sample)
        # map labels for sample indices if subsampled
        if X_silh.shape[0] == X.shape[0]:
            silh = silhouette_score(X_silh, labels, metric="euclidean")
        else:
            # quick mapping: refit predict on X_silh for consistency
            # (we could approximate by nearest centroid)
            silh = silhouette_score(X_silh, km.predict(X_silh), metric="euclidean")
        scores[k] = silh - penalties[k]

    # choose best; prefer smaller K on ties within threshold
    best_k = max(scores, key=lambda k: (scores[k], -k))  # preliminary
    # search for any smaller K within tie margin
    best_score = scores[best_k]
    smaller_within = [k for k in candidates if k < best_k and (best_score - scores[k]) <= silh_delta_tie]
    if smaller_within:
        best_k = max(smaller_within)  # largest among smaller ones within margin
    return best_k, scores


def per_cluster_counts(cluster_size: int):
    n_c = clamp(round(PCT_PER_CLUSTER * cluster_size), MIN_PER_CLUSTER, MAX_PER_CLUSTER)
    n_c = min(n_c, cluster_size)  # cannot exceed cluster size
    n_close = int(math.ceil(n_c / 2))
    n_far = n_c - n_close
    # ensure at least 1 far if possible and cluster has enough room
    if n_far == 0 and cluster_size >= 2:
        n_far = 1
        n_close = max(1, n_c - n_far)
    return n_close, n_far


def distances_to_centroids(X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    # L2 distance to own centroid
    return np.linalg.norm(X - centers[labels], axis=1)


def save_artifacts(feature_cols, imputer: SimpleImputer, scaler: RobustScaler):
    # feature list
    (ARTIFACT_DIR / "feature_cols.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    # imputer medians
    med = pd.Series(imputer.statistics_, index=feature_cols, name="median")
    med.to_csv(ARTIFACT_DIR / "imputer_medians.csv", index=True)
    # scaler center & scale
    sc_df = pd.DataFrame({
        "center_": getattr(scaler, "center_", np.zeros(len(feature_cols))),
        "scale_": getattr(scaler, "scale_", np.ones(len(feature_cols))),
    }, index=feature_cols)
    sc_df.to_csv(ARTIFACT_DIR / "robust_scaler_params.csv", index=True)
    # meta json
    meta = {
        "PCT_PER_CLUSTER": PCT_PER_CLUSTER,
        "MIN_PER_CLUSTER": MIN_PER_CLUSTER,
        "MAX_PER_CLUSTER": MAX_PER_CLUSTER,
        "MISSING_ROW_MAX_RATIO": MISSING_ROW_MAX_RATIO,
    }
    (ARTIFACT_DIR / "sampling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --------------- Main -------------------
if __name__ == "__main__":
    df0 = pd.read_csv(INPUT_CSV)
    if ID_COL not in df0.columns:
        raise ValueError(f"ID_COL '{ID_COL}' not found in input.")

    # 1) feature columns (numeric only, excluding declared non-numeric)
    feature_cols = select_numeric_feature_cols(df0, EXCLUDE_COLS)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    # 2) gentle row-level missingness filter (before impute to avoid zombie points)
    df1, keep_mask_lv1 = filter_rows_by_missing_ratio(df0, feature_cols, MISSING_ROW_MAX_RATIO)
    dropped_lv1 = int((~keep_mask_lv1).sum())

    # 3) replace +/-inf -> NaN, then impute & scale
    X_raw = df1[feature_cols].replace([np.inf, -np.inf], np.nan)
    # if a row is all-NaN after this, drop it (rare, but safe)
    keep_mask_lv2 = X_raw.notna().any(axis=1)
    df = df1.loc[keep_mask_lv2].reset_index(drop=True)
    X_raw = X_raw.loc[keep_mask_lv2].reset_index(drop=True)
    dropped_lv2 = int((~keep_mask_lv2).sum())

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = RobustScaler()
    X = scaler.fit_transform(X_imp)

    n = X.shape[0]
    if n < K_MIN:
        raise ValueError(f"Not enough rows for clustering after filtering: {n}")

    # 4) auto-K (lightweight silhouette)
    best_k, k_scores = auto_choose_k(
        X, n, K_MIN, K_MAX_CAP, MIN_PER_CLUSTER, SILH_DELTA_TIE, SILH_SAMPLE_MAX
    )
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    # 5) distances and per-cluster selection (3% near + 3% far with MIN/MAX)
    dists = distances_to_centroids(X, labels, centers)
    df_meta = pd.DataFrame({
        "cluster": labels,
        "dist": dists,
        "Key_tie": df[ID_COL].astype(str)  # deterministic tie-break
    }).reset_index().rename(columns={"index": "row_idx"})

    go_indices = []
    reserve_indices = []  # to refill after "no NULL" filtering

    for c in sorted(df_meta["cluster"].unique()):
        part = df_meta[df_meta["cluster"] == c].copy()
        part = part.sort_values(["dist", "Key_tie"], ascending=[True, True])
        size_c = part.shape[0]
        n_close, n_far = per_cluster_counts(size_c)

        # closest
        close_idx = part.head(n_close)["row_idx"].tolist()
        # farthest
        far_part = part.sort_values(["dist", "Key_tie"], ascending=[False, True])
        far_idx = far_part.head(n_far)["row_idx"].tolist()

        # keep an ordered list (close first, then far)
        chosen = close_idx + [i for i in far_idx if i not in close_idx]
        go_indices.extend(chosen)

        # build reserves (next closest + next farthest)
        # (used for backfilling if "no NULL" filter drops some rows in GO)
        next_close = part.iloc[n_close: n_close + MAX_PER_CLUSTER]["row_idx"].tolist()
        next_far   = far_part.iloc[n_far: n_far + MAX_PER_CLUSTER]["row_idx"].tolist()
        reserve = []
        for i in next_close:
            if i not in chosen and i not in reserve:
                reserve.append(i)
        for i in next_far:
            if i not in chosen and i not in reserve:
                reserve.append(i)
        reserve_indices.extend(reserve)

    go_indices = list(dict.fromkeys(go_indices))  # unique, keep order

    # 6) Build GO dataframe (then apply "no NULL in feature_cols" filter + backfill)
    go_df = df.iloc[go_indices].copy()

    # apply "no NULL in feature_cols" only for GO output
    non_null_mask = go_df[feature_cols].notna().all(axis=1)
    dropped_go_due_to_null = (~non_null_mask).sum()

    if dropped_go_due_to_null > 0:
        # backfill from reserves (that have no NULLs)
        need = int(dropped_go_due_to_null)
        candidates = df.iloc[reserve_indices]
        backfill_mask = candidates[feature_cols].notna().all(axis=1)
        backfill_idx = candidates.loc[backfill_mask].index.tolist()
        # take first 'need' distinct indices not already in go_indices
        used = set(go_df.index[non_null_mask].tolist())
        taken = []
        for idx in backfill_idx:
            if idx not in used:
                taken.append(idx)
                used.add(idx)
                if len(taken) >= need:
                    break
        # rebuild go_df
        go_df = pd.concat([df.iloc[go_indices][non_null_mask], df.loc[taken]], axis=0)
        # deduplicate in case of any overlap
        go_df = go_df.loc[~go_df.index.duplicated(keep="first")]

    # 7) exact-duplicate drop on feature_cols to avoid boring repeats
    before_dedup = go_df.shape[0]
    if feature_cols:
        go_df = go_df.loc[~go_df[feature_cols].duplicated(keep="first")]
    dedup_dropped = before_dedup - go_df.shape[0]

    # 8) add empty HUMAN_LABEL col and save GO
    go_df = go_df.copy()
    go_df["HUMAN_LABEL"] = ""
    go_df.to_csv(GO_OUT_CSV, index=False)

    # 9) UNLAB = all remaining rows (relative to df used for clustering)
    go_idx_set = set(go_df.index.tolist())
    rest_df = df.loc[[i for i in df.index if i not in go_idx_set]].copy()
    rest_df.to_csv(UNLAB_OUT_CSV, index=False)

    # 10) persist artifacts for reproducibility
    save_artifacts(feature_cols, imputer, scaler)

    # 11) logs / summary
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    print("\n==== Sampling Summary ====")
    print(f"Input rows: {df0.shape[0]}")
    print(f"Dropped by row-missingness (>{int(MISSING_ROW_MAX_RATIO*100)}%): {dropped_lv1}")
    print(f"Dropped all-NaN-after-inf-replace guard: {dropped_lv2}")
    print(f"Rows clustered: {df.shape[0]}")
    print(f"Auto-selected K: {best_k}")
    print("Silhouette (penalized) per K:", {k: round(v, 4) for k, v in k_scores.items()})
    print("Cluster sizes:", counts)
    print(f"GO before dedup: {before_dedup}, dedup removed: {dedup_dropped}")
    print(f"Saved GO: {GO_OUT_CSV} ({go_df.shape[0]} rows)")
    print(f"Saved UNLAB: {UNLAB_OUT_CSV} ({rest_df.shape[0]} rows)")
    print("Artifacts saved in:", str(ARTIFACT_DIR))
