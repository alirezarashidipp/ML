import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

INPUT_CSV = "STEP_13_INFERENCE_WITH_FEATURES.csv"
OUTPUT_CSV = "STEP_14_ACTIVE_LEARNING_SELECTION.csv"
ID_COL = "Key"
CONFIDENCE_COL = "Confidence"
PROB_COLS = ["proba_Poor", "proba_Acceptable", "proba_Good"]
TARGET_N = 800
CONFIDENCE_THRESHOLD = 0.8
N_CLUSTERS = 80
FEATURE_COLS: List[str] = []

def compute_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    has_prob_cols = all(col in df.columns for col in PROB_COLS)
    if has_prob_cols:
        probs = df[PROB_COLS].values
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        df["uncertainty"] = 1.0 - margin
    else:
        df["uncertainty"] = 1.0 - df[CONFIDENCE_COL].astype(float)
    return df

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = set([CONFIDENCE_COL, "uncertainty"]) | set(PROB_COLS)
    if ID_COL in df.columns:
        exclude_cols.add(ID_COL)
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    return feature_cols

def select_diverse_uncertain_samples(df_low: pd.DataFrame, feature_cols: List[str], n_samples: int, n_clusters: int, random_state: int = 42) -> pd.DataFrame:
    if len(df_low) <= n_samples:
        return df_low.copy()
    n_clusters = min(n_clusters, len(df_low))
    X = df_low[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_low = df_low.copy()
    df_low["cluster"] = cluster_labels
    clusters = sorted(df_low["cluster"].unique())
    K = len(clusters)
    base_per_cluster = n_samples // K
    extra = n_samples % K
    selected_rows = []
    for i, c in enumerate(clusters):
        cluster_df = df_low[df_low["cluster"] == c]
        n_pick = base_per_cluster + (1 if i < extra else 0)
        n_pick = min(n_pick, len(cluster_df))
        if n_pick == 0:
            continue
        cluster_df_sorted = cluster_df.sort_values("uncertainty", ascending=False)
        selected_rows.append(cluster_df_sorted.head(n_pick))
    if not selected_rows:
        raise ValueError("No rows were selected from any cluster.")
    selected = pd.concat(selected_rows, ignore_index=True)
    if len(selected) > n_samples:
        selected = selected.sort_values("uncertainty", ascending=False).head(n_samples)
    return selected

def main(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV, feature_cols: Optional[List[str]] = None):
    df = pd.read_csv(input_csv)
    df = compute_uncertainty(df)
    df_low = df[df[CONFIDENCE_COL] < CONFIDENCE_THRESHOLD].copy()
    if feature_cols is None or len(feature_cols) == 0:
        feature_cols = infer_feature_columns(df_low)
    selected = select_diverse_uncertain_samples(df_low=df_low, feature_cols=feature_cols, n_samples=TARGET_N, n_clusters=N_CLUSTERS, random_state=42)
    selected.to_csv(output_csv, index=False)
    return selected

if __name__ == "__main__":
    main()
