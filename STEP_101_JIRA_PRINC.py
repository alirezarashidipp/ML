# ============================================================
# STEP 101 - Embedding fallback for WHO/WHAT/WHY flags
# Input : STEP_100_JIRA_PRINC.csv
# Output: STEP_101_JIRA_PRINC.csv
# ============================================================

import pandas as pd
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util


# ---------------- Config ----------------
INPUT_CSV  = "STEP_100_JIRA_PRINC.csv"
OUTPUT_CSV = "STEP_101_JIRA_PRINC.csv"
TEXT_COL   = "ISSUE_DESC_STR_CLEANED"
KEY_COL    = "Key"

TARGET_FLAGS = ["has_role_defined", "has_goal_defined", "has_reason_defined"]

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\all_MiniLM_L6_v2-1.0.0"
embedder = SentenceTransformer(MODEL_PATH)


# ---------------- Prototypes ----------------
EMB_PROTOTYPES = {
    "has_role_defined": [
        "As a user, I want to",
        "As an administrator, I need to",
        "As a customer, I want to",
        "As a developer, I need to",
        "As a <persona>, I want to",
        "As a <role>, I need to",
    ],
    "has_goal_defined": [
        "I want to",
        "I need to",
        "We want to",
        "We need to",
        "The system should",
        "Allow users to",
        "Enable the user to",
        "Please add a feature to",
    ],
    "has_reason_defined": [
        "So that I can",
        "So we can",
        "In order to",
        "To reduce risk",
        "To improve user experience",
        "For compliance reasons",
        "To avoid incidents",
    ],
}

# Thresholds (tune later if needed)
EMB_THR = {
    "has_role_defined": 0.62,
    "has_goal_defined": 0.60,
    "has_reason_defined": 0.58,
}

# Precompute prototype embeddings once ✅
PROTO_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k, v in EMB_PROTOTYPES.items()}


# ---------------- Helpers ----------------
def split_chunks(text: str) -> List[str]:
    """
    Single-line input => no chunking.
    Use the whole text as one chunk.
    """
    if not isinstance(text, str):
        return []
    t = text.strip()
    return [t] if t else []


def embedding_upgrade_flags(text: str, flags: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(text, str) or not text.strip():
        return flags

    missing = [k for k in TARGET_FLAGS if int(flags.get(k, 0)) == 0]
    if not missing:
        return flags

    chunks = split_chunks(text)
    if not chunks:
        return flags

    # Encode once (single chunk)
    chunk_emb = embedder.encode(chunks, convert_to_tensor=True)

    for k in missing:
        sim = util.cos_sim(chunk_emb, PROTO_EMB[k])  # [1, num_protos]
        best = float(sim.max().item())
        if best >= EMB_THR[k]:
            flags[k] = 1

    return flags


# ---------------- Pipeline ----------------
def step_101_embedding():
    df = pd.read_csv(INPUT_CSV)

    for col in [KEY_COL, TEXT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for f in TARGET_FLAGS:
        if f not in df.columns:
            df[f] = 0

    before = {f: int(df[f].sum()) for f in TARGET_FLAGS}

    def _row_apply(row):
        flags = {f: int(row.get(f, 0)) for f in TARGET_FLAGS}
        upgraded = embedding_upgrade_flags(row[TEXT_COL], flags)
        for f in TARGET_FLAGS:
            row[f] = int(upgraded[f])
        return row

    df_out = df.apply(_row_apply, axis=1)

    after = {f: int(df_out[f].sum()) for f in TARGET_FLAGS}
    delta = {f: after[f] - before[f] for f in TARGET_FLAGS}

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"✅ Saved {len(df_out)} rows to {OUTPUT_CSV}")
    print("📌 Upgrades via embedding:")
    for f in TARGET_FLAGS:
        print(f" - {f}: +{delta[f]} (before={before[f]}, after={after[f]})")


if __name__ == "__main__":
    step_101_embedding()
