# ============================================================
# WHO detection (NLI: bart-large-mnli)
# - Local model: bart-large-mnli-tensors
# - Output: True/False + probability
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"

TEXT_COL = "ISSUE_DESC_STR_CLEANED"  # Jira description column
KEY_COL = "Key"                     # Jira key column (optional but recommended)

INPUT_CSV = "STEP_4_LANG_SEP.csv"
OUTPUT_CSV = "STEP_210_JIRA_NLI_WHO_OWNER.csv"

THRESHOLD = 0.60          # if P(ENTAILMENT) >= threshold -> True
DEVICE = -1               # -1=CPU, 0=GPU
BATCH_SIZE = 16


# ============================================================
# NLI setup (cached singleton)
# ============================================================

_NLI = None

def _get_nli():
    global _NLI
    if _NLI is None:
        _NLI = pipeline(
            task="text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=DEVICE,
            truncation=True,
            return_all_scores=True,
        )
    return _NLI


# ============================================================
# Core: detect_who
# ============================================================

def detect_who(
    text: str,
    threshold: float = THRESHOLD,
) -> Dict[str, Any]:
    """
    Detect whether the story explicitly states an owner/requester/actor (WHO).
    Returns:
      {
        "has_owner_defined": bool,
        "p_entailment": float,
        "p_neutral": float,
        "p_contradiction": float
      }
    """
    if text is None or not str(text).strip():
        return {
            "has_owner_defined": False,
            "p_entailment": 0.0,
            "p_neutral": 0.0,
            "p_contradiction": 0.0,
        }

    premise = str(text).strip()

    # Hypothesis crafted for "requester/owner is explicitly stated"
    hypothesis = (
        "The Jira story explicitly names a requester, owner, actor, or responsible role/team."
    )

    nli = _get_nli()
    scores = nli({"text": premise, "text_pair": hypothesis})[0]

    # scores example: [{"label":"ENTAILMENT","score":0.83}, ...]
    label2score = {d["label"].upper(): float(d["score"]) for d in scores}

    p_ent = label2score.get("ENTAILMENT", 0.0)
    p_neu = label2score.get("NEUTRAL", 0.0)
    p_con = label2score.get("CONTRADICTION", 0.0)

    has_owner = (p_ent >= threshold)

    return {
        "has_owner_defined": bool(has_owner),
        "p_entailment": round(p_ent, 4),
        "p_neutral": round(p_neu, 4),
        "p_contradiction": round(p_con, 4),
    }


# ============================================================
# Batch: process_jira_csv
# ============================================================

def process_jira_csv(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
    text_col: str = TEXT_COL,
    key_col: str = KEY_COL,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")

    if key_col not in df.columns:
        # اگر Key نداری مشکلی نیست، فقط برای trace بهتره
        df[key_col] = None

    # Apply row-wise (safe + simple). If you need faster, we can vectorize later.
    results = df[text_col].apply(lambda t: detect_who(t, threshold=threshold))
    res_df = pd.json_normalize(results)

    out = pd.concat([df[[key_col, text_col]].copy(), res_df], axis=1)
    out.to_csv(output_csv, index=False)
    return out


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    out_df = process_jira_csv()
    print("Saved:", OUTPUT_CSV)
    print(out_df.head(5).to_string(index=False))
