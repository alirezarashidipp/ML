import pandas as pd
import numpy as np

# -------- Config --------
SCORED_CSV = "8_STEP_XGBOOST_unlabeled_scored.csv"   # scored unlabeled data
FLAGS_CSV  = "story_flags.csv"                        # file with 0/1 flags
OUT_CSV    = "FINAL_TWO_COLS.csv"
ID_COL     = "Key"                                    # ID column in BOTH files

# Confidence thresholds (kept simple)
MIN_MAX_PROBA = 0.60
MIN_MARGIN    = 0.15

# Readability/Agile/AC mappings (exact phrases requested)
READ_MAP = {2: "Easy to Understand", 1: "Normal", 0: "Hard to Read"}
AGILE_MAP = {True: "Agile Core OK", False: "Agile Core Missing"}
AC_MAP = {1: "AC Present", 0: "AC Missing"}

def to_idx(val):
    # Accept 0/1/2 or textual labels; returns 0/1/2 or None
    if pd.isna(val):
        return None
    try:
        iv = int(val)
        return iv if iv in (0,1,2) else None
    except Exception:
        s = str(val).strip().lower()
        if s in {"2","easy to understand","easy","simple","good"}: return 2
        if s in {"1","normal","acceptable","medium"}: return 1
        if s in {"0","hard to read","hard","complex","poor"}: return 0
        return None

# 1) Load scored and filter by confidence
sc = pd.read_csv(SCORED_CSV)
for col in (ID_COL,):
    if col not in sc.columns:
        raise ValueError(f"ID column '{col}' not found in {SCORED_CSV}")

if {"max_proba","margin"}.issubset(sc.columns):
    sc = sc[(sc["max_proba"] >= MIN_MAX_PROBA) & (sc["margin"] >= MIN_MARGIN)].copy()
elif "uncertain" in sc.columns:
    sc = sc[~sc["uncertain"].astype(bool)].copy()

label_candidates = ["pred_label_idx","pred_label","label","y_pred"]
label_col = next((c for c in label_candidates if c in sc.columns), None)
if label_col is None:
    raise ValueError(f"Label column not found. Tried: {label_candidates}")

read_idx = sc[label_col].apply(to_idx)
read_text = read_idx.map(READ_MAP).fillna("Hard to Read")  # safe default
read_df = sc[[ID_COL]].copy()
read_df["__read__"] = read_text

# 2) Load flags (0/1) and summarize Agile/AC
fl = pd.read_csv(FLAGS_CSV)
need = ["has_role_defined","has_goal_defined","has_reason_defined","has_acceptance_criteria"]
missing = [c for c in need if c not in fl.columns]
if missing:
    raise ValueError(f"Missing required columns in {FLAGS_CSV}: {missing}")
if ID_COL not in fl.columns:
    raise ValueError(f"ID column '{ID_COL}' not found in {FLAGS_CSV}")

fl[need] = fl[need].fillna(0).astype(int).clip(0,1)
core_sum = fl[["has_role_defined","has_goal_defined","has_reason_defined"]].sum(axis=1)
agile_txt = core_sum.ge(2).map(AGILE_MAP)
ac_txt = fl["has_acceptance_criteria"].map(AC_MAP)

ag_df = fl[[ID_COL]].copy()
ag_df["__agile__"] = agile_txt
ag_df["__ac__"] = ac_txt

# 3) Merge and build Final_Desc
m = read_df.merge(ag_df, on=ID_COL, how="left")
m["__agile__"] = m["__agile__"].fillna(AGILE_MAP[False])
m["__ac__"] = m["__ac__"].fillna(AC_MAP[0])

m["Final_Desc"] = m["__read__"] + " | " + m["__agile__"] + " | " + m["__ac__"]
out = m[[ID_COL, "Final_Desc"]].rename(columns={ID_COL: "key"})
out.to_csv(OUT_CSV, index=False)

print(f"Saved: {OUT_CSV}")
print(out.head().to_string(index=False))
