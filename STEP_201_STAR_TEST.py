import math
import pandas as pd

ALPHA = 0.7
BETA = 0.3
PENALTY_CAP = 3.0
CLARITY_MIN = 5
CLARITY_MAX = 95
STAR_MIN = 1.0

INPUT_FILE = "STEP_200_FINAL_DESC.csv"
OUTPUT_FILE = "STEP_201_STAR.csv"

COL_CLARITY = "Expected_Score_0-100"
COL_ROLE = "has_role_defined"
COL_GOAL = "has_goal_defined"
COL_VALUE = "has_reason_defined"
COL_AC = "has_acceptance_criteria"
COL_OUT = "STORY_QUALITY_INDEX"

def compute_star_robust(clarity: float, role: int, goal: int, value: int, ac_present: int) -> float:
    for flag, name in zip([role, goal, value, ac_present], ['role', 'goal', 'value', 'ac_present']):
        if flag not in (0, 1):
            raise ValueError(f"Input flag '{name}' must be 0 or 1. Received: {flag}")
    if not isinstance(clarity, (int, float)):
        raise TypeError(f"Clarity score must be numeric. Received: {type(clarity)}")
    clarity_clamped = min(max(clarity, CLARITY_MIN), CLARITY_MAX)
    x_c = (clarity_clamped - CLARITY_MIN) / (CLARITY_MAX - CLARITY_MIN)
    x_s = (role + goal + value) / 3.0
    x_a = float(ac_present)
    K = ((3 * x_s) + (2 * x_a) + 1) / 6.0
    I = math.pow(x_c, ALPHA) * math.pow(K, BETA)
    Y_cont = STAR_MIN + 4.0 * I
    if ac_present == 0:
        Y_capped = min(Y_cont, PENALTY_CAP)
    else:
        Y_capped = Y_cont
    Y_final = round(2 * Y_capped) / 2.0
    return max(STAR_MIN, Y_final)

def main():
    df = pd.read_csv(INPUT_FILE)
    required_cols = [COL_CLARITY, COL_ROLE, COL_GOAL, COL_VALUE, COL_AC]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {missing}")
    for c in [COL_ROLE, COL_GOAL, COL_VALUE, COL_AC]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
    clarity_num = pd.to_numeric(df[COL_CLARITY], errors="coerce")
    df[COL_OUT] = "No Data"
    mask_numeric = clarity_num.notna()
    if mask_numeric.any():
        roles = df.loc[mask_numeric, COL_ROLE].astype(int).values
        goals = df.loc[mask_numeric, COL_GOAL].astype(int).values
        values = df.loc[mask_numeric, COL_VALUE].astype(int).values
        acs = df.loc[mask_numeric, COL_AC].astype(int).values
        clar = clarity_num.loc[mask_numeric].values
        stars = []
        for cl, r, g, v, ac in zip(clar, roles, goals, values, acs):
            try:
                stars.append(compute_star_robust(cl, r, g, v, ac))
            except Exception:
                stars.append("No Data")
        df.loc[mask_numeric, COL_OUT] = stars
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
