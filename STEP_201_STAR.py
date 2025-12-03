import numpy as np
import pandas as pd

# ---------- Config ----------
ALPHA = 0.7
BETA = 0.3
W_S = 3.0
W_A = 2.0
PENALTY_CAP = 3.0
CLARITY_MIN = 5
CLARITY_MAX = 95
STAR_MIN = 1.0

INPUT_FILE = "STEP_200_FINAL_DESC.csv"
OUTPUT_FILE = "STEP_201_STAR.csv"

COL_CLARITY = "Expected_Score_0_100"
COL_ROLE = "has_role_defined"
COL_GOAL = "has_goal_defined"
COL_VALUE = "has_reason_defined"
COL_AC = "has_acceptance_criteria"
COL_OUT = "STORY_QUALITY_INDEX"

# ---------- Main ----------
def main():
    df = pd.read_csv(INPUT_FILE)

    # فقط ردیف‌هایی که مقدار عددی دارند را نگه دار
    df[COL_CLARITY] = pd.to_numeric(df[COL_CLARITY], errors="coerce")
    df = df.dropna(subset=[COL_CLARITY])

    # محاسبات مدل ستاره‌ای
    clamped = df[COL_CLARITY].clip(CLARITY_MIN, CLARITY_MAX)
    x_c = (clamped - CLARITY_MIN) / (CLARITY_MAX - CLARITY_MIN)
    x_s = (df[COL_ROLE] + df[COL_GOAL] + df[COL_VALUE]) / 3.0
    x_a = df[COL_AC].astype(float)

    K = ((W_S * x_s) + (W_A * x_a) + 1) / (W_S + W_A + 1)
    I = np.power(x_c, ALPHA) * np.power(K, BETA)
    y = STAR_MIN + 4.0 * I
    y = np.where(df[COL_AC] == 0, np.minimum(y, PENALTY_CAP), y)
    y = np.maximum(STAR_MIN, np.round(y * 2) / 2)

    df[COL_OUT] = y

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
