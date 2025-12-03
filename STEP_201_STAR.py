import numpy as np
import pandas as pd


INPUT_FILE = "STEP_200_FINAL_DESC.csv"
OUTPUT_FILE = "STEP_201_STAR.csv"


COL_CLARITY = "Expected_Score_0_100"
COL_ROLE = "has_role_defined"
COL_GOAL = "has_goal_defined"
COL_VALUE = "has_reason_defined"
COL_AC = "has_acceptance_criteria"
COL_OUT = "STORY_QUALITY_INDEX"



def compute_star_index(
    df: pd.DataFrame,
    alpha: float = 0.7,
    beta: float = 0.3,
    w_s: float = 3.0,
    w_a: float = 2.0,
    penalty_cap: float = 3.0,
    clarity_min: float = 5.0,
    clarity_max: float = 95.0,
    star_min: float = 1.0,
) -> pd.DataFrame:

    df = df.copy()
    df[COL_CLARITY] = pd.to_numeric(df[COL_CLARITY], errors="coerce")
    df = df.dropna(subset=[COL_CLARITY])

    clamped = df[COL_CLARITY].clip(clarity_min, clarity_max)
    x_c = (clamped - clarity_min) / (clarity_max - clarity_min)
    x_s = (df[COL_ROLE] + df[COL_GOAL] + df[COL_VALUE]) / 3.0
    x_a = df[COL_AC].astype(float)

    K = ((w_s * x_s) + (w_a * x_a) + 1) / (w_s + w_a + 1)
    I = np.power(x_c, alpha) * np.power(K, beta)
    y = star_min + 4.0 * I
    y = np.where(df[COL_AC] == 0, np.minimum(y, penalty_cap), y)
    y = np.maximum(star_min, np.round(y * 2) / 2)

    df[COL_OUT] = y
    return df


def main():
    df = pd.read_csv(INPUT_FILE)
    df_out = compute_star_index(df)
    df_out.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
