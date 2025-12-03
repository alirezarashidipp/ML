import pandas as pd
from star_model import compute_star_index, COL_CLARITY, COL_ROLE, COL_GOAL, COL_VALUE, COL_AC, COL_OUT
from star_model import INPUT_FILE, OUTPUT_FILE  


ALPHA = 0.7
BETA = 0.3
W_S = 3.0
W_A = 2.0
PENALTY_CAP = 3.0
CLARITY_MIN = 5
CLARITY_MAX = 95
STAR_MIN = 1.0


data = [
    {COL_CLARITY: 10, COL_ROLE: 0, COL_GOAL: 0, COL_VALUE: 0, COL_AC: 0},
    {COL_CLARITY: 30, COL_ROLE: 1, COL_GOAL: 0, COL_VALUE: 0, COL_AC: 0},
    {COL_CLARITY: 60, COL_ROLE: 1, COL_GOAL: 1, COL_VALUE: 0, COL_AC: 1},
    {COL_CLARITY: 80, COL_ROLE: 1, COL_GOAL: 1, COL_VALUE: 1, COL_AC: 0},
    {COL_CLARITY: 90, COL_ROLE: 1, COL_GOAL: 1, COL_VALUE: 1, COL_AC: 1},
]

df = pd.DataFrame(data)


df_out = compute_star_index(
    df,
    alpha=ALPHA,
    beta=BETA,
    w_s=W_S,
    w_a=W_A,
    penalty_cap=PENALTY_CAP,
    clarity_min=CLARITY_MIN,
    clarity_max=CLARITY_MAX,
    star_min=STAR_MIN,
)

print(df_out[[COL_CLARITY, COL_ROLE, COL_GOAL, COL_VALUE, COL_AC, COL_OUT]])
