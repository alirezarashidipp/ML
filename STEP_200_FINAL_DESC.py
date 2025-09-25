# -*- coding: utf-8 -*-
# STEP_16_FINAL_REPORT.py
# Merge predictions + Agile info and generate Final_Desc text
# Requires: pandas, numpy

import pandas as pd

# ========= Config =========
PRED_FILE  = "predictions.csv"
AGILE_FILE = "STEP_100_JIRA_PINC.csv"
OUT_FILE   = "final_report.csv"

# ========= Load =========
pred_df  = pd.read_csv(PRED_FILE)
agile_df = pd.read_csv(AGILE_FILE)

# ========= Merge on Key (outer join to keep all keys) =========
df = pd.merge(pred_df, agile_df, on="Key", how="outer")

# ========= Fill missing with "No info" =========
for col in ["Confidence","Final_Label","has_acceptance_criteria",
            "has_role_defined","has_goal_defined","has_reason_defined"]:
    if col in df.columns:
        df[col] = df[col].fillna("No info")

# ========= Helper to map binary → text =========
def inc_or_lack(val, name):
    if val == 1 or val == "1":
        return f"includes a {name}"
    elif val == 0 or val == "0":
        return f"lacks a {name}"
    else:
        return f"No info about {name}"

# ========= Build Final_Desc =========
desc_list = []
for _, row in df.iterrows():
    label = row["Final_Label"] if row["Final_Label"] != "No info" else "Unknown"
    role  = inc_or_lack(row["has_role_defined"], "defined Role")
    goal  = inc_or_lack(row["has_goal_defined"], "Goal")
    reason= inc_or_lack(row["has_reason_defined"], "Reason")
    ac    = inc_or_lack(row["has_acceptance_criteria"], "Acceptance Criteria")

    desc = f"This ticket is {label} in readability. It {role}, {goal}, {reason}, and {ac}."
    desc_list.append(desc)

df["Final_Desc"] = desc_list

# ========= Save =========
df.to_csv(OUT_FILE, index=False)
print(f"Final report saved to {OUT_FILE}")



