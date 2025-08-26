# -*- coding: utf-8 -*-
# Simple MVP tests for cleaning pipeline

import pandas as pd
from cleaner import clean_text, clean_df   # فرض بر اینه کد اصلیت رو گذاشتی توی cleaner.py

cases = [
    {
        "raw": "Non-AIM users",
        "expected": "non aim users."
    },
    {
        "raw": "summary:\nas an Ops user, i want to check.",
        "expected": "summary, as an ops user, i want to check."
    },
    {
        "raw": "as a customer\ni want to do this and that so i have new thing\n\nacceptence criterai:\nadding more column\nhaving new data",
        "expected": "as a customer i want to do this and that so i have new thing. acceptence criterai: adding more column having new data."
    },
    {
        "raw": "*Acceptance Criteria:*\ngiven data works well.",
        "expected": "acceptance criteria: given data works well."
    },
]

print("=== Unit-like tests ===")
for i, c in enumerate(cases, 1):
    cleaned = clean_text(c["raw"])
    print(f"\nCase {i}:")
    print(" RAW      :", repr(c["raw"]))
    print(" CLEANED  :", repr(cleaned))
    print(" EXPECTED :", repr(c["expected"]))

# --------- Test on a DataFrame ---------
print("\n=== DataFrame test ===")
df = pd.DataFrame({"ISSUE_DESC_STR": [c["raw"] for c in cases]})
df_out = clean_df(df, "ISSUE_DESC_STR")
print(df_out.head())
df_out.to_csv("test_cleaned.csv", index=False, encoding="utf-8")
print("Saved: test_cleaned.csv")
