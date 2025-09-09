# STEP_23_AC_SEPERATION.py
import pandas as pd
import re

INPUT_CSV  = "STEP_4_LANG_SEP.csv"
OUTPUT_CSV = "2_STEP_23_AC_SEPERATION.csv"

PATTERNS = [
    r"acceptance criteria",
    r"\bac\b",
]

pattern = re.compile("|".join(PATTERNS), re.IGNORECASE)

def split_ac(text: str):
    """Return (before, after) where 'after' starts at the first header match."""
    if not isinstance(text, str):
        return "", ""
    m = pattern.search(text)
    if not m:
        return text, ""
    return text[:m.start()], text[m.start():]

def main():
    df = pd.read_csv(INPUT_CSV, usecols=["Key", "ISSUE_DESC_STR_CLEANED"])

    before_list, after_list = [], []
    for text in df["ISSUE_DESC_STR_CLEANED"]:
        before, after = split_ac(text)
        before_list.append(before)
        after_list.append(after)

    df["ISSUE_DESC_STR_CLEANED"] = before_list
    df["AC_TEXT"] = after_list

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("Saved:", OUTPUT_CSV)
    print(df.head())

if __name__ == "__main__":
    main()
