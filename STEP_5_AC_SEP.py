import pandas as pd
import re

INPUT_CSV  = "STEP_4_LANG_SEP.csv"
OUTPUT_CSV = "2_STEP_23_AC_SEPERATION.csv"

# put all patterns here; add new scenarios easily
PATTERNS = [
    r"acceptance criteria",
    r"\bac\b",          # standalone "ac"
    # r"criteria section",  # example new one
]

pattern = re.compile("|".join(PATTERNS), re.IGNORECASE)

def main():
    df = pd.read_csv(INPUT_CSV, usecols=["Key", "ISSUE_DESC_STR_CLEANED"])

    before_list, after_list = [], []

    for text in df["ISSUE_DESC_STR_CLEANED"]:
        if isinstance(text, str):
            m = pattern.search(text)
            if m:
                before_list.append(text[:m.start()])
                after_list.append(text[m.start():])
            else:
                before_list.append(text)
                after_list.append("")
        else:
            before_list.append("")
            after_list.append("")

    df["ISSUE_DESC_STR_CLEANED"] = before_list
    df["AC_TEXT"] = after_list

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("Saved:", OUTPUT_CSV)
    print(df.head())

if __name__ == "__main__":
    main()
