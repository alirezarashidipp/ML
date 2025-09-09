import pandas as pd

INPUT_CSV   = "STEP_5_AC_SET.csv"
OUTPUT_MAIN = "STEP_5_AC_SET_main.csv"
OUTPUT_ONLY_AC = "STEP_5_AC_SET_only_ac.csv"

DESC_COL = "ISSUE_DESCC_STR_CLEANED"
AC_COL   = "AC_TEXT"
ID_COL   = "Key"

def main():
    df = pd.read_csv(INPUT_CSV, usecols=[ID_COL, DESC_COL, AC_COL])

    # 1) rows where DESC is not empty
    mask_desc_not_empty = df[DESC_COL].astype(str).str.strip() != ""
    df_main = df.loc[mask_desc_not_empty].copy()
    df_main.to_csv(OUTPUT_MAIN, index=False, encoding="utf-8-sig")

    # 2) rows where DESC is empty and AC is not empty
    mask_desc_empty = df[DESC_COL].astype(str).str.strip() == ""
    mask_ac_not_empty = df[AC_COL].astype(str).str.strip() != ""
    df_only_ac = df.loc[mask_desc_empty & mask_ac_not_empty].copy()
    df_only_ac["Final_Desc"] = "This Ticket contains only AC"
    df_only_ac.to_csv(OUTPUT_ONLY_AC, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_MAIN, len(df_main))
    print("Saved:", OUTPUT_ONLY_AC, len(df_only_ac))

if __name__ == "__main__":
    main()
