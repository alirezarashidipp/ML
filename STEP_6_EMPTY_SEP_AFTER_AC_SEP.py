import pandas as pd

INPUT_CSV        = "STEP_5_AC_SET.csv"
OUTPUT_MAIN      = "STEP_5_AC_SET_main.csv"
OUTPUT_ONLY_AC   = "STEP_5_AC_SET_only_ac.csv"

ID_COL   = "Key"
DESC_COL = "ISSUE_DESCC_STR_CLEANED"
AC_COL   = "AC_TEXT"

EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-"}

def is_empty_series(s: pd.Series) -> pd.Series:
    # True if value is NaN or "blank-like" after strip/lower
    s_stripped = s.astype(str).str.strip()
    return s.isna() | s_stripped.str.lower().isin(EMPTY_TOKENS)

def main():
    df = pd.read_csv(INPUT_CSV, usecols=[ID_COL, DESC_COL, AC_COL])
    # (اختیاری) اگر شک داری تیتر ستون‌ها فاصله‌ی اضافه داشته باشن:
    # df.rename(columns=lambda c: c.strip(), inplace=True)

    desc_empty = is_empty_series(df[DESC_COL])
    ac_empty   = is_empty_series(df[AC_COL])

    # 1) خروجی اصلی: DESC پر باشد
    df_main = df.loc[~desc_empty].copy()
    df_main.to_csv(OUTPUT_MAIN, index=False, encoding="utf-8-sig")

    # 2) خروجی دوم: DESC خالی و AC پر باشد
    df_only_ac = df.loc[desc_empty & ~ac_empty].copy()
    df_only_ac["Final_Desc"] = "This Ticket contains only AC"
    df_only_ac.to_csv(OUTPUT_ONLY_AC, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_MAIN, len(df_main))
    print("Saved:", OUTPUT_ONLY_AC, len(df_only_ac))

    # sanity check
    print("\nCounts → desc_empty:", int(desc_empty.sum()),
          "| ac_non_empty:", int((~ac_empty).sum()))

if __name__ == "__main__":
    main()
