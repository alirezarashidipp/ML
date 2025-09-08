import pandas as pd
import numpy as np

INPUT_CSV   = "STEP_0_DA_JIRA_MASTER_with_lang.csv"
OUT_EN_CSV  = "STEP_0_DA_JIRA_MASTER_en80.csv"
OUT_REST_CSV = "STEP_0_DA_JIRA_MASTER_non_en_or_empty.csv"

if __name__ == "__main__":
    cols = ["Key", "Description", "ISSUE_DESC_STR_CLEANED", "TEXT_LANG", "CONFIDENCE"]
    df = pd.read_csv(INPUT_CSV, usecols=cols)

    # English gate: 'en' and confidence >= 0.80
    conf = pd.to_numeric(df["CONFIDENCE"], errors="coerce")
    mask_en = (df["TEXT_LANG"].str.lower() == "en") & (conf >= 0.80)

    # 1) English-only: keep columns unchanged
    df_en = df.loc[mask_en].copy()
    df_en.to_csv(OUT_EN_CSV, index=False, encoding="utf-8-sig")

    # 2) Others: add Final_Desc
    df_rest = df.loc[~mask_en].copy()
    empty_mask = df_rest["ISSUE_DESC_STR_CLEANED"].isna() | (df_rest["ISSUE_DESC_STR_CLEANED"].astype(str).str.strip() == "")
    df_rest["Final_Desc"] = np.where(empty_mask, "Desc is empty when cleaned up", "Desc is not English")
    df_rest.to_csv(OUT_REST_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved {OUT_EN_CSV}: {len(df_en)} rows")
    print(f"Saved {OUT_REST_CSV}: {len(df_rest)} rows")
