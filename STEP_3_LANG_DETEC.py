import pandas as pd
from langdetect import detect_langs

def detect_language(text: str):
    """Return language code and confidence score"""
    if not isinstance(text, str) or not text.strip():
        return "und", 0.0
    try:
        langs = detect_langs(text)
        if not langs:
            return "und", 0.0
        top = langs[0]
        return top.lang, round(top.prob, 3)
    except Exception:
        return "und", 0.0

def main():
    input_csv   = "STEP_2_CLEANING.csv"
    output_lang = "STEP_3_LANG_DETECT.csv"
    output_empty = "STEP_3_EMPTY_SEPERATION_AFTER_CLEANING.csv"

    # Read required columns
    df = pd.read_csv(input_csv, usecols=["Key", "Description", "ISSUE_DESC_STR_CLEANED"])

    # Separate rows with empty cleaned descriptions
    df_empty = df[df["ISSUE_DESC_STR_CLEANED"].isna() | (df["ISSUE_DESC_STR_CLEANED"].str.strip() == "")]
    df_nonempty = df.drop(df_empty.index)

    # Create Final_Desc column with constant message
    df_empty_out = df_empty[["Key", "Description"]].copy()
    df_empty_out["Final_Desc"] = "Description became null after cleaning"

    # Save empty rows to separate CSV
    df_empty_out.to_csv(output_empty, index=False, encoding="utf-8-sig")

    # Run language detection on non-empty rows
    df_nonempty["TEXT_LANG"], df_nonempty["CONFIDENCE"] = zip(
        *df_nonempty["ISSUE_DESC_STR_CLEANED"].map(detect_language)
    )

    # Save language-detected output
    df_nonempty.to_csv(output_lang, index=False, encoding="utf-8-sig")

    # Preview
    print(f"✅ Empty rows separated: {len(df_empty)}")
    print(f"✅ Processed rows with text: {len(df_nonempty)}")
    print(df_nonempty.head())

if __name__ == "__main__":
    main()
