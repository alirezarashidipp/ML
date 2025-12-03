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
    input_csv  = "STEP_2_CLEANING.csv"
    output_csv = "STEP_3_LANG_DETECT.csv"

    # Read all three columns
    df = pd.read_csv(input_csv, usecols=["Key", "Description", "ISSUE_DESC_STR_CLEANED"])

    # Run detection on ISSUE_DESC_STR_CLEANED
    df["TEXT_LANG"], df["CONFIDENCE"] = zip(*df["ISSUE_DESC_STR_CLEANED"].map(detect_language))

    # Save output
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Preview
    print(df.head())

if __name__ == "__main__":
    main()
