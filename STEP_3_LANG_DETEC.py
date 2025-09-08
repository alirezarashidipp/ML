import pandas as pd
from langdetect import detect_langs


INPUT_CSV  = "STEP_0_DA_JIRA_MASTER.csv"
OUTPUT_CSV = "STEP_0_DA_JIRA_MASTER_with_lang.csv"

# فقط ستون‌های Key و Description را بخوان
df = pd.read_csv(INPUT_CSV, usecols=["Key", "Description"])

def detect_language(text):
    if not isinstance(text, str) or not text.strip():
        return "und", 0.0
    try:
        langs = detect_langs(text)
        if not langs:
            return "und", 0.0
        top = langs[0]
        return top.lang, round(top.prob, 3)
    except:
        return "und", 0.0


df["TEXT_LANG"], df["CONFIDENCE"] = zip(*df["Description"].map(detect_language))


df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(df.head())
