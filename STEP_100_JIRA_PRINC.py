
import re
import spacy
import pandas as pd
from spacy.matcher import Matcher
from typing import Dict

# ---------------- Config ----------------
INPUT_CSV  = "STEP_4_LANG_SEP.csv"
OUTPUT_CSV = "STEP_100_JIRA_PRINC.csv"
TEXT_COL   = "ISSUE_DESC_STR_CLEANED"
KEY_COL    = "Key"

nlp = spacy.load("en_core_web_sm_qbf")
matcher = Matcher(nlp.vocab)

FEATURE_KEYS = [
    "has_acceptance_criteria",
    "has_role_defined",
    "has_goal_defined",
    "has_reason_defined",
]

# ---- Pattern definitions ----
PATTERNS = {
    "has_role_defined": [
        [{"LOWER": "as"}, {"LOWER": {"IN": ["a", "an", "the"]}}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
         "as a user",
    ],
    "has_goal_defined": [
        "i want to", "i would like to", "i need to", "i can", "i am able to", "wish to",
        "we want to", "we would like to", "we need to", "we can", "we are able to",
        "i'm able to", "we're able to", "i'd like to", "we'd like to",
        "im able to", "were able to", "id like to", "wed like to",
    ],
    "has_reason_defined": [
        "so that", "so i can", "in order to"
    ],
    "has_acceptance_criteria": [
        "acceptance criteria", "acceptance criterion", "acc. crit.", "ac1", "ac"
    ],
}


# ---- Build matcher patterns ----
def _phrase_to_pattern(phrase: str):
    return [{"LOWER": tok} for tok in phrase.split()]

for key, items in PATTERNS.items():
    spacy_patterns = []
    for item in items:
        if isinstance(item, str):
            spacy_patterns.append(_phrase_to_pattern(item))
        elif isinstance(item, list):
            spacy_patterns.append(item)
    if spacy_patterns:
        matcher.add(key, spacy_patterns)

# ---- Main function ----
def flag_story_quality(text: str) -> Dict[str, int]:
    if not isinstance(text, str):
        return {k: 0 for k in FEATURE_KEYS}

    flags = {k: 0 for k in FEATURE_KEYS}

    clean_text = re.sub(r"[\r\n\t]+", " ", text)
    clean_text = re.sub(r"[{}[\]*—\-_/#\\+]", " ", clean_text)
    clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

    doc = nlp(clean_text)
    for match_id, start, end in matcher(doc):
        feature = nlp.vocab.strings[match_id]
        flags[feature] = 1

    return flags

# ---- Pipeline ----
def main():
    df = pd.read_csv(INPUT_CSV)

    flag_rows = df[TEXT_COL].apply(flag_story_quality)
    flag_df = pd.DataFrame(list(flag_rows))

    df_out = pd.concat([df[[KEY_COL, TEXT_COL]], flag_df], axis=1)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"✅ Saved {len(df_out)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
