# -*- coding: utf-8 -*-
# Jira/Confluence text cleaner (one-function, with examples)

import re, html
import pandas as pd
from ftfy import fix_text
import contractions

def clean_text(raw: str) -> str:
    # ---------- normalize ----------
    s = "" if raw is None else str(raw)                 # None → ""
    s = fix_text(s)                                     # "caf\u00e9" → "café"
    s = contractions.fix(s)                             # "don't" → "do not"
    s = html.unescape(s).replace("\u00A0", " ")         # "Tom &amp; Jerry" → "Tom & Jerry"
    s = re.sub(r"[\u200B-\u200D\u2060]", "", s)         # "zero\u200bwidth" → "zerowidth"
    s = re.sub(r"[ \t]+", " ", s).lower()               # "Hello   World" → "hello world"

    # ---------- remove blocks ----------
    s = re.sub(r"\{(code|noformat)[^}]*\}[\s\S]*?\{\1\}", " ", s, flags=re.I)  # "{code}x{code}" → " "
    s = re.sub(r"```[\s\S]*?```", " ", s)               # "```print(hi)```" → " "
    s = re.sub(r"![^!\n]+!", " ", s)                    # "!image.png!" → " "
    s = re.sub(r"\[\^[^\]]+\]", " ", s)                 # "[^attachment]" → " "
    s = re.sub(r"https?://\S+|www\.\S+", " ", s, flags=re.I)  # "see https://abc.com" → "see  "
    s = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", s)  # "mail me: x@y.com" → "mail me:  "
    s = re.sub(r"\b[A-Z][A-Z0-9]+-\d+\b", " ", s)       # "BUG-1234 fixed" → " fixed"
    s = re.sub(r"<[^>]+>", " ", s)                      # "<b>bold</b>" → " bold "

    # ---------- remove styles ----------
    s = re.sub(r"\*([^\*]+)\*", r"\1", s)               # "*bold*" → "bold"
    s = re.sub(r"_([^_]+)_", r"\1", s)                  # "_italic_" → "italic"
    s = re.sub(r"(?<!\w)-([^-]+?)-(?!\w)", r"\1", s)    # "-deleted-" → "deleted"
    s = re.sub(r"\+([^+]+)\+", r"\1", s)                # "+inserted+" → "inserted"
    s = re.sub(r"\{\{([^}]+)\}\}", r"\1", s)            # "{{mono}}" → "mono"
    s = re.sub(r"\{color[^}]*\}|\{color\}", " ", s, flags=re.I)  # "{color:red}" → " "
    s = re.sub(r"\{anchor[^}]*\}", " ", s, flags=re.I)  # "{anchor:here}" → " "

    # ---------- remove abbreviations ----------
    s = re.sub(r"\bi\.e\.\b", " ", s, flags=re.I)       # "i.e." → " "
    s = re.sub(r"\be\.g\.\b", " ", s, flags=re.I)       # "e.g." → " "
    s = re.sub(r"\betc\.\b", " ", s, flags=re.I)        # "etc." → " "

    # ---------- hyphen rule ----------
    s = s.replace("-", " ")                             # "non-functional" → "non functional"

    # ---------- remove versions + numbers ----------
    s = re.sub(r"\bv\.?\d+(?:\.\d+){1,3}\b", " ", s, flags=re.I)  # "v1.2.3" → " "
    s = re.sub(r"\b\d+\b", " ", s)                      # "25 bugs" → " bugs"

    # ---------- keep only wanted punct ----------
    s = "".join(ch if ch.isalnum() or ch in ".!?;," or ch.isspace() else " " for ch in s)
    # "hello@world!" → "hello world!"

    # ---------- split lines & clean headings ----------
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        t = re.sub(r"^h[1-6]\.\s*|^\s*#{1,6}\s+|^bq\.\s*", "", t)  # "h1. Title" → "Title"
        t = re.sub(r"^\s*[\*\+\#]+\s+|^\s*\d+\.\s+", "", t)        # "- item" → "item"
        if re.match(r"^\s*\|\|", t):  # table header
            continue
        if re.match(r"^\s*\|", t):   # table row
            cells = [c.strip() for c in t.strip("|").split("|") if c.strip()]
            if cells:
                lines.append(", ".join(cells) + ".")  # "|a|b|" → "a, b."
        else:
            lines.append(t)

    # ---------- punctuation rules (newline → , or .) ----------
    out = []
    for i, line in enumerate(lines):
        words = len(line.split())
        if i < len(lines) - 1:               #
            if words <= 4:
                out.append(line + ",")       # "add logs now" → "add logs now,"
            else:
                out.append(line + ".")       # "we need to fix a bug" → "we need to fix a bug."
        else:
            out.append(line)
    s = " ".join(out).strip()

    # ---------- ensure ending punctuation ----------
    if s and s[-1] not in ".!?;:":
        s += "."

    # ---------- normalize spaces ----------
    s = re.sub(r"\s+", " ", s)
    return s


# --- Apply to CSV ---
def clean_df(df: pd.DataFrame, text_col: str, out_col: str = "ISSUE_DESC_STR_CLEANED") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = [clean_text(x) for x in out[text_col].astype(str).tolist()]
    return out



df = pd.read_csv("1_STEP_FLAG.csv")

# Apply cleaning
df["cleaned_str_final"] = df["cleaned_str"].astype(str).apply(clean_text)

# Save to new CSV
df.to_csv("cleaning.csv", index=False)
