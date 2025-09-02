# -*- coding: utf-8 -*-
import pandas as pd
import re, html
from ftfy import fix_text
import contractions

def clean_text(raw: str) -> str:
    s = "" if raw is None else str(raw)
    s = fix_text(s)
    s = contractions.fix(s)
    s = html.unescape(s).replace("\u00A0", " ")
    s = re.sub(r"[\u200B-\u200D\u2060]", "", s)
    s = re.sub(r"[ \t]+", " ", s).lower()

    # remove blocks
    s = re.sub(r"\{(code|noformat)[^}]*\}[\s\S]*?\{\1\}", " ", s, flags=re.I)
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"![^!\n]+!", " ", s)
    s = re.sub(r"\[\^[^\]]+\]", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s, flags=re.I)
    s = re.sub(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", s)
    s = re.sub(r"\b[A-Z][A-Z0-9]+-\d+\b", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)

    # remove styles
    s = re.sub(r"\*([^\*]+)\*", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"(?<!\w)-([^-]+?)-(?!\w)", r"\1", s)  # strike-through
    s = re.sub(r"\+([^+]+)\+", r"\1", s)
    s = re.sub(r"\{\{([^}]+)\}\}", r"\1", s)
    s = re.sub(r"\{color[^}]*\}|\{color\}", " ", s, flags=re.I)
    s = re.sub(r"\{anchor[^}]*\}", " ", s, flags=re.I)

    # remove abbreviations
    s = re.sub(r"\bi\.e\.\b", " ", s, flags=re.I)
    s = re.sub(r"\be\.g\.\b", " ", s, flags=re.I)
    s = re.sub(r"\betc\.\b", " ", s, flags=re.I)

    # hyphen rule: replace all "-" with space
    s = s.replace("-", " ")

    # remove versions + numbers
    s = re.sub(r"\bv\.?\d+(?:\.\d+){1,3}\b", " ", s, flags=re.I)
    s = re.sub(r"\b\d+\b", " ", s)

    # keep only wanted punct
    s = "".join(ch if ch.isalnum() or ch in ".!?;," or ch.isspace() else " " for ch in s)

    # split into lines
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        t = re.sub(r"^h[1-6]\.\s*|^\s*#{1,6}\s+|^bq\.\s*", "", t)
        t = re.sub(r"^\s*[\*\+\#]+\s+|^\s*\d+\.\s+", "", t)
        if re.match(r"^\s*\|\|", t):  # header row
            continue
        if re.match(r"^\s*\|", t):   # table row
            cells = [c.strip() for c in t.strip("|").split("|") if c.strip()]
            if cells:
                lines.append(", ".join(cells) + ".")
        else:
            lines.append(t)

    # punctuation rules
    out = []
    for i, line in enumerate(lines):
        words = len(line.split())
        if i < len(lines) - 1:
            if words <= 4:
                out.append(line + ",")
            else:
                out.append(line + ".")
        else:
            out.append(line)
    s = " ".join(out).strip()

    # ensure ending punctuation
    if s and s[-1] not in ".!?;:":
        s += "."

    # normalize spaces
    s = re.sub(r"\s+", " ", s)
    return s

# --------- Main job ---------
# Read CSV
df = pd.read_csv("1_STEP_FLAG.csv")

# Apply cleaning
df["cleaned_str_final"] = df["cleaned_str"].astype(str).apply(clean_text)

# Save to new CSV
df.to_csv("cleaning.csv", index=False)
