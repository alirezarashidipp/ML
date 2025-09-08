# -*- coding: utf-8 -*-
# Final Jira/Confluence text cleaner (revised)
# - Hyphen rule: internal hyphens between word chars -> space (e.g., "Non-AIM" -> "Non AIM")
# - Short-line comma rule: if a line has <= SHORT_LINE_MAX_WORDS and is followed by a non-empty line,
#   join with a comma instead of a period
# - Normalizes text (ftfy, html, contractions)
# - Removes code/media/urls/emails/jira-ids/html tags
# - Handles wiki/markdown structure: headings, quotes, bullets, numbered lists
# - Flattens tables: drop header rows, join cells with ", ", end each row with "."
# - Converts "ac1:" into "acceptance criteria: ..."
# - Keeps versions/decimals (e.g., v1.2.3, 2.0), drops plain numbers
# - Keeps only punctuation: . ! ? ; ,
# - Sentence segmentation with spaCy (en_core_web_sm)
# - API: clean_text(str) -> str, clean_df(DataFrame, text_col, out_col="ISSUE_DESC_STR_CLEANED") -> DataFrame

import re
import html
import string
from dataclasses import dataclass
from typing import List
import pandas as pd
from ftfy import fix_text
import contractions
import spacy

# Load spaCy pipeline once
_NLP = spacy.load("en_core_web_sm")
if "sentencizer" not in _NLP.pipe_names and "senter" not in _NLP.pipe_names:
    _NLP.add_pipe("sentencizer")

# ---------- Regex ----------
RE_CODE_ATL = re.compile(r"\{(code|noformat)[^}]*\}[\s\S]*?\{\1\}", re.I)
RE_CODE_MD  = re.compile(r"```[\s\S]*?```", re.M)
RE_IMAGE    = re.compile(r"![^!\n]+!", re.I)
RE_ATTACH   = re.compile(r"\[\^[^\]]+\]", re.I)
RE_URL      = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL    = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
RE_JIRA     = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b", re.I)
RE_HTML     = re.compile(r"<[^>]+>")

RE_H_WIKI   = re.compile(r"^h[1-6]\.\s*", re.I)
RE_H_MD     = re.compile(r"^\s*#{1,6}\s+")
RE_BQ       = re.compile(r"^bq\.\s*", re.I)
RE_BULLET   = re.compile(r"^\s*[\*\-\+\#]+\s+")
RE_NUM_LI   = re.compile(r"^\s*\d+\.\s+")
RE_T_HDR    = re.compile(r"^\s*\|\|")
RE_T_ROW    = re.compile(r"^\s*\|")
RE_LINK_TXT = re.compile(r"\[([^\]\|]+)\|[^\]]+\]")     # [text|url] -> text
RE_LINK_BRK = re.compile(r"\[([^\]]+)\]")               # [text]     -> text

RE_BOLD  = re.compile(r"\*([^\*]+)\*")
RE_IT    = re.compile(r"_([^_]+)_")
RE_STRK  = re.compile(r"-([^-]+)-")
RE_INS   = re.compile(r"\+([^+]+)\+")
RE_MONO  = re.compile(r"\{\{([^}]+)\}\}")
RE_COLOR = re.compile(r"\{color[^}]*\}|\{color\}", re.I)
RE_ANCH  = re.compile(r"\{anchor[^}]*\}", re.I)

RE_VER   = re.compile(r"\b(v?\d+(?:\.\d+){1,3})\b", re.I)  # v1.2.3 / 2.0
RE_NUM   = re.compile(r"\b\d+\b")

RE_IE  = re.compile(r"\bi\.\s*e\.", re.I)
RE_EG  = re.compile(r"\be\.\s*g\.", re.I)
RE_ETC = re.compile(r"\betc\.\b", re.I)

KEEP_PUNCT = ".!?;,"  # keep comma for table cell joining
# ---- Minimal change here: keep '-' out of translation, we control it ourselves
TRANS = {ord(ch): " " for ch in string.punctuation if ch not in KEEP_PUNCT + "-"}

RE_ZW  = re.compile(r"[\u200B-\u200D\u2060]")  # zero-widths

# Internal hyphen rule (word-char - word-char)
RE_INTERNAL_HYPHEN = re.compile(r"(?<=\w)-(?!\s)(?=\w)")

# Short-line comma rule threshold
SHORT_LINE_MAX_WORDS = 4

@dataclass
class CleanConfig:
    keep_case: bool = False
    min_line_len_for_period: int = 30

def _normalize(x: str, cfg: CleanConfig) -> str:
    s = "" if x is None else str(x)
    s = fix_text(s)
    s = contractions.fix(s)
    s = html.unescape(s).replace("\u00A0", " ")
    s = RE_ZW.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s if cfg.keep_case else s.lower()

def _strip_blocks(s: str) -> str:
    s = RE_CODE_ATL.sub(" ", s)
    s = RE_CODE_MD.sub(" ", s)
    s = RE_IMAGE.sub(" ", s)
    s = RE_ATTACH.sub(" ", s)
    s = RE_LINK_TXT.sub(r"\1", s)
    s = RE_LINK_BRK.sub(r"\1", s)
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" ", s)
    s = RE_JIRA.sub(" ", s)
    s = RE_HTML.sub(" ", s)
    return s

def _structure_lines(s: str) -> List[str]:
    out = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            out.append("")
            continue
        t = RE_H_WIKI.sub("", t)
        t = RE_H_MD.sub("", t)
        t = RE_BQ.sub("", t)
        if RE_BULLET.match(t) or RE_NUM_LI.match(t):
            t = RE_BULLET.sub("", t)
            t = RE_NUM_LI.sub("", t)
        out.append(t)
    return out

def _tables(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        t = ln.strip()
        if not t:
            out.append("")
            continue
        if RE_T_HDR.match(t):
            continue  # drop header row
        if RE_T_ROW.match(t):
            cells = [c.strip() for c in t.strip().strip("|").split("|") if c.strip()]
            if cells:
                out.append(", ".join(cells) + ".")
            continue
        out.append(ln)
    return out

def _styles(s: str) -> str:
    s = RE_BOLD.sub(r"\1", s)
    s = RE_IT.sub(r"\1", s)
    s = RE_STRK.sub(r"\1", s)
    s = RE_INS.sub(r"\1", s)
    s = RE_MONO.sub(r"\1", s)
    s = RE_COLOR.sub(" ", s)
    s = RE_ANCH.sub(" ", s)
    return s

def _abbr_protect(s: str) -> str:
    s = RE_IE.sub("<ABBR_IE>", s)
    s = RE_EG.sub("<ABBR_EG>", s)
    s = RE_ETC.sub("<ABBR_ETC>", s)
    return s

def _abbr_restore(s: str) -> str:
    return s.replace("<ABBR_IE>", "i.e.").replace("<ABBR_EG>", "e.g.").replace("<ABBR_ETC>", "etc.")

def _apply_internal_hyphen_rule(s: str) -> str:
    # Replace internal hyphen between word chars with a space
    # e.g., "Non-AIM users" -> "Non AIM users"
    return RE_INTERNAL_HYPHEN.sub(" ", s)

def _numbers_and_punct(s: str) -> str:
    kept = {}
    def keep_ver(m):
        k = f"<VER{len(kept)}>"
        kept[k] = m.group(0)
        return k
    s = RE_VER.sub(keep_ver, s)        # protect versions
    s = RE_NUM.sub(" ", s)             # drop plain numbers
    s = s.translate(TRANS)             # keep only .!?;,
    for k, v in kept.items():
        s = s.replace(k, v)            # restore versions
    return s

def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])

def _ac_and_delimit(lines: List[str], min_len: int) -> str:
    # Implements:
    # - "ac\d+:" inline or header -> "acceptance criteria: ..."
    # - Short-line comma rule: if current line has <= SHORT_LINE_MAX_WORDS and next non-empty exists, join with comma
    # - Otherwise, for long lines lacking final punctuation, add a period
    out: List[str] = []
    i = 0
    while i < len(lines):
        t = lines[i].strip()
        if not t:
            i += 1
            continue

        # Inline AC: "ac1: text"
        m_inline = re.match(r"^ac\d*\s*:\s*(.+)$", t, re.I)
        if m_inline:
            out.append(f"acceptance criteria: {m_inline.group(1).strip()}.")
            i += 1
            continue

        # Header AC: "ac1:" on its own -> consume following non-empty lines
        if re.match(r"^ac\d*\s*:\s*$", t, re.I):
            j, parts = i + 1, []
            while j < len(lines) and lines[j].strip():
                parts.append(lines[j].strip())
                j += 1
            if parts:
                out.append(f"acceptance criteria: {' '.join(parts)}.")
            i = j
            continue

        # Short-line comma rule
        # Find next non-empty line (peek)
        j = i + 1
        next_non_empty = None
        while j < len(lines):
            if lines[j].strip():
                next_non_empty = lines[j].strip()
                break
            j += 1

        # Strip trailing colon from header-like short label before comma join
        cur = t[:-1] if t.endswith(":") else t

        if next_non_empty is not None and _word_count(cur) <= SHORT_LINE_MAX_WORDS:
            # Join with comma + space, and replace the next line with the joined result
            merged = f"{cur}, {next_non_empty}"
            lines[j] = merged
            i += 1
            continue

        # Long line: add period if needed
        if len(cur) >= min_len and re.search(r"[A-Za-z0-9)]$", cur) and not re.search(r"[.!?;:]$", cur):
            cur += "."
        out.append(cur)
        i += 1

    # Join with sentence breaks where we already added periods, commas handled above
    s = " ".join(out).strip()
    # Normalize spaces and multi-dots
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(\.\s*){2,}", ". ", s)
    # Fix ': .' artifacts if any remain
    s = re.sub(r":\s*\.", ".", s)
    # Ensure final punctuation
    if s and s[-1] not in ".!?;:":
        s += "."
    return s

def _spacy_sentences(text: str) -> str:
    doc = _NLP(text)
    sents = []
    for s in doc.sents:
        t = s.text.strip()
        if not t:
            continue
        if t[-1] not in ".!?;:":
            t += "."
        sents.append(t)
    return " ".join(sents).strip() if sents else text.strip()

# ---------- Public API ----------
def clean_text(raw_text: str, cfg: CleanConfig = CleanConfig()) -> str:
    s = _normalize(raw_text, cfg)
    s = _strip_blocks(s)
    lines = _structure_lines(s)
    lines = _tables(lines)
    s = _styles("\n".join(lines))
    s = _abbr_protect(s)
    s = _apply_internal_hyphen_rule(s)     # hyphen rule BEFORE punctuation fold
    s = _numbers_and_punct(s)
    s = _abbr_restore(s)
    s = _ac_and_delimit(s.splitlines(), cfg.min_line_len_for_period)
    s = _spacy_sentences(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_df(df: pd.DataFrame, text_col: str, out_col: str = "ISSUE_DESC_STR_CLEANED") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = [clean_text(x) for x in out[text_col].astype(str).tolist()]
    return out
