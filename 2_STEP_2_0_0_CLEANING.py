# -*- coding: utf-8 -*-
# MVP Cleaning Pipeline for Jira/Confluence-style text
# Features:
#   - Normalize (ftfy/html/contractions)
#   - Strip/Tokenize blocks: code, image, attachment, urls, emails, jira ids
#   - Structure: headings, quotes, bullets -> sentences
#   - Tables: drop header row, join cells with comma, end each row with dot
#   - Acceptance Criteria: ac1: ... -> "acceptance criteria: ..."
#   - Numbers policy: drop plain numbers, keep versions/decimals
#   - Punctuation policy: keep only .!?;
#   - Dot insertion heuristics for dot-less text
#   - Optional spaCy sentencizer
#   - QC meta: sent_count, avg_sent_len, ac_count, added_period, table_rows, blocks_removed, char_reduction_pct

import re, html, string, json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
from ftfy import fix_text
import contractions

# -------- Optional spaCy --------
try:
    import spacy
    _NLP = spacy.blank("en")
    if "sentencizer" not in _NLP.pipe_names:
        _NLP.add_pipe("sentencizer")
    USE_SPACY_DEFAULT = True
except Exception:
    _NLP = None
    USE_SPACY_DEFAULT = False

# -------- Regexes --------
RE_CODE_BLOCK = re.compile(r"\{(code|noformat)[^}]*\}[\s\S]*?\{\1\}", re.IGNORECASE)
RE_MD_CODE    = re.compile(r"```[\s\S]*?```", re.MULTILINE)
RE_IMAGE      = re.compile(r"![^!\n]+!", re.IGNORECASE)            # !pic.png!
RE_ATTACHMENT = re.compile(r"\[\^[^\]]+\]", re.IGNORECASE)         # [^file]
RE_URL        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL      = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_JIRA       = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b", re.IGNORECASE)

RE_HTML_TAG   = re.compile(r"<[^>]+>")
RE_HEADING    = re.compile(r"^h[1-6]\.\s*", re.IGNORECASE)
RE_QUOTE      = re.compile(r"^bq\.\s*", re.IGNORECASE)
RE_BULLET     = re.compile(r"^\s*[\*\-\+\#]+\s+")
RE_NUMBERED   = re.compile(r"^\s*\d+\.\s+")
RE_TABLE_HDR  = re.compile(r"^\s*\|\|")         # header row: || ... ||
RE_TABLE_ROW  = re.compile(r"^\s*\|")           # data row:   | ... |
RE_LINK_TXT   = re.compile(r"\[([^\]\|]+)\|[^\]]+\]")  # [text|url] -> text
RE_LINK_BRK   = re.compile(r"\[([^\]]+)\]")             # [text] -> text

RE_BOLD       = re.compile(r"\*([^\*]+)\*")
RE_ITALIC     = re.compile(r"_([^_]+)_")
RE_STRIKE     = re.compile(r"-([^-]+)-")
RE_INSERT     = re.compile(r"\+([^+]+)\+")
RE_MONO       = re.compile(r"\{\{([^}]+)\}\}")
RE_COLOR      = re.compile(r"\{color[^}]*\}|\{color\}", re.IGNORECASE)
RE_ANCHOR     = re.compile(r"\{anchor[^}]*\}", re.IGNORECASE)

RE_VERSION    = re.compile(r"\b(v?\d+(?:\.\d+){1,3})\b", re.IGNORECASE)  # v1.2.3, 2.0, 1.2
RE_PLAIN_NUM  = re.compile(r"\b\d+\b")

RE_ABBR_IE    = re.compile(r"\bi\.\s*e\.", re.IGNORECASE)
RE_ABBR_EG    = re.compile(r"\be\.\s*g\.", re.IGNORECASE)
RE_ABBR_ETC   = re.compile(r"\betc\.\b", re.IGNORECASE)

# -------- Config --------
@dataclass
class CleanConfig:
    keep_case: bool = False
    table_mode: str = "flatten"          # fixed as requested
    code_mode: str  = "drop"             # "drop" or "token"
    link_mode: str  = "keep_text"        # "drop" or "keep_text"
    numbers_mode: str = "token_version"  # "drop" | "keep" | "token_version"
    min_line_len_for_period: int = 30
    use_spacy_sentencizer: bool = USE_SPACY_DEFAULT
    emit_meta: bool = True

# -------- Helpers --------
def _normalize(text: str, cfg: CleanConfig) -> str:
    if not isinstance(text, str):
        return ""
    text = fix_text(text)
    text = contractions.fix(text)
    text = html.unescape(text)
    text = text.replace("\u00A0", " ")  # no-break space
    text = re.sub(r"[ \t]+", " ", text)
    if not cfg.keep_case:
        text = text.lower()
    return text

def _strip_blocks(text: str, stats: Dict[str, int], cfg: CleanConfig) -> str:
    before = text
    # code blocks
    n1 = len(RE_CODE_BLOCK.findall(text))
    text = RE_CODE_BLOCK.sub(" <CODE_BLOCK> " if cfg.code_mode=="token" else " ", text)
    n2 = len(RE_MD_CODE.findall(text))
    text = RE_MD_CODE.sub(" <CODE_BLOCK> " if cfg.code_mode=="token" else " ", text)
    stats["blocks_code"] += (n1 + n2)

    # media & attachments
    stats["blocks_image"] += len(RE_IMAGE.findall(text))
    text = RE_IMAGE.sub(" ", text)
    stats["blocks_attach"] += len(RE_ATTACHMENT.findall(text))
    text = RE_ATTACHMENT.sub(" ", text)

    # urls/emails/jira
    if cfg.link_mode == "keep_text":
        text = RE_LINK_TXT.sub(r"\1", text)
        text = RE_LINK_BRK.sub(r"\1", text)
    else:
        # drop links entirely
        stats["blocks_links"] += len(RE_LINK_TXT.findall(text)) + len(RE_LINK_BRK.findall(text))
        text = RE_LINK_TXT.sub(" ", text)
        text = RE_LINK_BRK.sub(" ", text)

    stats["blocks_url"] += len(RE_URL.findall(text))
    text = RE_URL.sub(" ", text)
    stats["blocks_email"] += len(RE_EMAIL.findall(text))
    text = RE_EMAIL.sub(" ", text)
    stats["blocks_jira"] += len(RE_JIRA.findall(text))
    text = RE_JIRA.sub(" ", text)

    # html tags
    text = RE_HTML_TAG.sub(" ", text)

    stats["blocks_removed_total"] += len(before) - len(text)
    return text

def _handle_structure_lines(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            out.append("")  # preserve blank markers (used by dotting later)
            i += 1
            continue

        # headings/quotes
        line = RE_HEADING.sub("", line)
        line = RE_QUOTE.sub("", line)

        # bullets & numbered
        if RE_BULLET.match(line) or RE_NUMBERED.match(line):
            line = RE_BULLET.sub("", line)
            line = RE_NUMBERED.sub("", line)

        out.append(line)
        i += 1
    return out

def _process_tables(lines: List[str], stats: Dict[str, int]) -> List[str]:
    # Implements: drop header rows (|| ... ||), data rows (| ... |) -> "cell1, cell2, ... ."
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(s)
            continue
        if RE_TABLE_HDR.match(s):
            # header row -> drop
            stats["table_headers_dropped"] += 1
            continue
        if RE_TABLE_ROW.match(s):
            # data row -> split by |, strip empties
            # remove leading/trailing | and split
            row = s.strip().strip("|")
            cells = [c.strip() for c in row.split("|")]
            cells = [c for c in cells if c]  # remove empty
            if cells:
                stats["table_rows_flattened"] += 1
                out.append(", ".join(cells) + ".")
            continue
        out.append(ln)
    return out

def _apply_styles(text: str) -> str:
    text = RE_BOLD.sub(r"\1", text)
    text = RE_ITALIC.sub(r"\1", text)
    text = RE_STRIKE.sub(r"\1", text)
    text = RE_INSERT.sub(r"\1", text)
    text = RE_MONO.sub(r"\1", text)
    text = RE_COLOR.sub(" ", text)
    text = RE_ANCHOR.sub(" ", text)
    return text

def _numbers_and_punct(text: str, cfg: CleanConfig) -> str:
    # protect versions
    versions = {}
    def _protect(match):
        key = f"<VER{len(versions)}>"
        versions[key] = match.group(0)
        return key
    text = RE_VERSION.sub(_protect, text)

    if cfg.numbers_mode == "drop":
        text = RE_PLAIN_NUM.sub(" ", text)
    elif cfg.numbers_mode == "token_version":
        # drop only plain numbers; tokens for versions restored later
        text = RE_PLAIN_NUM.sub(" ", text)
    # else keep numbers as-is

    # keep only .!?; and turn other punct into space
    keep = ".!?;"
    trans = {ord(ch): " " for ch in string.punctuation if ch not in keep}
    text = text.translate(trans)

    # restore versions
    for k, v in versions.items():
        text = text.replace(k, v)
    return text

def _protect_abbr(text: str) -> Tuple[str, Dict[str, str]]:
    # protect common abbreviations before dot heuristics
    repl = {}
    def _rep(pattern, token):
        nonlocal text
        if pattern.search(text):
            text = pattern.sub(token, text)
            repl[token] = pattern.pattern  # only for bookkeeping
    _rep(RE_ABBR_IE, "<ABBR_IE>")
    _rep(RE_ABBR_EG, "<ABBR_EG>")
    _rep(RE_ABBR_ETC, "<ABBR_ETC>")
    return text, repl

def _restore_abbr(text: str) -> str:
    text = text.replace("<ABBR_IE>", "i.e.")
    text = text.replace("<ABBR_EG>", "e.g.")
    text = text.replace("<ABBR_ETC>", "etc.")
    return text

def _dot_insertion(lines: List[str], cfg: CleanConfig, stats: Dict[str, int]) -> str:
    # acceptance criteria detection: "ac\d*:" header followed by next non-empty line as content
    out_lines: List[str] = []
    i = 0
    ac_count = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            out_lines.append("")
            i += 1
            continue

        m_ac = re.match(r"^ac\d*\s*:\s*$", s, re.IGNORECASE)
        if m_ac:
            # consume following non-empty line(s) until blank
            j = i + 1
            content_parts = []
            while j < len(lines) and lines[j].strip():
                content_parts.append(lines[j].strip())
                j += 1
            content = " ".join(content_parts).strip()
            if content:
                out_lines.append(f"acceptance criteria: {content}.")
                ac_count += 1
            i = j
            continue

        # general line -> add dot if long enough and ends with word/number
        added_period = False
        if len(s) >= cfg.min_line_len_for_period and re.search(r"[A-Za-z0-9)]$", s):
            if not re.search(r"[.!?;:]$", s):
                s = s + "."
                added_period = True
        out_lines.append(s)
        if added_period:
            stats["added_period_lines"] += 1
        i += 1

    stats["ac_count"] += ac_count
    # join and collapse blank lines
    text = "\n".join(out_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    # convert newlines to sentence breaks
    text = re.sub(r"\s*\n\s*", ". ", text)
    # normalize multi-dots
    text = re.sub(r"(\.\s*){2,}", ". ", text)
    # fix ":."
    text = re.sub(r":\s*\.", ".", text)
    # ensure end punctuation
    text = text.strip()
    if text and text[-1] not in ".!?;:":
        text += "."
        stats["added_period_end"] += 1
    return text

def _spacy_sentences(text: str) -> List[str]:
    if not _NLP:
        return [text]
    doc = _NLP(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    # ensure each ends with punctuation
    out = []
    for s in sents:
        if s[-1] not in ".!?;:":
            s = s + "."
        out.append(s)
    return out

def _qc_meta(raw: str, cleaned: str, stats: Dict[str, int], sents: List[str]) -> Dict[str, Any]:
    words = lambda s: [w for w in re.split(r"\s+", s.strip()) if w]
    sent_lens = [len(words(s)) for s in sents] if sents else [0]
    avg_len = (sum(sent_lens) / max(1, len(sent_lens))) if sent_lens else 0.0
    meta = {
        "sent_count": len(sents),
        "avg_sent_len": round(avg_len, 2),
        "ac_count": stats.get("ac_count", 0),
        "added_period_lines": stats.get("added_period_lines", 0),
        "added_period_end": stats.get("added_period_end", 0),
        "table_rows_flattened": stats.get("table_rows_flattened", 0),
        "table_headers_dropped": stats.get("table_headers_dropped", 0),
        "blocks_removed_total_delta_chars": stats.get("blocks_removed_total", 0),
        "char_reduction_pct": round(
            (max(0, len(raw) - len(cleaned)) / max(1, len(raw))) * 100, 1
        ),
        "warnings": []
    }
    if meta["sent_count"] == 1 and meta["avg_sent_len"] >= 50:
        meta["warnings"].append("single_very_long_sentence")
    if meta["char_reduction_pct"] > 70:
        meta["warnings"].append("aggressive_cleaning")
    if len(cleaned.strip()) == 0:
        meta["warnings"].append("empty_output")
    return meta

# -------- Public API --------
def clean_text(raw: str, cfg: Optional[CleanConfig] = None) -> Tuple[str, Dict[str, Any]]:
    cfg = cfg or CleanConfig()
    stats = {
        "blocks_code": 0, "blocks_image": 0, "blocks_attach": 0, "blocks_links": 0,
        "blocks_url": 0, "blocks_email": 0, "blocks_jira": 0,
        "blocks_removed_total": 0,
        "table_headers_dropped": 0, "table_rows_flattened": 0,
        "ac_count": 0, "added_period_lines": 0, "added_period_end": 0,
    }
    raw_in = "" if raw is None else str(raw)

    t = _normalize(raw_in, cfg)
    t = _strip_blocks(t, stats, cfg)
    lines = _handle_structure_lines(t)
    lines = _process_tables(lines, stats)
    t = "\n".join(lines)
    t = _apply_styles(t)
    # abbreviations protection before punctuation/dotting
    t, _ = _protect_abbr(t)
    t = _numbers_and_punct(t, cfg)
    t = _restore_abbr(t)
    # dot insertion & AC handling
    lines2 = [ln for ln in t.splitlines()]
    t = _dot_insertion(lines2, cfg, stats)

    # optional spaCy sentence refine
    sents = _spacy_sentences(t) if cfg.use_spacy_sentencizer else [t]
    final = " ".join(sents).strip()

    meta = _qc_meta(raw_in, final, stats, sents) if cfg.emit_meta else {}
    return final, meta

def clean_df(df: pd.DataFrame, text_col: str, cfg: Optional[CleanConfig] = None) -> pd.DataFrame:
    cfg = cfg or CleanConfig()
    outs, metas = [], []
    for x in df[text_col].tolist():
        ct, meta = clean_text(x, cfg)
        outs.append(ct)
        metas.append(json.dumps(meta, ensure_ascii=False))
    res = df.copy()
    res["clean_text"] = outs
    res["clean_meta"] = metas
    return res

# -------- Example (comment out in production) --------
if __name__ == "__main__":
import pandas as pd

    # Load your CSV
    df = pd.read_csv("input.csv")
    
    # Apply cleaning on column ISSUE_DESC_STR (assume this is your raw text column)
    df["ISSUE_DESC_STR_CLEANED"] = df["ISSUE_DESC_STR"].apply(lambda x: clean_text(x)[0])
    
    # Save to CSV
    df.to_csv("output_cleaned.csv", index=False, encoding="utf-8")
    print("Saved: output_cleaned.csv")

