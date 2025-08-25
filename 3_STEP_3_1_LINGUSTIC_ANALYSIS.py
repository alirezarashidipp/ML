from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordfreq import zipf_frequency

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

# -----------------------------
# Helpers
# -----------------------------

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

def is_alpha_token(t: Token) -> bool:
    return (not t.is_space) and (not t.is_punct) and t.text.strip() != ""

def finite_verb(t: Token) -> bool:
    # Finite VERB/AUX (VerbForm=Fin)
    if t.pos_ not in {"VERB", "AUX"}:
        return False
    vform = t.morph.get("VerbForm")
    return ("Fin" in vform) if vform else False

def predicate_adjective_root_with_copula(t: Token) -> bool:
    # e.g., "System is unstable." -> ADJ with copula child
    if t.pos_ != "ADJ":
        return False
    for child in t.children:
        if child.dep_ == "cop" and child.lemma_.lower() == "be":
            return True
    return False

def count_passive_finite_verbs(sent: Span) -> Tuple[int, int]:
    """
    Count passive finite verbs using common patterns:
      - verb with a child dep_ == 'auxpass'
      - sentence has 'nsubjpass' attached to a verb
    Returns (passive_count, finite_verb_count)
    """
    finite_verbs = [t for t in sent if finite_verb(t)]
    passive = set()
    for t in sent:
        if t.dep_ == "nsubjpass" and t.head in finite_verbs:
            passive.add(t.head)
    for v in finite_verbs:
        for ch in v.children:
            if ch.dep_ == "auxpass":
                passive.add(v)
                break
    return len(passive), len(finite_verbs)

def dependency_distance_sent(sent: Span) -> List[int]:
    dists = []
    for t in sent:
        if t.head is not None and t.head != t:
            dists.append(abs(t.i - t.head.i))
    return dists

def noun_chunks_lengths(doc: Doc) -> List[int]:
    return [len(chunk) for chunk in doc.noun_chunks]

NOMINAL_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ance", "ence", "ity", "ship", "acy", "ure", "ism"
)

SUBORDINATE_DEPS: Set[str] = {"advcl", "ccomp", "acl", "relcl"}
NEGATION_LEMMAS: Set[str] = {"no", "not", "never", "neither", "nor"}

def load_jargon_lexicon(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    terms = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                terms.add(w)
    return terms or None

def mattr(tokens: List[str], window: int = 50) -> float:
    """
    Moving Average Type-Token Ratio (Covington & McFall).
    If len(tokens) < window: fallback to TTR for whole doc.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    scores = []
    for i in range(0, n - window + 1):
        win = tokens[i:i + window]
        scores.append(len(set(win)) / window)
    return float(np.mean(scores)) if scores else 0.0

# -----------------------------
# Per-document feature extraction
# -----------------------------

def extract_scalar_features(
    doc: Doc,
    jargon_lexicon: Optional[Set[str]] = None,
    mattr_window: int = 50,
) -> dict:
    sents: List[Span] = [s for s in doc.sents if any(is_alpha_token(t) for t in s)]
    num_sents = len(sents)

    # Sentence lengths (tokens, alpha + punct excluded)
    sent_lengths = [sum(1 for t in s if is_alpha_token(t)) for s in sents]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    std_sent_len = float(np.std(sent_lengths, ddof=0)) if sent_lengths else 0.0
    sent_len_cv = safe_div(std_sent_len, avg_sent_len)

    # Clauses approximated by finite verb count
    finite_verbs_per_sent = [sum(1 for t in s if finite_verb(t)) for s in sents]
    total_clauses = sum(finite_verbs_per_sent)
    clauses_per_sent = safe_div(total_clauses, num_sents)

    # Subordination: subordinate clause heads by dep labels
    subordinate_count = sum(1 for t in doc if t.dep_ in SUBORDINATE_DEPS)
    subordination_ratio = safe_div(subordinate_count, total_clauses)

    # Coordination index: CCONJ per clause
    cconj_count = sum(1 for t in doc if t.pos_ == "CCONJ")
    coordination_index = safe_div(cconj_count, total_clauses)

    # Average dependency distance
    dep_dists = []
    for s in sents:
        dep_dists.extend(dependency_distance_sent(s))
    avg_dep_dist = float(np.mean(dep_dists)) if dep_dists else 0.0

    # NP mean length
    np_lens = noun_chunks_lengths(doc)
    np_mean_len = float(np.mean(np_lens)) if np_lens else 0.0

    # Nominalization density (suffix heuristic on NOUN lemmas)
    lemmas = [t.lemma_.lower() for t in doc if is_alpha_token(t)]
    nominal_nouns = sum(
        1 for t in doc
        if t.pos_ == "NOUN"
        and any(t.lemma_.lower().endswith(suf) for suf in NOMINAL_SUFFIXES)
    )
    nominalization_density = safe_div(nominal_nouns, max(1, len([t for t in doc if is_alpha_token(t)])))

    # NOUN:VERB ratio
    noun_count = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    noun_verb_ratio = safe_div(noun_count, verb_count)

    # Passive-voice % (finite verbs)
    passive_count = 0
    finite_total = 0
    for s in sents:
        p, f = count_passive_finite_verbs(s)
        passive_count += p
        finite_total += f
    passive_voice_pct = 100.0 * safe_div(passive_count, finite_total)

    # Mean Zipf frequency (lemmas, alphabetic)
    lemma_tokens = [t.lemma_.lower() for t in doc if is_alpha_token(t)]
    zipfs = [zipf_frequency(lem, "en") for lem in lemma_tokens]
    mean_zipf = float(np.mean(zipfs)) if zipfs else 0.0

    # Lexical diversity (MATTR) over lemma tokens (alpha only)
    lex_div_mattr = float(mattr(lemma_tokens, window=mattr_window)) if lemma_tokens else 0.0

    # Technical/jargon density
    # If a custom lexicon is provided, use it (ratio of lemma tokens in lexicon).
    # Else fallback: proportion of low-frequency lemmas (Zipf < 3.0).
    if lemma_tokens:
        if jargon_lexicon:
            tech_hits = sum(1 for lem in lemma_tokens if lem in jargon_lexicon)
            technical_density = safe_div(tech_hits, len(lemma_tokens))
        else:
            technical_density = safe_div(sum(1 for z in zipfs if z < 3.0), len(zipfs))
    else:
        technical_density = 0.0

    # Negation density (per sentence)
    neg_tokens = 0
    for s in sents:
        for t in s:
            if t.dep_ == "neg" or t.lemma_.lower() in NEGATION_LEMMAS:
                neg_tokens += 1
    negation_density_per_sentence = safe_div(neg_tokens, num_sents)

    # Idea density: predicates per 10 words
    # Approximate predicates := VERB tokens + copular predicate ADJs.
    predicate_count = sum(1 for t in doc if t.pos_ == "VERB")
    predicate_count += sum(1 for t in doc if predicate_adjective_root_with_copula(t))
    token_count_alpha = len([t for t in doc if is_alpha_token(t)])
    idea_density_per_10w = 10.0 * safe_div(predicate_count, token_count_alpha)

    return {
        "avg_sentence_length_tokens": avg_sent_len,
        "sentence_length_cv": sent_len_cv,
        "clause_density_finite_per_sentence": clauses_per_sent,
        "subordination_ratio": subordination_ratio,
        "coordination_index_cconj_per_clause": coordination_index,
        "avg_dependency_distance": avg_dep_dist,
        "np_mean_length_tokens": np_mean_len,
        "nominalization_density": nominalization_density,
        "noun_verb_ratio": noun_verb_ratio,
        "passive_voice_percent_finite_verbs": passive_voice_pct,
        "mean_zipf_lemma": mean_zipf,
        "lexical_diversity_mattr": lex_div_mattr,
        "technical_jargon_density": technical_density,
        "negation_density_per_sentence": negation_density_per_sentence,
        "idea_density_predicates_per_10w": idea_density_per_10w,
    }

def doc_to_lemma_string(doc: Doc) -> str:
    # For TF-IDF: lemma-based, alpha-only, lowercased.
    toks = [t.lemma_.lower() for t in doc if is_alpha_token(t)]
    return " ".join(toks)

# -----------------------------
# Main pipeline
# -----------------------------

def build_nlp(model_name: str) -> Language:
    nlp = spacy.load(model_name, disable=[])
    # Ensure sentence boundaries are set (parsing already does it).
    return nlp

def process(
    df: pd.DataFrame,
    nlp: Language,
    jargon_lexicon: Optional[Set[str]],
    mattr_window: int,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_rows = []
    lemma_docs: List[str] = []
    ids: List = []

    # Fast, memory-safe streaming
    for key_id, text in zip(df["KEY_ID"].tolist(), df["DESC"].fillna("").astype(str).tolist()):
        ids.append(key_id)

    for doc in nlp.pipe(df["DESC"].fillna("").astype(str).tolist(), batch_size=64):
        features = extract_scalar_features(doc, jargon_lexicon=jargon_lexicon, mattr_window=mattr_window)
        feature_rows.append(features)
        lemma_docs.append(doc_to_lemma_string(doc))

    feat_df = pd.DataFrame(feature_rows)
    feat_df.insert(0, "KEY_ID", df["KEY_ID"].tolist())
    return feat_df, lemma_docs

def add_tfidf(
    base_df: pd.DataFrame,
    lemma_docs: List[str],
    max_features: int = 500,
) -> pd.DataFrame:
    # Lemma unigrams + bigrams, english stopwords removed.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        stop_words="english",
        norm="l2",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(lemma_docs)
    vocab = [f"tfidf_{v}" for v in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(X.toarray(), columns=vocab)
    tfidf_df.insert(0, "KEY_ID", base_df["KEY_ID"].tolist())

    merged = base_df.merge(tfidf_df, on="KEY_ID", how="left")
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to input CSV with KEY_ID,DESC")
    ap.add_argument("--output", required=True, type=str, help="Output CSV path")
    ap.add_argument("--spacy-model", default="en_core_web_sm", type=str, help="SpaCy English model")
    ap.add_argument("--tfidf-max-features", default=500, type=int, help="Number of TF-IDF features")
    ap.add_argument("--mattr-window", default=50, type=int, help="MATTR window size")
    ap.add_argument("--jargon-lexicon", default=None, type=str, help="Optional path to custom jargon lexicon (one term per line)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    expected_cols = {"KEY_ID", "DESC"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Input must have columns: {expected_cols}")

    nlp = build_nlp(args.spacy_model)
    jargon_lex = load_jargon_lexicon(args.jargon_lexicon)

    scalar_df, lemma_docs = process(df, nlp, jargon_lex, args.mattr_window)
    full_df = add_tfidf(scalar_df, lemma_docs, max_features=args.tfidf_max_features)

    full_df.to_csv(out_path, index=False)
    print(f"Saved features to: {out_path}  (rows={len(full_df)}, cols={len(full_df.columns)})")

if __name__ == "__main__":
    main()
