

import pandas as pd
import numpy as np
import spacy
from functools import lru_cache
from wordfreq import zipf_frequency

# -------- paths & model --------
INPUT_CSV  = "STEP6_EMPTY_SEP_AFTER_AC_SEP.csv"
OUTPUT_CSV = "STEP_7_LANG_ANALYSIS.csv" 
SPACY_MODEL = "en_core_web_sm_qbf"     

# -------- small helpers (kept minimal) --------
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def finite_verb(tok):
    if tok.pos_ not in {"VERB", "AUX"}:
        return False
    vf = tok.morph.get("VerbForm")
    return bool(vf and "Fin" in vf)

def predicate_adj_root_with_copula(tok):
    if tok.pos_ != "ADJ" or tok.dep_ != "ROOT":
        return False
    return any(ch.dep_ == "cop" and ch.lemma_.lower() == "be" for ch in tok.children)

@lru_cache(maxsize=200_000)
def zipf_en(lemma):
    return zipf_frequency(lemma, "en")

def mattr(tokens, window=50):
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    acc = []
    for i in range(0, n - window + 1):
        win = tokens[i:i+window]
        acc.append(len(set(win)) / window)
    return float(np.mean(acc)) if acc else 0.0

# constants for feature logic
NOMINAL_SUFFIXES = ("tion","sion","ment","ness","ance","ence","ity","ship","acy","ure","ism")
SUBORD_DEPS = {"advcl","ccomp","acl","relcl"}
NEG_LEMMAS  = {"no","not","never","neither","nor"}
HEDGES      = {"may","might","could","should","would","perhaps","possibly","likely","seem","appears"}

# -------- core feature extraction (single compact function) --------
def extract_15_features(doc):
    # sentences that have at least one alphabetic token
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    num_sents = len(sents)

    # 1,2) sentence length mean & CV (alpha-only)
    slens = [sum(1 for t in s if t.is_alpha) for s in sents]
    avg_sent_len = float(np.mean(slens)) if slens else 0.0
    std_sent_len = float(np.std(slens)) if slens else 0.0
    sent_len_cv  = safe_div(std_sent_len, avg_sent_len)

    # 3) clause density ~ finite verbs per sentence
    fins_total = 0
    for s in sents:
        fins_total += sum(1 for t in s if finite_verb(t))
    clauses_per_sent = safe_div(fins_total, num_sents)

    # 4) subordination ratio (#subordinate deps / total clauses)
    subordinate_count = sum(1 for t in doc if t.dep_ in SUBORD_DEPS)
    subordination_ratio = safe_div(subordinate_count, fins_total)

    # 5) coordination index (CCONJ per clause)
    cconj_count = sum(1 for t in doc if t.pos_ == "CCONJ")
    coordination_index = safe_div(cconj_count, fins_total)

    # 6) avg dependency distance (absolute index gap to head, per token)
    dep_dists = []
    for s in sents:
        for t in s:
            if t.head is not None and t.head != t:
                dep_dists.append(abs(t.i - t.head.i))
    avg_dep_dist = float(np.mean(dep_dists)) if dep_dists else 0.0

    # 7) NP mean length (noun_chunks length in tokens)
    np_lens = [len(ch) for ch in doc.noun_chunks]
    np_mean_len = float(np.mean(np_lens)) if np_lens else 0.0

    # alpha tokens & lemmas
    alpha_toks = [t for t in doc if t.is_alpha]
    token_count_alpha = len(alpha_toks)
    lemmas = [t.lemma_.lower() for t in alpha_toks]

    # 8) nominalization density (suffix heuristic on NOUN lemmas)
    nominal_nouns = sum(1 for t in doc if t.pos_ == "NOUN"
                        and any(t.lemma_.lower().endswith(s) for s in NOMINAL_SUFFIXES))
    nominalization_density = safe_div(nominal_nouns, max(1, token_count_alpha))

    # 9) noun:verb ratio
    noun_count = sum(1 for t in doc if t.pos_ in {"NOUN","PROPN"})
    verb_count = sum(1 for t in doc if t.pos_ == "VERB")
    noun_verb_ratio = safe_div(noun_count, verb_count)

    # 10) passive voice % among finite verbs (auxpass or nsubjpass)
    passive_heads = 0
    finite_total  = 0
    for s in sents:
        fins = [t for t in s if finite_verb(t)]
        finite_total += len(fins)
        heads = {t.head for t in s if t.dep_ == "nsubjpass" and t.head in fins}
        for v in fins:
            if any(ch.dep_ == "auxpass" for ch in v.children):
                heads.add(v)
        passive_heads += len(heads)
    passive_voice_pct = 100.0 * safe_div(passive_heads, finite_total)

    # 11) mean Zipf frequency over lemmas
    zipfs = [zipf_en(lem) for lem in lemmas] if lemmas else []
    mean_zipf = float(np.mean(zipfs)) if zipfs else 0.0

    # 12) lexical diversity (MATTR) over lemmas
    lex_div_mattr = float(mattr(lemmas, window=50)) if lemmas else 0.0

    # 13) technical/jargon density (fallback: ratio of lemmas with Zipf < 3.0)
    technical_density = safe_div(sum(1 for z in zipfs if z < 3.0), len(zipfs)) if zipfs else 0.0

    # 14) negation density per sentence (neg dep or simple hedge list)
    neg_tokens = 0
    for s in sents:
        for t in s:
            if t.dep_ == "neg" or t.lemma_.lower() in NEG_LEMMAS or t.lemma_.lower() in HEDGES:
                neg_tokens += 1
    negation_density_per_sentence = safe_div(neg_tokens, num_sents)

    # 15) idea density per 10 words (verbs + predicate ADJ with copula)
    preds = sum(1 for t in doc if t.pos_ == "VERB")
    preds += sum(1 for t in doc if predicate_adj_root_with_copula(t))
    idea_density_per_10w = 10.0 * safe_div(preds, token_count_alpha)

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

# -------- main (super simple) --------
def main():
    df = pd.read_csv(INPUT_CSV, usecols=["Key", "ISSUE_DESC_STR_CLEANED"])
    nlp = spacy.load(SPACY_MODEL)

    feats = []
    for doc in nlp.pipe(df["DESC"].fillna("").astype(str).tolist(), batch_size=64):
        feats.append(extract_15_features(doc))

    out = pd.DataFrame(feats)
    out.insert(0, "Key", df["Key"].tolist())
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  rows={len(out)}  cols={len(out.columns)}")

if __name__ == "__main__":
    main()
