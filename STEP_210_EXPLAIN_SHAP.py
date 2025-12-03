import pandas as pd
import re
import random

INPUT_FILE = "STEP_201_STAR.csv"
OUTPUT_FILE = "STEP_202_FEEDBACK.csv"

df = pd.read_csv(INPUT_FILE)

feature_map = {
    "avg_sentence_length_tokens": "sentence length",
    "sentence_length_cv": "variation in sentence length",
    "subordination_ratio": "use of complex clauses",
    "coordination_index_cconj_per_clause": "use of conjunctions",
    "lexical_diversity_mattr": "vocabulary diversity",
    "FRE_norm": "readability score",
    "Fog_norm": "reading difficulty",
    "technical_jargon_density": "technical jargon",
    "noun_verb_ratio": "noun-to-verb balance",
    "passive_voice_percent_finite_verbs": "passive voice usage",
    "idea_density_predicates_per_10w": "idea density",
}

groups = {
    "readability": ["FRE_norm","Fog_norm","ARI_norm","LIX_norm","SPACHE_norm","SMOG_norm","FKGL_norm","CLI_norm","DC_norm"],
    "style": ["avg_sentence_length_tokens","sentence_length_cv","coordination_index_cconj_per_clause","subordination_ratio","passive_voice_percent_finite_verbs"],
    "vocabulary": ["lexical_diversity_mattr","technical_jargon_density","noun_verb_ratio","mean_zipf_lemma"],
    "content": ["idea_density_predicates_per_10w","nominalization_density"]
}

TEMPLATES = {
    "positive": [
        "It benefits from strong {desc}.",
        "Good {desc} helps improve clarity.",
        "The {desc} adds depth and quality.",
        "Its {desc} positively influences the overall impression."
    ],
    "negative": [
        "However, {desc} slightly reduces readability.",
        "The {desc} makes the text harder to follow.",
        "A bit too much {desc} hurts clarity.",
        "Its {desc} could be simplified for better flow."
    ],
    "neutral": [
        "{desc} seems balanced and has little overall impact.",
        "No strong effect from {desc} detected."
    ]
}

def interpret_shap(top5_str):
    if not isinstance(top5_str, str) or not top5_str.strip():
        return "No SHAP information available."

    pairs = re.findall(r"([\w_]+): ([+-]?\d+\.\d+)", top5_str)
    if not pairs:
        return "No interpretable SHAP values."

    parts = []
    for feat, val_str in pairs:
        val = float(val_str)
        desc = feature_map.get(feat, feat.replace("_", " "))
        if val > 0.1:
            text = random.choice(TEMPLATES["positive"]).format(desc=desc)
        elif val < -0.1:
            text = random.choice(TEMPLATES["negative"]).format(desc=desc)
        else:
            text = random.choice(TEMPLATES["neutral"]).format(desc=desc)
        parts.append(text)

    summary = " ".join(parts)
    summary = summary.replace("..", ".").strip()
    summary = summary[0].upper() + summary[1:]
    return summary

df["Feedback"] = df["Top5_SHAP_Features"].apply(interpret_shap)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Natural feedback generated → {OUTPUT_FILE}")
