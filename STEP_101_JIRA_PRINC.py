# ============================================================
# STEP 101 - Embedding Recall Recovery (Production Version)
# Author: Ali
# Purpose:
#   - Preserve spaCy precision (oracle)
#   - Recover false negatives using semantic embeddings
# ============================================================

import pandas as pd
from typing import Dict, List
import spacy
from sentence_transformers import SentenceTransformer, util


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV  = "STEP_100_JIRA_PRINC.csv"
OUTPUT_CSV = "STEP_101_JIRA_PRINC.csv"

TEXT_COL = "ISSUE_DESC_STR_CLEANED"
KEY_COL  = "Key"

TARGET_FLAGS = [
    "has_role_defined",
    "has_goal_defined",
    "has_reason_defined",
]

MAX_SENTENCES = 15

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\all_MiniLM_L6_v2-1.0.0"


# ============================================================
# LOAD MODELS
# ============================================================

print("🔹 Loading embedding model...")
embedder = SentenceTransformer(MODEL_PATH)

print("🔹 Loading spaCy sentencizer...")
if not spacy.util.is_package("en_core_web_md"):
    sent_nlp = spacy.blank("en")
    sent_nlp.add_pipe("sentencizer")
else:
    sent_nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
    if "sentencizer" not in sent_nlp.pipe_names:
        sent_nlp.add_pipe("sentencizer")


# ============================================================
# EMBEDDING PROTOTYPES
# ============================================================

EMB_PROTOTYPES = {

    # -------- ROLE (WHO) --------
    "has_role_defined": [
        "As a user, I want to",
        "As an administrator, I want to",
        "As a developer, I need to",
        "As a product owner, I want to",
        "As a support agent, I need to",
        "As a system architect, we need",
        "As a stakeholder, I want to see",

        "The end-user should have the ability to",
        "External clients must be able to",
        "The API consumer needs a way to",
        "The mobile app user wants to",

        "Users with read-only access should",
        "Privileged users can perform",
        "Authorized personnel must be able to",
        "Unauthenticated visitors should be redirected",

        "The system needs to",
        "The backend service should",
        "The database administrator needs to",
        "The automated script must be able to",
    ],

    # -------- GOAL (WHAT) --------
    "has_goal_defined": [
        "I want to perform an action",
        "I need to be able to",
        "Allow the user to",
        "Enable the functionality of",
        "Provide a mechanism for",
        "The system shall provide the capability to",

        "Create a new record in the system",
        "Update the existing configuration",
        "Delete obsolete data from the logs",
        "View and export report summaries",
        "Search for specific entries in the database",

        "Implement a new endpoint for",
        "Integrate with the third-party service",
        "Automate the process of synchronization",
        "Enhance the validation logic for",
        "The application should support multi-factor authentication",

        "Migrate data from the old platform",
        "Fix the bug related to the UI",
        "Refactor the legacy code in the module",
        "Optimize the performance of the query",
    ],

    # -------- REASON (WHY) --------
    "has_reason_defined": [
        "So that I can achieve a benefit",
        "So that the workflow is not interrupted",
        "In order to improve efficiency and speed",
        "To ensure better user experience",
        "With the aim of increasing productivity",

        "To reduce manual work and human error",
        "To mitigate security risks and vulnerabilities",
        "To prevent data leakage during transfer",
        "For compliance with GDPR and privacy laws",
        "In order to satisfy audit requirements",
        "To meet the regulatory standards of the industry",

        "To reduce operational costs",
        "To provide better insights for decision making",
        "To increase system reliability and uptime",
        "To maintain backward compatibility",
        "This is required for the upcoming release",
        "To avoid customer dissatisfaction and churn",
        "Because the current process is too slow",
    ],
}


# ============================================================
# THRESHOLDS (threshold, margin)
# ============================================================

EMB_THR = {
    "has_role_defined":   (0.40, 0.01),
    "has_goal_defined":   (0.40, 0.01),
    "has_reason_defined": (0.40, 0.01),
}


# ============================================================
# PRECOMPUTE PROTOTYPE EMBEDDINGS
# ============================================================

print("🔹 Encoding prototype embeddings...")
PROTO_EMB = {
    k: embedder.encode(
        v,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    for k, v in EMB_PROTOTYPES.items()
}


# ============================================================
# HELPERS
# ============================================================

def split_chunks(text: str) -> List[str]:
    """
    Sentence + soft clause chunking.
    Handles '.', ',' and ';' with Jira-style noise.
    """
    if not isinstance(text, str):
        return []

    t = text.strip().replace(";", ". ")
    if not t:
        return []

    doc = sent_nlp(t)
    chunks = []

    for s in doc.sents:
        sent = s.text.strip()
        if len(sent) < 5:
            continue

        if len(sent) > 180:
            parts = sent.split(",")
            chunks.extend(
                p.strip() for p in parts if len(p.strip()) > 10
            )
        else:
            chunks.append(sent)

    return chunks[:MAX_SENTENCES]


def embedding_upgrade_flags(text: str, flags: Dict[str, int]) -> Dict[str, int]:
    """
    Semantic recall recovery.
    spaCy positives are NEVER overridden.
    """

    missing = [k for k in TARGET_FLAGS if int(flags.get(k, 0)) == 0]
    if not missing:
        return flags

    chunks = split_chunks(text)
    if not chunks:
        return flags

    chunk_emb = embedder.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    for k in missing:
        thr, margin = EMB_THR[k]
        sim = util.cos_sim(chunk_emb, PROTO_EMB[k])

        flat = sim.flatten()
        best = float(flat.max().item())

        if flat.numel() > 1:
            top2 = flat.topk(2).values
            second = float(top2[1])
        else:
            second = 0.0

        if best >= thr and (best - second) >= margin:
            flags[k] = 1

    return flags


# ============================================================
# PIPELINE
# ============================================================

def step_101_embedding():
    print(f"📂 Reading input: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    for col in [KEY_COL, TEXT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    for f in TARGET_FLAGS:
        if f not in df.columns:
            df[f] = 0

    before = {f: int(df[f].sum()) for f in TARGET_FLAGS}

    print("🚀 Running embedding recall recovery...")

    def _process(row):
        flags = {f: int(row.get(f, 0)) for f in TARGET_FLAGS}

        if all(flags.values()):
            return row

        upgraded = embedding_upgrade_flags(row[TEXT_COL], flags)

        for f in TARGET_FLAGS:
            row[f] = int(upgraded[f])

        return row

    df_out = df.apply(_process, axis=1)

    after = {f: int(df_out[f].sum()) for f in TARGET_FLAGS}
    delta = {f: after[f] - before[f] for f in TARGET_FLAGS}

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("-" * 40)
    print(f"✅ Saved {len(df_out)} rows → {OUTPUT_CSV}")
    print("📊 Recovery summary:")
    for f in TARGET_FLAGS:
        print(f" - {f}: +{delta[f]} (total = {after[f]})")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    step_101_embedding()
