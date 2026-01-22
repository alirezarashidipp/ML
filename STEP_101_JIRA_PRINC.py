# ============================================================
# STEP 101 - Embedding fallback (Refined by Ali & AI)
# ============================================================

import pandas as pd
from typing import Dict, List
import spacy
from sentence_transformers import SentenceTransformer, util

# ---------------- Config ----------------
INPUT_CSV  = "STEP_100_JIRA_PRINC.csv"
OUTPUT_CSV = "STEP_101_JIRA_PRINC.csv"
TEXT_COL   = "ISSUE_DESC_STR_CLEANED"
KEY_COL    = "Key"

TARGET_FLAGS = ["has_role_defined", "has_goal_defined", "has_reason_defined"]
MAX_SENTENCES = 15

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\all_MiniLM_L6_v2-1.0.0"
# Load model once
embedder = SentenceTransformer(MODEL_PATH)

# Setup Spacy
if not spacy.util.is_package("en_core_web_md"):
    sent_nlp = spacy.blank("en")
    sent_nlp.add_pipe("sentencizer")
else:
    sent_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    if "sentencizer" not in sent_nlp.pipe_names:
        sent_nlp.add_pipe("sentencizer")


# ---------------- Prototypes ----------------
EMB_PROTOTYPES = {
    # ---------------- ROLE (WHO) ----------------
    "has_role_defined": [
        # classic user story
        "As a user, I want to",
        "As an administrator, I want to",
        "As a customer, I want to",
        "As a developer, I want to",
        "As a product owner, I want to",
        "As a support agent, I want to",

        # variants people write in Jira
        "As an admin, I need to",
        "As a manager, I need to",
        "As an analyst, I need to",
        "As an engineer, I need to",
        "As a tester, I need to",

        # implicit-role formulations (often appears without 'As a')
        "Admin should be able to",
        "Users should be able to",
        "A user should be able to",
        "The administrator should be able to",
        "Customer should be able to",
        "Developers should be able to",
    ],

    # ---------------- GOAL (WHAT) ----------------
    "has_goal_defined": [
        # classic goal
        "I want to perform an action",
        "I need to perform an action",
        "I want to be able to do something",
        "I need to be able to do something",

        # system/requirement style
        "The system should allow the user to perform an action",
        "The system shall allow the user to perform an action",
        "The system should support the ability to perform an action",
        "The application should allow users to perform an action",
        "Allow the user to perform an action",
        "Enable the user to perform an action",

        # common Jira phrasing
        "Please add the ability to perform an action",
        "Implement functionality to perform an action",
        "Provide an option to perform an action",
        "Users can perform an action",
        "Users should be able to perform an action",

        # very common action intents
        "I want to log in",
        "I want to login",
        "I want to reset my password",
        "I want to update my profile",
        "I want to view my account details",
        "I want to download a report",
    ],

    # ---------------- REASON (WHY / VALUE) ----------------
    "has_reason_defined": [
        # classic user story reason
        "So that I can achieve a benefit",
        "So that we can achieve a benefit",
        "So that the user can achieve a benefit",

        # common variants
        "In order to achieve a benefit",
        "In order to reduce risk",
        "In order to improve efficiency",
        "To improve user experience",
        "To reduce manual work",
        "To prevent errors",

        # enterprise drivers
        "For compliance reasons",
        "For audit purposes",
        "To meet regulatory requirements",
        "To reduce incidents",
        "To improve security",
        "To increase reliability",
    ],
}


EMB_THR = {
    "has_role_defined": 0.40,
    "has_goal_defined": 0.40,
    "has_reason_defined": 0.40,
}

# Precompute Prototypes
PROTO_EMB = {k: embedder.encode(v, convert_to_tensor=True) for k, v in EMB_PROTOTYPES.items()}


# ---------------- Helpers ----------------
def split_chunks(text: str) -> List[str]:
    """
    Sentence chunking handling '.' and ';'
    """
    if not isinstance(text, str):
        return []
    
    t = text.strip()
    if not t:
        return []

    # FIX: Replace semicolon with dot so Spacy treats it as a break
    t = t.replace(';', '. ') 
    
    doc = sent_nlp(t)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 3] # Skip tiny noise

    if not sents:
        return [t]

    return sents[:MAX_SENTENCES]


def embedding_upgrade_flags(text: str, flags: Dict[str, int]) -> Dict[str, int]:
    missing = [k for k in TARGET_FLAGS if int(flags.get(k, 0)) == 0]
    
    # Optimization: If no flags are missing, return immediately
    if not missing:
        return flags

    chunks = split_chunks(text)
    if not chunks:
        return flags

    # Encode locally (Small batch of 15 items)
    # This is still not "Global Batch", but handles the 15 sentences in parallel
    chunk_emb = embedder.encode(chunks, convert_to_tensor=True)

    for k in missing:
        sim = util.cos_sim(chunk_emb, PROTO_EMB[k])
        # Find max similarity across all chunks vs all prototypes
        best = float(sim.max().item())
        
        if best >= EMB_THR[k]:
            flags[k] = 1

    return flags


# ---------------- Pipeline ----------------
def step_101_embedding():
    print(f"📂 Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Validation
    for col in [KEY_COL, TEXT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    for f in TARGET_FLAGS:
        if f not in df.columns:
            df[f] = 0

    before = {f: int(df[f].sum()) for f in TARGET_FLAGS}

    print("🚀 Running Embedding Analysis (Row-by-Row)...")
    
    # Using a simple progress indicator if possible, otherwise standard apply
    # We use a wrapper to handle the apply logic cleanly
    def _process(row):
        flags = {f: int(row.get(f, 0)) for f in TARGET_FLAGS}
        # Only run embedding if strictly necessary
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

    print("-" * 30)
    print(f"✅ Saved {len(df_out)} rows to {OUTPUT_CSV}")
    print("📌 Results:")
    for f in TARGET_FLAGS:
        print(f" - {f}: +{delta[f]} (Total: {after[f]})")

if __name__ == "__main__":
    step_101_embedding()
