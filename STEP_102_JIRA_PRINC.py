# ============================================================
# WHO / WHAT / WHY detection (NLI Zero-Shot)
# - Reusable module
# - Batch (CSV), Real-time (API), Standalone
# ============================================================

from typing import Dict, Tuple, List
from transformers import pipeline



import pandas as pd



# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"

TEXT_COL = "ISSUE_DESC_STR_CLEANED"
KEY_COL  = "Key"

INPUT_CSV  = "STEP_4_LANG_SEP.csv"
OUTPUT_CSV = "STEP_200_JIRA_NLI_WHO_WHAT_WHY.csv"

AGGREGATION = "max"      # "max" | "mean"
RETURN_DEBUG = False     # True for diagnostics
THRESHOLDS = {"WHO": 0.55, "WHAT": 0.55, "WHY": 0.65}


# ============================================================
# LABELS & TEMPLATES
# ============================================================

# ---------- config ----------
LABELS = {
    "WHO":  "an explicit actor/role (WHO)",
    "WHAT": "an explicit goal/action (WHAT)",
    # improved WHY label (slightly clearer separation from WHAT)
    "WHY":  "an explicit reason, business value, or intended outcome (WHY)",
}

TEMPLATES = {
    "WHO": [
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        "The story has a role/actor mentioned such as user/admin/developer/system: {}.",
        "An actor is specified (e.g., 'As a ...', 'User should be able to ...'): {}.",
    ],
    "WHAT": [
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        "A concrete requested capability or action is stated (e.g., 'I want to ...', 'System should ...'): {}.",
        "The story contains a goal or required functionality: {}.",
    ],
    "WHY": [
        # base
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        "A reason or business value is stated using cues like 'so that', 'in order to', 'because': {}.",
        "The story explains the benefit/outcome/value of the request: {}.",

        # --- MORE CONTEXT for VALUE / WHY ---
        "This Jira user story explicitly explains a reason or value (WHY): {}.",
        "The story states the benefit/outcome of the request (WHY): {}.",
        "The text contains an explicit rationale for the action (WHY): {}.",
        "The story includes a 'so that / in order to / because' style justification (WHY): {}.",

        "The story explains what positive outcome will happen if the request is implemented (WHY): {}.",
        "The story states the intended impact or result of the change (WHY): {}.",
        "The story describes the value delivered to a user, customer, or business (WHY): {}.",
        "The story clarifies the purpose of the request beyond the action itself (WHY): {}.",

        "The story mentions value such as faster delivery, time saving, or efficiency (WHY): {}.",
        "The story mentions value such as improved user experience or usability (WHY): {}.",
        "The story mentions value such as reduced risk, fewer errors, or higher safety (WHY): {}.",
        "The story mentions value such as reliability, availability, or performance improvement (WHY): {}.",
        "The story mentions value such as cost reduction or operational savings (WHY): {}.",
        "The story mentions compliance, audit, regulatory, or security justification (WHY): {}.",
        "The story mentions data quality, accuracy, or consistency as the reason (WHY): {}.",

        "The story explains the underlying business objective behind the request (WHY): {}.",
        "The story includes a motivation statement (e.g., 'to reduce...', 'to prevent...', 'to enable...') (WHY): {}.",
        "The story includes a value statement about what is gained or avoided (WHY): {}.",
        "The story includes a justification describing why the change matters (WHY): {}.",

        "The story indicates preventing an issue, avoiding failure, or mitigating a problem as the WHY: {}.",
        "The story explains the reason as reducing manual work, rework, or bottlenecks (WHY): {}.",
        "The story explains the reason as improving traceability, transparency, or monitoring (WHY): {}.",
    ],
}


# ============================================================
# LOAD MODEL (ONCE)
# ============================================================

print("🔹 Loading NLI model...")
clf = pipeline(
    "zero-shot-classification",
    model=MODEL_PATH,
    truncation=True,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _aggregate(scores: List[float], how: str = "max") -> float:
    if not scores:
        return 0.0
    return max(scores) if how == "max" else sum(scores) / len(scores)



def _score_dimension(text: str, dim: str) -> Tuple[float, List[Tuple[str, float]]]:
    label = LABELS[dim]
    scores = []
    debug = []

    for tpl in TEMPLATES[dim]:
        res = clf(
            text,
            [label],
            hypothesis_template=tpl,
            multi_label=True,
        )
        s = float(res["scores"][0])
        scores.append(s)
        if RETURN_DEBUG:
            debug.append((tpl, s))

    final_score = _aggregate(scores, AGGREGATION)
    return final_score, debug


# ============================================================
# ✅ CORE BUSINESS FUNCTION (REAL-TIME SAFE)
# ============================================================

def detect_who_what_why(text: str) -> Dict[str, object]:



    """
    Core reusable logic:
    - No file IO
    - Safe for FastAPI / services





    """

    if not isinstance(text, str) or not text.strip():
        return {
            "scores": {"WHO": 0.0, "WHAT": 0.0, "WHY": 0.0},
            "flags":  {"WHO": False, "WHAT": False, "WHY": False},
            "debug":  {},

        }

    scores = {}
    flags = {}
    debug = {}










    for dim in ["WHO", "WHAT", "WHY"]:
        s, d = _score_dimension(text, dim)
        scores[dim] = s
        flags[dim] = s >= THRESHOLDS[dim]

        if RETURN_DEBUG:
            debug[dim] = sorted(d, key=lambda x: x[1], reverse=True)[:5]

    return {
        "scores": scores,
        "flags": flags,
        "debug": debug if RETURN_DEBUG else None,

    }


# ============================================================
# ✅ BATCH PIPELINE (CSV → CSV)
# ============================================================

def process_jira_nli_csv(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,



) -> pd.DataFrame:

    df = pd.read_csv(input_csv)

    results = df[TEXT_COL].apply(detect_who_what_why)

    scores_df = pd.DataFrame([r["scores"] for r in results])
    flags_df  = pd.DataFrame([r["flags"]  for r in results])

    scores_df = scores_df.add_prefix("nli_score_")
    flags_df  = flags_df.add_prefix("has_")

    df_out = pd.concat(
        [df[[KEY_COL, TEXT_COL]], scores_df, flags_df],
        axis=1
    )

    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df_out)} rows to {output_csv}")


    return df_out




# ============================================================
# ✅ STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    text = "As a developer, I want to deploy the code so that releases are faster."

    out = detect_who_what_why(text)

    print("Text:", text)
    for k in ["WHO", "WHAT", "WHY"]:
        print(f"{k}: {out['scores'][k]*100:.2f}% | flag={out['flags'][k]}")

    # Uncomment for batch run
    # process_jira_nli_csv()
