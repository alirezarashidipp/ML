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


AGGREGATION = "max"   # "max" (recommended) or "mean"


# ---------- code ----------
from transformers import pipeline

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"
clf = pipeline("zero-shot-classification", model=MODEL_PATH)

def aggregate(scores, how="max"):
    if not scores:
        return 0.0
    if how == "mean":
        return sum(scores) / len(scores)
    return max(scores)

def who_what_why_scores(text: str):
    out = {}
    for key, label in LABELS.items():
        template_scores = []
        for tpl in TEMPLATES[key]:
            res = clf(
                text,
                [label],                       # single label => cleaner signal
                hypothesis_template=tpl,
                multi_label=True,              # keep independent probability
            )
            template_scores.append(float(res["scores"][0]))
        out[key] = aggregate(template_scores, AGGREGATION)
    return out

# demo
text = "As a developer, I want to deploy the code so that releases are faster."
scores = who_what_why_scores(text)

print("Text:", text)
for k in ["WHO", "WHAT", "WHY"]:
    print(f"{k}: {scores[k]*100:.2f}%")
