# ---------- config ----------
LABELS = {
    "WHO":  "an explicit actor/role (WHO)",
    "WHAT": "an explicit goal/action (WHAT)",
    "WHY":  "an explicit reason/value (WHY)",
}

TEMPLATES = {
    "WHO": [
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        # more context (boost WHO precision/recall)
        "The story has a role/actor mentioned such as user/admin/developer/system: {}.",
        "An actor is specified (e.g., 'As a ...', 'User should be able to ...'): {}.",
    ],
    "WHAT": [
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        # more context (boost WHAT)
        "A concrete requested capability or action is stated (e.g., 'I want to ...', 'System should ...'): {}.",
        "The story contains a goal or required functionality: {}.",
    ],
    "WHY": [
        "This Jira user story explicitly states {}.",
        "The text clearly includes {}.",
        "There is {} in the story.",
        # more context (boost WHY)
        "A reason or business value is stated using cues like 'so that', 'in order to', 'because': {}.",
        "The story explains the benefit/outcome/value of the request: {}.",
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
