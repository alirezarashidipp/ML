import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# For BART-MNLI: typically label mapping is 0=contradiction, 1=neutral, 2=entailment
ENTAIL_ID = 2

def split_sents(text: str):
    # ساده و کافی برای Jira (می‌تونی بعداً با spaCy بهترش کنی)
    s = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [x.strip() for x in s if x.strip()]

@torch.no_grad()
def entail_prob(premise: str, hypothesis: str) -> float:
    inp = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    logits = model(**inp).logits[0]
    probs = F.softmax(logits, dim=-1)
    return float(probs[ENTAIL_ID].cpu())

def best_entail_over_sentences(text: str, hypothesis: str) -> float:
    sents = split_sents(text)
    if not sents:
        return entail_prob(text, hypothesis)
    return max(entail_prob(s, hypothesis) for s in sents)

HYP = {
    "WHO": "The text explicitly specifies an actor or role (WHO) responsible for the request.",
    "WHAT": "The text explicitly states a desired action, capability, or goal (WHAT).",
    "WHY": "The text explicitly explains a reason, benefit, or value (WHY).",
}

def detect_who_what_why(text: str, thr_who=0.55, thr_what=0.55, thr_why=0.55):
    scores = {k: best_entail_over_sentences(text, h) for k, h in HYP.items()}
    flags = {
        "has_WHO": scores["WHO"] >= thr_who,
        "has_WHAT": scores["WHAT"] >= thr_what,
        "has_WHY": scores["WHY"] >= thr_why,
    }
    return scores, flags

text = "As a developer, I want to deploy the code so that releases are faster."
scores, flags = detect_who_what_why(text)

print("Text:", text)
print("Scores:", {k: f"{v:.2%}" for k,v in scores.items()})
print("Flags:", flags)
