from transformers import pipeline

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"
WHO_THRESHOLD = 0.55

# ---------------- LOAD MODEL (ONCE) ----------------
clf = pipeline(
    "zero-shot-classification",
    model=MODEL_PATH,
    truncation=True,
)

# ---------------- CORE FUNCTION ----------------
def detect_who(text: str) -> dict:
    """
    Returns:
      {
        "who_score": float (0..1),
        "has_who": bool
      }
    """
    if not isinstance(text, str) or not text.strip():
        return {"who_score": 0.0, "has_who": False}

    label = "an explicit actor/role (WHO)"
    template = "This Jira user story explicitly states {}."

    res = clf(
        text,
        [label],
        hypothesis_template=template,
        multi_label=True,
    )

    score = float(res["scores"][0])
    return {"who_score": score, "has_who": score >= WHO_THRESHOLD}


# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    t = "As a developer, I want to deploy the code so that releases are faster."
    out = detect_who(t)
    print(out)
