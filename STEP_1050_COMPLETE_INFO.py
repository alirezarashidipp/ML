"""
MVP: Local Llama-3.2-1B-Instruct → Jira Story → Structured JSON
- CPU-only
- Deterministic (repeatable)
- Single-pass read of story description
- Minimal code (no extra ifs/loops)
- Transformers 4.49.0 | Torch 2.7.0 | Python 3.9 | tokenizers = 0.21.0 | langgraph = 0.2.62

# model: meta-llama/Llama-3.2-1B-Instruct 
from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch 
path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA" 
try: 
   print("Step 1: Loading Tokenizer...") 
   tokenizer = AutoTokenizer.from_pretrained(path) 
   print("Step 2: Loading Model (This may take a while)...") 
   model = AutoModelForCausalLM.from_pretrained(path)
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- Runtime Switch ----------------
USE_LOCAL_MODEL = True  # True = Local Llama | False = API (stub)

# ---------------- Config ----------------
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"
MAX_NEW_TOKENS = 560

# ---------------- Load (Local only) ----------------
if USE_LOCAL_MODEL:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()


def build_prompt(story: str) -> str:
    return f"""
You are a strict JSON generator for Jira story intelligence.
Return ONLY valid JSON. No markdown. No extra text.

GENERAL RULES:
- If something is NOT explicitly stated, set it to null and set is_defined=false.
- confidence is a self-rated confidence in [0.0, 1.0] with ONE decimal (e.g., 0.7).
- Keep extracted text short and grounded in the story.
- Never invent technologies, systems, or constraints that are not hinted in the text.

ALLOWED ENUMS:
- intent.type: "Create" | "Modify" | "Remove" | "Migrate" | "Integrate" | "Investigate" | "Enforce" | null
- value.category: "Customer" | "Cost" | "Risk" | "Compliance" | "Internal Efficiency" | null
- value.direct_customer_impact: "No" | "Low" | "Medium" | "High"
- scope.shape: "Needle" | "Balloon" | "Fog" | "Brick" | null
- execution_risk.level: "Low" | "Medium" | "High"
- priorities: subset of
  ["Drive customer-centricity", "Deliver focused sustainable growth", "Be simple and agile"]

TECH/SKILLS CONSTRAINTS:
- delivery_signals.tech_stack_indicators: up to 3 items (e.g., "Frontend", "Backend", "Database", "API", "Python", "React")
- delivery_signals.skills_required_top3: up to 3 items (e.g., "Python", "SQL", "React")

DELIVERY CLARIFICATION QUESTIONS (VERY IMPORTANT):
- Generate EXACTLY 2 questions.
- Questions MUST be about DELIVERY/CONTENT CLARIFICATIONS, NOT about identifying WHAT or WHY.
- DO NOT ask: "What is the goal?" / "What is requested?" / "Why is this needed?" / "Can you clarify?" / "Provide more details".
- Each question must focus on ONE of these areas:
  (1) Acceptance criteria / testability
  (2) Scope boundaries (in-scope / out-of-scope)
  (3) Dependencies / integration specifics (which system/API/data)
  (4) Edge cases / failure scenarios / rollback
  (5) Non-functional requirements (security, performance, logging, audit)
- Questions must be specific and verifiable, referencing story terms when possible.
- If the story already includes clear measurable AC + clear scope + clear dependencies, then ask about failure scenarios and audit/logging.

Now analyze the story:

STORY:
\"\"\"{story}\"\"\"

Return JSON with EXACT structure:

{{
  "ownership": {{
    "is_defined": false,
    "confidence": 0.0,
    "owner_or_actor": null
  }},
  "intent": {{
    "is_defined": false,
    "confidence": 0.0,
    "primary_intent": null,
    "type": null
  }},
  "value": {{
    "is_defined": false,
    "confidence": 0.0,
    "category": null,
    "direct_customer_impact": "No"
  }},
  "delivery_clarifications": {{
    "top2_questions": ["", ""]
  }},
  "scope": {{
    "shape": null
  }},
  "execution_risk": {{
    "level": "Low",
    "primary_risk_driver": "Unclear scope boundary"
  }},
  "strategic_alignment": {{
    "hsbc_priorities": []
  }},
  "delivery_signals": {{
    "tech_stack_indicators": [],
    "skills_required_top3": []
  }}
}}
""".strip()


@torch.no_grad()
def run_llm(prompt: str) -> str:
    if USE_LOCAL_MODEL:
        inputs = tokenizer(prompt, return_tensors="pt")
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,   # deterministic
            temperature=0.0,
        )
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # API mode stub: plug your API client here later
    raise NotImplementedError("API mode is not wired yet. Set USE_LOCAL_MODEL=True for local inference.")


def analyze_story(story: str) -> dict:
    text = run_llm(build_prompt(story))

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "ownership": {"is_defined": False, "confidence": 0.0, "owner_or_actor": None},
            "intent": {"is_defined": False, "confidence": 0.0, "primary_intent": None, "type": None},
            "value": {"is_defined": False, "confidence": 0.0, "category": None, "direct_customer_impact": "No"},
            "delivery_clarifications": {"top2_questions": ["", ""]},
            "scope": {"shape": None},
            "execution_risk": {"level": "High", "primary_risk_driver": "Unclear scope boundary"},
            "strategic_alignment": {"hsbc_priorities": []},
            "delivery_signals": {"tech_stack_indicators": [], "skills_required_top3": []},
        }


if __name__ == "__main__":
    jira_description = (
        "As an admin, I want to reset user passwords via the portal to reduce support tickets. "
        "Acceptance Criteria: Admin can trigger a reset link; user receives email within 2 minutes. "
        "We must integrate with the existing Identity API."
    )

    result = analyze_story(jira_description)
    print(json.dumps(result, indent=2))
