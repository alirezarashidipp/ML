# model: meta-llama/Llama-3.2-1B-Instruct
# Python== 3.9
# Torch = 2.7.0
# transformers = 4.49.0
# tokenizers = 0.21.0
# langgraph = 0.2.62

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

# ----------------------------
# 1) Configuration
# ----------------------------
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"

DEVICE = "cpu"  # Keep CPU for stability; switch to "cuda" if you have a GPU and correct setup
DTYPE = torch.float32  # Safer for CPU

# ----------------------------
# 2) Load model + tokenizer
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model.eval()

# Ensure pad token is set (common for LLaMA-like tokenizers)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# 3) State
# ----------------------------
class JiraState(TypedDict, total=False):
    raw_text: str
    extracted_json: dict
    missing_fields: List[str]
    questions: List[str]
    answers: Dict[str, str]
    final_story: str


# ----------------------------
# 4) Prompts
# ----------------------------
EXTRACT_PROMPT = """
You are an information extractor for Agile Jira stories.

Hard rules:
- DO NOT invent anything.
- Only extract what is explicitly present in the user's text.
- For every extracted field, provide an evidence quote copied from the user's text.
- If not present, set the field to null and evidence to null.
- Return STRICT JSON only. No extra commentary.

Schema:
{
  "role": {"value": string|null, "evidence": string|null},
  "what": {"value": string|null, "evidence": string|null},
  "why":  {"value": string|null, "evidence": string|null},
  "acceptance_criteria": [{"value": string, "evidence": string}]
}

User text:
<<<{raw_text}>>>
""".strip()

WRITE_PROMPT = """
You are an Agile story rewriter.

Hard rules:
- Use ONLY the provided fields. DO NOT add new facts.
- If something is missing, do not guess; mark it clearly as MISSING.
- Output clean Jira-ready text.

Fields JSON:
{fields_json}

Write:
1) User Story: As a {Role}, I want {WHAT}, so that {WHY}.
2) Acceptance Criteria: bullets (only those provided; if none, write "MISSING").
""".strip()


# ----------------------------
# 5) Helpers
# ----------------------------
def _generate_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_json(text: str) -> dict:
    """
    Tries strict JSON first; if that fails, extracts the first JSON object block.
    This is important because small models sometimes add extra text.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")

    return json.loads(match.group(0))


def llm_call(prompt: str, mode: str) -> str:
    """
    mode:
      - 'extract' => deterministic, lowest hallucination
      - 'write'   => slightly flexible but still controlled
    """
    if mode == "extract":
        return _generate_text(prompt, max_new_tokens=256, temperature=0.0)
    elif mode == "write":
        return _generate_text(prompt, max_new_tokens=256, temperature=0.2)
    else:
        raise ValueError("Unknown mode.")


# ----------------------------
# 6) Nodes
# ----------------------------
def extract_fields(state: JiraState) -> JiraState:
    prompt = EXTRACT_PROMPT.format(raw_text=state["raw_text"])
    out = llm_call(prompt, mode="extract")
    extracted = _extract_json(out)
    return {"extracted_json": extracted}


def validate_fields(state: JiraState) -> JiraState:
    ex = state["extracted_json"]

    missing: List[str] = []
    for key in ["role", "what", "why"]:
        val = ex.get(key, {}).get("value")
        ev = ex.get(key, {}).get("evidence")
        if val is None or str(val).strip() == "" or ev is None or str(ev).strip() == "":
            missing.append(key)

    ac = ex.get("acceptance_criteria", [])
    if not isinstance(ac, list) or len(ac) == 0:
        missing.append("acceptance_criteria")

    questions: List[str] = []
    if "role" in missing:
        questions.append("Who is the Role? (e.g., Customer / PO / Admin / Developer)")
    if "what" in missing:
        questions.append("What exactly is the deliverable / requested change? (WHAT)")
    if "why" in missing:
        questions.append("What is the value or reason? (WHY)")
    if "acceptance_criteria" in missing:
        questions.append("Provide Acceptance Criteria (preferably Given/When/Then or bullet points).")

    return {"missing_fields": missing, "questions": questions}


def need_clarification(state: JiraState) -> str:
    return "ask" if state.get("missing_fields") else "write"


def ask_user(state: JiraState) -> JiraState:
    return state


def merge_answers_and_loop(state: JiraState) -> JiraState:
    answers = state.get("answers", {})
    if answers:
        appended = "\n\nUser clarifications:\n" + "\n".join([f"{k}: {v}" for k, v in answers.items()])
        return {"raw_text": state["raw_text"] + appended}
    return state


def write_story(state: JiraState) -> JiraState:
    fields_json = json.dumps(state["extracted_json"], ensure_ascii=False)
    prompt = WRITE_PROMPT.format(fields_json=fields_json)
    out = llm_call(prompt, mode="write")
    return {"final_story": out}


# ----------------------------
# 7) Build graph
# ----------------------------
graph = StateGraph(JiraState)
graph.add_node("extract", extract_fields)
graph.add_node("validate", validate_fields)
graph.add_node("ask", ask_user)
graph.add_node("merge", merge_answers_and_loop)
graph.add_node("write", write_story)

graph.add_edge(START, "extract")
graph.add_edge("extract", "validate")
graph.add_conditional_edges("validate", need_clarification, {"ask": "ask", "write": "write"})
graph.add_edge("ask", "merge")
graph.add_edge("merge", "extract")
graph.add_edge("write", END)

app = graph.compile()


# ----------------------------
# 8) Simple CLI runner
# ----------------------------
def run_once(user_text: str) -> JiraState:
    state: JiraState = {"raw_text": user_text}
    while True:
        state = app.invoke(state)

        if state.get("questions"):
            print("\nMissing fields detected. Please answer:")
            answers: Dict[str, str] = {}
            for q in state["questions"]:
                ans = input(f"- {q}\n  > ").strip()
                answers[q] = ans

            state["answers"] = answers
            state["questions"] = []
            state["missing_fields"] = []
            continue

        break

    return state


if __name__ == "__main__":
    text = input("Paste your Jira task text:\n> ").strip()
    result = run_once(text)
    print("\n" + "=" * 40)
    print("FINAL STORY:")
    print(result.get("final_story", "No final_story produced."))
    print("=" * 40)
