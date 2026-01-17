# ============================================================
# Jira Story Mini-Agent (Role + What) using LangGraph + LangChain
# Local model: meta-llama/Llama-3.2-1B-Instruct (safetensors)
# Python 3.9 | torch 2.7.0 | transformers 4.49.0
# tokenizers 0.21.0 | langgraph 0.2.62
# ------------------------------------------------------------
# Behavior (Console/Spyder):
# 1) Asks: "Write your Description:"
# 2) Extracts ROLE + WHAT from the text (only these 2)
# 3) If missing, asks user (Role is missing / What is missing)
# 4) Finally rewrites as: "As a <ROLE>, I want <WHAT>."
# ============================================================

from typing import Optional, TypedDict, Literal, Dict, Any
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, START, END


# ----------------------------
# 0) CONFIG: set your local model path
# ----------------------------
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"


# ----------------------------
# 1) Load local model (CPU)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)  # CPU by default

# For extraction: deterministic, no sampling
extract_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,
    temperature=0.0,
    max_new_tokens=120,
    return_full_text=False,
)

# For rewrite: also deterministic (keep it stable)
rewrite_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,
    temperature=0.0,
    max_new_tokens=80,
    return_full_text=False,
)

llm_extract = HuggingFacePipeline(pipeline=extract_pipe)
llm_rewrite = HuggingFacePipeline(pipeline=rewrite_pipe)


# ----------------------------
# 2) State definition
# ----------------------------
class JiraState(TypedDict, total=False):
    user_input: str
    role: Optional[str]
    what: Optional[str]
    missing: Literal["role", "what", "none"]
    assistant_message: str


# ----------------------------
# 3) Prompts
# ----------------------------
extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an information extractor for Jira user story inputs.\n"
     "Extract ONLY two fields: ROLE and WHAT.\n"
     "Return ONLY the following two lines, no extra text:\n"
     "ROLE: <value or NONE>\n"
     "WHAT: <value or NONE>\n"
     "Rules:\n"
     "- ROLE is a persona (admin, user, customer, analyst, developer, etc.).\n"
     "- WHAT is a concrete requested change/capability (one sentence).\n"
     "- If a field is missing or unclear, write NONE.\n"),
    ("human", "{text}")
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite into one simple Agile user story in plain English.\n"
     "Return ONLY one line in this exact format:\n"
     "As a <ROLE>, I want <WHAT>.\n"
     "Keep it short, factual, and do not add new requirements.\n"),
    ("human", "ROLE: {role}\nWHAT: {what}")
])

extract_chain = extract_prompt | llm_extract | StrOutputParser()
rewrite_chain = rewrite_prompt | llm_rewrite | StrOutputParser()


# ----------------------------
# 4) Helpers
# ----------------------------
def _clean_value(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    # remove wrapping quotes
    s = s.strip("\"'“”‘’")
    return s if s else None


def _parse_extract(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Expected:
      ROLE: ...
      WHAT: ...
    But we parse robustly.
    """
    role = None
    what = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for l in lines:
        if l.upper().startswith("ROLE:"):
            val = l.split(":", 1)[1].strip()
            role = None if val.upper() == "NONE" else _clean_value(val)
        elif l.upper().startswith("WHAT:"):
            val = l.split(":", 1)[1].strip()
            what = None if val.upper() == "NONE" else _clean_value(val)

    return role, what


def _normalize_story_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # If model accidentally returns multiple lines, take first non-empty
    if "\n" in s:
        s = next((x.strip() for x in s.splitlines() if x.strip()), s).strip()
    return s


def _looks_like_story(s: str) -> bool:
    s2 = s.strip().lower()
    return s2.startswith("as a ") and " i want " in s2 and s2.endswith(".")


# ----------------------------
# 5) Nodes
# ----------------------------
def node_extract(state: JiraState) -> JiraState:
    raw = extract_chain.invoke({"text": state["user_input"]})
    role, what = _parse_extract(raw)
    return {**state, "role": role, "what": what}


def node_check_missing(state: JiraState) -> JiraState:
    if not _clean_value(state.get("role")):
        return {**state, "missing": "role"}
    if not _clean_value(state.get("what")):
        return {**state, "missing": "what"}
    return {**state, "missing": "none"}


def node_rewrite(state: JiraState) -> JiraState:
    role = _clean_value(state.get("role")) or ""
    what = _clean_value(state.get("what")) or ""
    story = rewrite_chain.invoke({"role": role, "what": what})
    story = _normalize_story_line(story)

    # If the model returns something slightly off-format, enforce minimal formatting:
    if not _looks_like_story(story):
        story = f"As a {role}, I want {what}."
        story = _normalize_story_line(story)

    return {**state, "assistant_message": story, "missing": "none"}


# ----------------------------
# 6) Graph
# ----------------------------
g = StateGraph(JiraState)
g.add_node("extract", node_extract)
g.add_node("check_missing", node_check_missing)
g.add_node("rewrite", node_rewrite)

g.add_edge(START, "extract")
g.add_edge("extract", "check_missing")

def route_after_check(state: JiraState) -> str:
    # If anything missing, stop and ask user via console loop (outside graph)
    return "rewrite" if state.get("missing") == "none" else END

g.add_conditional_edges("check_missing", route_after_check, {
    "rewrite": "rewrite",
    END: END
})

g.add_edge("rewrite", END)

app = g.compile()


# ----------------------------
# 7) Console loop (Spyder-friendly)
# ----------------------------
def interactive_console():
    print("Write your Description:")
    user_text = input("> ").strip()

    state: JiraState = {
        "user_input": user_text,
        "role": None,
        "what": None
    }

    # First pass: extract/check/rewrite if possible
    out: JiraState = app.invoke(state)

    # If graph ended early (missing fields), ask user and rerun until complete
    while out.get("missing") in ("role", "what") or not out.get("assistant_message"):
        missing = out.get("missing")

        if missing == "role":
            print("\nRole is missing.")
            role_in = input("Please provide Role (e.g., admin, customer, analyst): ").strip()
            state["role"] = role_in
        elif missing == "what":
            print("\nWhat is missing.")
            what_in = input("Please describe WHAT should be done (one sentence): ").strip()
            state["what"] = what_in
        else:
            # If missing isn't set but no message, we attempt to continue safely
            if not state.get("role"):
                print("\nRole is missing.")
                state["role"] = input("Please provide Role (e.g., admin, customer, analyst): ").strip()
            elif not state.get("what"):
                print("\nWhat is missing.")
                state["what"] = input("Please describe WHAT should be done (one sentence): ").strip()

        out = app.invoke(state)

        # If still missing, app.invoke will end early again (END path)
        # So we need to run check_missing ourselves to know what's missing after providing one field.
        if out.get("missing") not in ("role", "what", "none"):
            # Safety: run check_missing locally
            out = node_check_missing({**state, **out})

    print("\nFinal Jira Story:")
    print(out["assistant_message"])


# ----------------------------
# 8) Run
# ----------------------------
if __name__ == "__main__":
    try:
        interactive_console()
    except Exception as e:
        print(f"\nError occurred: {e}")
