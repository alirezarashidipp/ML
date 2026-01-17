# indentified missing Role, but output was not text.
import os
import torch
import json
from typing import TypedDict, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, END

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# برای جلوگیری از خطای حافظه در مدل‌های 1B، گاهی float16 بهتر است
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"--- System Initialization ---")
print(f"Device: {DEVICE}")

# ==========================================
# MODEL LOADER
# ==========================================
class LocalLLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.1, # دما پایین برای دقت بالا در استخراج
            do_sample=True, # برای مدل های Instruct بهتر است True باشد
        )

    def generate(self, messages) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(prompt)
        return outputs[0]["generated_text"][len(prompt):].strip()

# Load Model
llm_engine = LocalLLM(MODEL_PATH)

# ==========================================
# GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    raw_input: str
    role: Optional[str]
    intent: Optional[str]
    missing_info: bool # پرچمی برای مشخص کردن وضعیت ناقص بودن
    final_story: str

# ==========================================
# NODES
# ==========================================

def analyzer_node(state: AgentState) -> AgentState:
    """
    LLM نقش تحلیل‌گر را بازی می‌کند. به جای Regex، از هوش مصنوعی برای پیدا کردن نقش استفاده می‌کنیم.
    """
    print("\n[Analyzer] Analyzing input for Role and Intent...")
    
    # پرامپت مهندسی شده برای استخراج دقیق
    system_msg = (
        "You are a Data Sciecne Agile Assistant. Analyze the user text."
        "Extract the 'Role' (who) and the 'Intent' (what action)."
        "Return the answer in this specific format:\n"
        "ROLE: <extracted_role_or_NONE>\n"
        "INTENT: <extracted_intent_or_NONE>"
    )
    
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": state["raw_input"]}
    ]
    
    response = llm_engine.generate(msgs)
    
    # پارس کردن خروجی متنی مدل به متغیرهای پایتون
    extracted_role = None
    extracted_intent = None
    
    for line in response.split('\n'):
        if "ROLE:" in line:
            val = line.split("ROLE:")[1].strip()
            if val.upper() != "NONE" and len(val) > 2:
                extracted_role = val
        if "INTENT:" in line:
            val = line.split("INTENT:")[1].strip()
            if val.upper() != "NONE" and len(val) > 2:
                extracted_intent = val
    
    # اگر Intent پیدا نشد، کل متن ورودی را به عنوان Intent در نظر می‌گیریم (فرض منطقی)
    if not extracted_intent:
        extracted_intent = state["raw_input"]

    # تعیین وضعیت
    missing = False
    if not extracted_role:
        missing = True
        
    return {
        "role": extracted_role,
        "intent": extracted_intent,
        "missing_info": missing
    }

def question_node(state: AgentState) -> AgentState:
    """
    اگر اطلاعات ناقص بود، از کاربر می‌پرسد.
    """
    print(f"\n[System] I detected the intent: '{state['intent']}'")
    print("[System] But I don't know the ROLE (Who is this for?).")
    
    new_role = input(">> Please enter the Role (e.g., Data Engineer): ")
    
    return {"role": new_role, "missing_info": False}

def writer_node(state: AgentState) -> AgentState:
    """
    وقتی همه چیز کامل شد، متن نهایی را می‌نویسد.
    """
    print("\n[Writer] Generating User Story...")
    
    system_msg = (
        "You are an expert Agile Owner. Write a standard User Story."
        "Format: 'As a <Role>, I want <Intent>.'"
        "Do not include 'So that'. Output only the story."
    )
    
    user_msg = f"Role: {state['role']}\nIntent: {state['intent']}"
    
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    final_text = llm_engine.generate(msgs)
    return {"final_story": final_text}

# ==========================================
# LOGIC / EDGES
# ==========================================

def route_condition(state: AgentState) -> Literal["ask", "write"]:
    if state["missing_info"]:
        return "ask"
    return "write"

# ==========================================
# GRAPH BUILD
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("analyzer", analyzer_node)
workflow.add_node("ask", question_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("analyzer")

workflow.add_conditional_edges(
    "analyzer",
    route_condition,
    {
        "ask": "ask",
        "write": "writer"
    }
)

workflow.add_edge("ask", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    print("\n--- APP STARTED ---")
    user_in = input("Enter feature request: ")
    
    # مثال ورودی ناقص: "i want to add a login page"
    # مثال ورودی کامل: "as a admin i want to delete users"
    
    result = app.invoke({
        "raw_input": user_in, 
        "role": None, 
        "intent": None, 
        "missing_info": False,
        "final_story": ""
    })
    
    print("\n" + "*"*30)
    print("RESULT:")
    print(result["final_story"])
    print("*"*30)
