# model: meta-llama/Llama-3.2-1B-Instruct
# Python== 3.9
# Torch = 2.7.0
# transformers = 4.49.0
# tokenizers = 0.21.0
# langgraph = 0.2.62

import os
import torch
from typing import TypedDict, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, END

# ==========================================
# CONFIGURATION
# ==========================================
# Hardware and Model Configuration
MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Generation Parameters
GEN_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.1,  # Low temperature for deterministic output
    "do_sample": True,
    "top_p": 0.95
}

print(f"--- System Initialization ---")
print(f"Device: {DEVICE}")
print(f"Model Path: {MODEL_PATH}")

# ==========================================
# MODEL LOADER (Singleton Pattern)
# ==========================================
class LocalLLM:
    """
    Wrapper for the local Llama 3.2 model to handle loading and inference.
    """
    def __init__(self, model_path: str):
        try:
            print("Loading Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print("Loading Model (this may take a moment)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=TORCH_DTYPE,
                device_map=DEVICE
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE == "cuda" else -1
            )
            print("Model loaded successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates response using Llama 3 standard prompting template.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=GEN_CONFIG["max_new_tokens"],
            do_sample=GEN_CONFIG["do_sample"],
            temperature=GEN_CONFIG["temperature"],
            top_p=GEN_CONFIG["top_p"],
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the new generated text
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()
        return generated_text

# Initialize Model Global Instance
try:
    llm_engine = LocalLLM(MODEL_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit(1)

# ==========================================
# GRAPH STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    """
    Defines the state structure passed between nodes in the LangGraph.
    """
    raw_input: str          # Original input from user
    role: Optional[str]     # Extracted or provided role
    intent: Optional[str]   # Extracted intent/action
    final_output: str       # Final Agile User Story

# ==========================================
# NODE DEFINITIONS
# ==========================================

def extraction_node(state: AgentState) -> AgentState:
    """
    Analyzes raw input to extract Role and Intent using the LLM.
    """
    print("\n[Node] Extracting information...")
    
    system_prompt = (
        "You are an expert Agile Business Analyst. "
        "Your task is to extract the 'Role' and the 'Intent' from the user's description. "
        "If the Role is not explicitly stated, output 'MISSING'. "
        "Format your answer exactly as:\n"
        "ROLE: [extracted role or MISSING]\n"
        "INTENT: [extracted intent]"
    )
    
    response = llm_engine.generate(system_prompt, state["raw_input"])
    
    # Simple parsing logic
    extracted_role = None
    extracted_intent = None
    
    lines = response.split('\n')
    for line in lines:
        if line.upper().startswith("ROLE:"):
            role_val = line.split(":", 1)[1].strip()
            if role_val.upper() != "MISSING":
                extracted_role = role_val
        if line.upper().startswith("INTENT:"):
            extracted_intent = line.split(":", 1)[1].strip()
            
    # Fallback if intent wasn't parsed clearly, use raw input
    if not extracted_intent:
        extracted_intent = state["raw_input"]

    return {"role": extracted_role, "intent": extracted_intent}

def human_input_node(state: AgentState) -> AgentState:
    """
    Simulates a Human-in-the-loop interaction to get the missing role.
    """
    print("\n[Node] Role Missing.")
    print(f"System detected intent: '{state['intent']}'")
    
    # In a real web app, this would pause execution. Here, we use input().
    user_provided_role = input(">> Please specify the ROLE for this feature (e.g., Data Engineer): ")
    
    return {"role": user_provided_role}

def rewriting_node(state: AgentState) -> AgentState:
    """
    Synthesizes the final Agile User Story.
    """
    print("\n[Node] Rewriting Story...")
    
    role = state["role"]
    intent = state["intent"]
    
    # Direct formatting is preferred for strict adherence, 
    # but we can use LLM to ensure grammar is perfect.
    system_prompt = (
        "You are an Agile Coach. Rewrite the following inputs into a standard User Story."
        "Format: 'As a <Role>, I want to <Intent>.'"
        "Do NOT add a 'So that' clause. Do NOT output JSON. Output only the text."
    )
    
    user_prompt = f"Role: {role}\nIntent: {intent}"
    final_story = llm_engine.generate(system_prompt, user_prompt)
    
    return {"final_output": final_story}

# ==========================================
# EDGE LOGIC
# ==========================================

def check_role_condition(state: AgentState) -> Literal["ask_user", "rewrite"]:
    """
    Determines the next step based on whether the role exists.
    """
    if state.get("role") and state["role"].upper() != "MISSING":
        return "rewrite"
    return "ask_user"

# ==========================================
# GRAPH CONSTRUCTION
# ==========================================

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("extractor", extraction_node)
workflow.add_node("ask_user", human_input_node)
workflow.add_node("rewrite", rewriting_node)

# Add Edges
workflow.set_entry_point("extractor")

workflow.add_conditional_edges(
    "extractor",
    check_role_condition,
    {
        "ask_user": "ask_user",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("ask_user", "rewrite")
workflow.add_edge("rewrite", END)

# Compile Graph
app = workflow.compile()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*40)
    print(" AGENT READY. WAITING FOR INPUT.")
    print("="*40)

    # Example 1: Missing Role
    # input_text = "I want to add a new column to the user table."
    
    # Example 2: With Role
    # input_text = "As a Data Engineer, I want to migrate the database."

    user_input = input("\nEnter Description: ")
    
    initial_state = {"raw_input": user_input, "role": None, "intent": None, "final_output": ""}
    
    # Run the graph
    result = app.invoke(initial_state)
    
    print("\n" + "="*40)
    print(" FINAL AGILE USER STORY")
    print("="*40)
    print(result["final_output"])
    print("="*40)
