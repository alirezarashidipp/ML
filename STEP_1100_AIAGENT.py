import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

# 1. Model Path
path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\Local_LLM"

print("--- Loading Model & Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(path)
# Important: Using float32 for CPU to avoid gibberish output
model = AutoModelForCausalLM.from_pretrained(
    path, 
    torch_dtype=torch.float32, 
    device_map=None 
)

# 2. Pipeline Creation
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1, # Lower temperature for more stable output
    top_p=0.9,
    return_full_text=False
)

local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

print("--- Model Loaded Successfully ---")

# 3. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 4. Define Node
def call_model(state: AgentState):
    messages = state['messages']
    user_content = messages[-1].content
    
    # Manually formatting for Llama-3.2-1B Instruct
    system_prompt = "You are an expert Agile Product Owner. Convert input into Jira User Story (As a... I want to... So that...) and Acceptance Criteria."
    
    # Constructing a clear prompt for the model
    full_prompt = f"System: {system_prompt}\nUser: {user_content}\nAssistant:"
    
    response = local_llm.invoke(full_prompt)
    
    # Return the clean text response
    return {"messages": [response]}

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent_node", call_model)
workflow.add_edge(START, "agent_node")
workflow.add_edge("agent_node", END)

app = workflow.compile()

# 6. Execution / Testing
user_input = "fixing bug in prod when user wants to log in"
print(f"\nUser Input: {user_input}\n")
print("--- Agent is thinking (this might take a minute on CPU)... ---")

input_data = {"messages": [HumanMessage(content=user_input)]}
result = app.invoke(input_data)

print("\n--- Final Jira Ticket ---")
# The result is now a clean string
print(result['messages'][-1])
