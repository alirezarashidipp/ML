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
# مطمئن شویم پدینگ به درستی تنظیم شده
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
    max_new_tokens=256, # برای تست کوتاه‌تر کردیم
    do_sample=False,    # حذف نمونه‌برداری تصادفی برای دقت بیشتر
    repetition_penalty=1.2, # جلوگیری از تکرار (مثل 1: 1: 1)
    return_full_text=False
)

local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("--- Model Loaded Successfully ---")

# 3. Define State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 4. Define Node
def call_model(state: AgentState):
    user_message = state['messages'][-1].content
    
    # استفاده از ساختار استاندارد لاما 3
    chat = [
        {"role": "system", "content": "You are an expert Agile Product Owner. Convert input into Jira User Story and Acceptance Criteria."},
        {"role": "user", "content": user_message},
    ]
    
    # این بخش جادوی اصلی است: تبدیل به فرمت مخصوص مدل
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    response = local_llm.invoke(prompt)
    return {"messages": [response]}

# 5. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent_node", call_model)
workflow.add_edge(START, "agent_node")
workflow.add_edge("agent_node", END)
app = workflow.compile()

# 6. Execution
user_input = "fixing bug in prod when user wants to log in"
print(f"\nUser Input: {user_input}\n")
print("--- Agent is thinking... ---")

input_data = {"messages": [HumanMessage(content=user_input)]}
result = app.invoke(input_data)

print("\n--- Final Jira Ticket ---")
print(result['messages'][-1])
