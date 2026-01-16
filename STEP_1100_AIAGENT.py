import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
import operator

# 1. Model Path (Specify your local directory)
path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\Local_LLM"

print("--- Loading Model & Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, 
    torch_dtype=torch.float16, # For faster inference and lower VRAM usage
    device_map=None # Automatically utilize GPU if available
)

# 2. Pipeline Creation (Bridge)
# Converts the raw model into a standard LangChain tool
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Allowed output length
    do_sample=True,
    temperature=0.7,
    return_full_text=False # Only return the generated text
)

# Create the LLM object compatible with LangGraph
local_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

print("--- Model Loaded Successfully ---")

# 3. Define State
# The agent's memory that stores the message history
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 4. Define Node (The "Brain" of the logic)
def call_model(state: AgentState):
    messages = state['messages']
    
    # Add system prompt if not already present
    # This instructs the model to act as an Agile Product Owner
    system_prompt = """You are an expert Agile Product Owner. 
    Your task is to convert the user's raw input into a standard Jira User Story format.
    Output must include:
    1. User Story (As a... I want to... So that...)
    2. Acceptance Criteria
    Keep it clear and professional."""
    
    # Prepend the system prompt to the message list
    if not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=system_prompt))
        
    response = local_llm.invoke(messages)
    
    # Append the model's response to the message list
    return {"messages": [response]}

# 5. Build Graph (The Workflow Map)
workflow = StateGraph(AgentState)

# Add the processing node
workflow.add_node("agent_node", call_model)

# Define the execution flow (Start -> Node -> End)
workflow.add_edge(START, "agent_node")
workflow.add_edge("agent_node", END)

# Compile the graph
app = workflow.compile()

# 6. Execution / Testing
user_input = "fixing bug in prod when user wants to log in"
print(f"\nUser Input: {user_input}\n")
print("--- Agent is thinking... ---")

# Pass input data into the graph
input_data = {"messages": [HumanMessage(content=user_input)]}
result = app.invoke(input_data)

print("\n--- Final Jira Ticket ---")
# Print the final message (model's response)
print(result['messages'][-1].content)
