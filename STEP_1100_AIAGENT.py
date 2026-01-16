import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. مسیر مدل
path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\Local_LLM"

print("--- Loading Model & Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float32,
    device_map=None  # CPU
)

# 2. ساخت Pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    return_full_text=False
)

print("--- Model Loaded Successfully ---")

# 3. تعریف تابع برای ساخت Jira Ticket
def generate_jira_ticket(user_input: str):
    system_prompt = (
        "You are an expert Agile Product Owner. "
        "Convert input into Jira User Story (As a... I want to... So that...) "
        "and provide Acceptance Criteria."
    )
    full_prompt = f"{system_prompt}\n\nUser Input: {user_input}\n\nJira Ticket:"

    # گرفتن پاسخ از مدل
    outputs = text_generation_pipeline(full_prompt)
    # outputs یک لیست دیکشنری است، متن اصلی در outputs[0]['generated_text']
    ticket_text = outputs[0]['generated_text'].strip()
    return ticket_text

# 4. تست تابع
user_input = "fixing bug in prod when user wants to log in"
print("\nUser Input:", user_input)
print("\n--- Generated Jira Ticket ---\n")
jira_ticket = generate_jira_ticket(user_input)
print(jira_ticket)
