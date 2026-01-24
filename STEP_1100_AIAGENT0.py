# model: meta-llama/Llama-3.2-1B-Instruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\LLM\LLAMA"

try:
    print("Step 1: Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    print("Step 2: Loading Model (This may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(path)
    
    print("Step 3: Preparing Input...")
    prompt = "Where is capital of Poland?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("Step 4: Generating Response...")
    # Added do_sample=True for better results
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.5)
    
    print("Step 5: Decoding...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("-" * 30)
    print("Model Output:")
    print(response)
    print("-" * 30)

except Exception as e:
    print(f"Error occurred: {e}")
