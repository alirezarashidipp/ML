import json
import os

# ---------------------------------------------------------
# 1. تنظیمات اصلی (CONFIGURATION) - کنترل پنل شما
# ---------------------------------------------------------
CONFIG = {
    # انتخاب مود: 'HF' (هالینگ فیس), 'LOCAL' (دانلود شده), 'HSBC_API' (سرور شرکت)
    "SOURCE_MODE": "HF", 
    
    # تنظیمات Hugging Face
    "HF_MODEL_ID": "meta-llama/Llama-3.2-1B-Instruct",
    
    # تنظیمات Local (آدرس پوشه مدل روی سیستم شما)
    "LOCAL_PATH": "/content/drive/MyDrive/models/llama-1b-v2", 
    
    # تنظیمات API شرکت (HSBC)
    "API_URL": "https://api.hsbc.internal/v1/chat/completions",
    "API_KEY": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
    "API_TIMEOUT": 30
}

# ---------------------------------------------------------
# 2. بارگذاری اولیه (فقط برای حالت‌های لوکال و HF)
# ---------------------------------------------------------
pipeline_instance = None

if CONFIG["SOURCE_MODE"] in ["HF", "LOCAL"]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # تعیین مسیر مدل بر اساس انتخاب شما
    model_path = CONFIG["HF_MODEL_ID"] if CONFIG["SOURCE_MODE"] == "HF" else CONFIG["LOCAL_PATH"]
    
    print(f"🔄 Initializing Model from source: {CONFIG['SOURCE_MODE']} ({model_path})...")
    
    try:
        # تشخیص سخت‌افزار
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        pipeline_instance = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True
        )
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Hint: If using LOCAL, make sure the path is correct.")

elif CONFIG["SOURCE_MODE"] == "HSBC_API":
    import requests
    print("✅ System configured for API usage. No local model loading needed.")

# ---------------------------------------------------------
# 3. توابع کمکی (Abstraction Layer)
# ---------------------------------------------------------

def get_llm_response(messages):
    """
    این تابع تصمیم می‌گیرد درخواست را به کجا بفرستد
    بر اساس تنظیمات CONFIG["SOURCE_MODE"]
    """
    mode = CONFIG["SOURCE_MODE"]
    
    # --- روش ۱ و ۲: اجرا روی سخت‌افزار خودمان (HF / LOCAL) ---
    if mode in ["HF", "LOCAL"]:
        if pipeline_instance is None:
            return "Error: Model not loaded."
        
        outputs = pipeline_instance(messages)
        return outputs[0]["generated_text"][-1]["content"]

    # --- روش ۳: فراخوانی API شرکت (HSBC) ---
    elif mode == "HSBC_API":
        # اکثر APIهای شرکتی استاندارد OpenAI یا Azure را دنبال می‌کنند
        payload = {
            "model": "gpt-4-turbo-internal", # یا نام مدلی که شرکت داده
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['API_KEY']}", 
            # گاهی بانک‌ها هدرهای خاص خود را دارند مثل:
            # "x-api-key": CONFIG['API_KEY'],
            # "Ocp-Apim-Subscription-Key": CONFIG['API_KEY'] (اگر Azure باشد)
        }
        
        try:
            response = requests.post(
                CONFIG["API_URL"], 
                json=payload, 
                headers=headers, 
                timeout=CONFIG["API_TIMEOUT"],
                verify=False # در محیط‌های بانکی گاهی SSL داخلی self-signed است
            )
            response.raise_for_status()
            
            # پارس کردن جواب (معمولا ساختار choices[0].message.content دارند)
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"API Error: {str(e)}"

def clean_json_output(text):
    """پاکسازی متن برای استخراج JSON"""
    text = text.strip()
    if text.startswith("API Error") or text.startswith("Error"):
        return None
        
    if text.startswith("```json"):
        text = text.replace("```json", "", 1)
    if text.startswith("```"):
        text = text.replace("```", "", 1)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ---------------------------------------------------------
# 4. منطق اصلی برنامه (Main Logic - ثابت برای همه روش‌ها)
# ---------------------------------------------------------

def analyze_jira_ticket(ticket_text):
    system_prompt = """You are a JIRA analysis engine.
    Allowed Intents: [Create, Modify, Remove, Migrate, Integrate, Investigate, Enforce].
    
    Return a VALID JSON object with this structure:
    {
      "story_ownership": { "identified": boolean, "confidence": int, "owner": string or null },
      "primary_intent": { "defined": boolean, "confidence": int, "type": string }
    }
    Output ONLY JSON.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ticket_text},
    ]
    
    print(f"\n🚀 Processing via [{CONFIG['SOURCE_MODE']}]...")
    
    # 1. گرفتن متن خام از هر منبعی که انتخاب شده
    raw_response = get_llm_response(messages)
    
    # 2. تلاش برای تبدیل به آبجکت و نمایش
    clean_text = clean_json_output(raw_response)
    
    if clean_text is None:
        print("❌ Failed to get valid response.")
        print("Raw:", raw_response)
        return

    try:
        result = json.loads(clean_text)
        
        # نمایش خروجی
        print("-" * 30)
        so = result.get('story_ownership', {})
        print(f"STORY OWNERSHIP")
        print(f"Identified:      {'Yes' if so.get('identified') else 'No'} ({so.get('confidence', 0)}%)")
        print(f"Extracted Owner: {so.get('owner', 'N/A')}")
        
        print("-" * 30)
        pi = result.get('primary_intent', {})
        print(f"PRIMARY INTENT")
        print(f"Clearly Defined: {'Yes' if pi.get('defined') else 'No'} ({pi.get('confidence', 0)}%)")
        print(f"Intent Type:     {pi.get('type', 'N/A')}")
        print("-" * 30)
        
    except json.JSONDecodeError:
        print("❌ JSON Parsing Error. Model output was not valid JSON.")
        print("Raw Output:", raw_response)

# ---------------------------------------------------------
# 5. اجرا
# ---------------------------------------------------------

sample_text = "I am Payment Platform Product Owner, i want to build a system that easy integrate two code bases toghether."

analyze_jira_ticket(sample_text)
