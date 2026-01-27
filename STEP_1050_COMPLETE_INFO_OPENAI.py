from openai import OpenAI
import json

client = OpenAI()

def analyze_jira_story(jira_description: str) -> dict:
    prompt = f"""
You are a strict enterprise Jira Story Analyzer.

RULES (VERY IMPORTANT):
- Use ONLY the provided text.
- Do NOT infer or guess.
- If something is not explicitly stated, mark Identified = false.
- Confidence must be an INTEGER between 0 and 100.
- Evidence must be an EXACT snippet copied from the text (no paraphrasing).
- If Identified = false, Evidence and Category must be null.
- Return ONLY valid JSON. No markdown. No explanation.

Allowed Intent Types:
Create, Modify, Remove, Migrate, Integrate, Investigate, Enforce

Allowed Value Categories:
Customer, Cost, Risk, Compliance, Internal Efficiency

Allowed Customer Impact Levels:
Low, Medium, High

TEXT:
\"\"\"{jira_description}\"\"\"

OUTPUT JSON SCHEMA:
{{
  "who": {{
    "identified": boolean,
    "confidence": number,
    "actor": string | null,
    "evidence": string | null
  }},
  "what": {{
    "identified": boolean,
    "confidence": number,
    "intent_type": string | null,
    "intent_evidence": string | null
  }},
  "why": {{
    "identified": boolean,
    "confidence": number,
    "value_category": string | null,
    "value_evidence": string | null
  }},
  "customer_impact": {{
    "identified": boolean,
    "confidence": number,
    "impact_level": string | null,
    "impact_evidence": string | null
  }}
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        top_p=1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)
