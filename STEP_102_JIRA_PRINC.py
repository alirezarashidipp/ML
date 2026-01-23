from transformers import pipeline

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"
clf = pipeline("zero-shot-classification", model=MODEL_PATH)

text = "As a developer, I want to deploy the code so that releases are faster."

labels = [
    "a clear WHO (explicit actor/role like 'as a user/admin/developer/system')",
    "a clear WHAT (explicit goal/action like 'I want to ... / system should ...')",
    "a clear WHY (explicit reason/value like 'so that / in order to / because')",
]

res = clf(
    text,
    labels,
    hypothesis_template="This Jira user story contains {}.",
    multi_label=True,   # ✅ critical
)

print(res["sequence"])
for l, s in zip(res["labels"], res["scores"]):
    print(f"{l}: {s:.2%}")
