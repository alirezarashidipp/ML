from transformers import pipeline

MODEL_PATH = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL_CODE\bart-Large-mnli-tensors"

classifier = pipeline("zero-shot-classification", model=MODEL_PATH)

text = "As a developer, I want to deploy the code."
labels = ["Professional Role", "General Intent", "Greeting"]

result = classifier(text, labels)

print(f"\nText: {result['sequence']}")
for l, s in zip(result['labels'], result['scores']):
    print(f"{l}: {s:.2%}")
