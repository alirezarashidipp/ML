Annotation Guideline (Plain Version)

Task: For each ticket description, decide how easy it is to read and understand. Use the following scale:

0 – Poor readability

Very hard to read or understand.

Sentences are broken, unclear, or incomplete.

The main idea is missing or confusing.

1 – Acceptable readability

Can be read and understood, but not smooth.

Some parts are unclear or awkward.

The main idea is present, but details may be missing.

2 – Good readability

Easy to read and understand.

Sentences are clear, complete, and logical.

The main idea is expressed well and no extra effort is needed.

Rules:

Each ticket must have exactly one label: 0, 1, or 2.

Do not leave any ticket blank.

Focus only on readability, not on technical accuracy.





# -*- coding: utf-8 -*-
# Merge annotation Excel files (sheet="Label") + Validate + Graphs

import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Load & Merge =========
path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\OUTPUT\HUMAN_LABELLING\HUMAN_LABELING_*.xlsx"
files = glob.glob(path)
dfs = []

for f in files:
    name = f.split("_")[-1].split(".")[0].upper()   # e.g. "JOHN"
    df = pd.read_excel(f, sheet_name="Label")       # ← مشخصاً sheet="Label"
    df = df[["Key", "HUMAN_LABEL"]].copy()
    df.rename(columns={"HUMAN_LABEL": f"HUMAN_LABEL_{name}"}, inplace=True)
    dfs.append(df)

if not dfs:
    raise FileNotFoundError("No HUMAN_LABELING_*.xlsx files found in the directory!")

# Merge همه فایل‌ها روی Key
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Key", how="inner")

# ========= Validation =========
label_cols = [c for c in merged.columns if c.startswith("HUMAN_LABEL")]

# فقط مقادیر 0/1/2 مجاز هستند
for col in label_cols:
    bad = ~merged[col].isin([0, 1, 2])
    if bad.any():
        print(f"Invalid values in {col}:")
        print(merged.loc[bad, ["Key", col]])

# چک NaN
print("NaN counts per column:")
print(merged.isna().sum())

# ذخیره خروجی merged
merged.to_excel("MERGED_LABELS.xlsx", index=False)
print("[done] Saved MERGED_LABELS.xlsx")

# ========= Graphs =========

# 1) Pairwise agreement heatmap
n = len(label_cols)
agreement = pd.DataFrame(0.0, index=label_cols, columns=label_cols)

for i in range(n):
    for j in range(n):
        if i <= j:
            same = (merged[label_cols[i]] == merged[label_cols[j]]).mean()
            agreement.iloc[i, j] = same
            agreement.iloc[j, i] = same

plt.figure(figsize=(6,5))
sns.heatmap(agreement, annot=True, cmap="Blues", fmt=".2f")
plt.title("Pairwise Annotator Agreement")
plt.tight_layout()
plt.show()

# 2) Label distribution per annotator
fig, axes = plt.subplots(1, len(label_cols), figsize=(12,4), sharey=True)

for ax, col in zip(axes, label_cols):
    merged[col].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")

plt.suptitle("Label Distribution per Annotator")
plt.tight_layout()
plt.show()

# 3) Overall agreement rate
merged["all_equal"] = merged[label_cols].nunique(axis=1) == 1
agree_rate = merged["all_equal"].mean()

plt.figure(figsize=(4,4))
plt.bar(["Agreement", "Disagreement"],
        [agree_rate, 1-agree_rate], color=["green","red"])
plt.title("Overall Agreement Rate")
plt.ylim(0,1)
plt.show()

print(f"Overall agreement rate: {agree_rate:.2%}")

