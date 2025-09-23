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




import pandas as pd
import glob
from statsmodels.stats.inter_rater import fleiss_kappa

files = glob.glob(r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\OUTPUT\HUMAN_LABELLING\HUMAN_LABELING_*.xlsx")

dfs = []
for f in files:
    name = f.split("_")[-1].split(".")[0].upper()
    df = pd.read_excel(f, sheet_name="Label")
    df = df[["Key", "Label"]].copy()
    df.rename(columns={"Label": f"{name}_LABEL"}, inplace=True)
    dfs.append(df)

merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Key", how="inner")

label_cols = [c for c in merged.columns if c.endswith("_LABEL")]
n_annotators = len(label_cols)

merged["AGREEMENT"] = merged[label_cols].mode(axis=1).count(axis=1) / n_annotators
merged["HUMAN_LABEL"] = merged[label_cols].mode(axis=1)[0]

cats = [0,1,2]
mat = []
for _, row in merged.iterrows():
    counts = [(row[label_cols] == c).sum() for c in cats]
    mat.append(counts)

fkappa = fleiss_kappa(mat)
merged["FLEISS_KAPPA"] = fkappa

for c in cats:
    merged[f"AGREEMENT_ON_{c}"] = (merged[label_cols] == c).sum(axis=1) / n_annotators

merged.to_excel("FINAL_LABELS.xlsx", index=False)

