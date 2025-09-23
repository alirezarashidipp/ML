import pandas as pd
import glob
from statsmodels.stats.inter_rater import fleiss_kappa

files = glob.glob(r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\OUTPUT\HUMAN_LABELLING\HUMAN_LABELLING_*.xlsx")

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

merged["AGREEMENT"] = merged[label_cols].apply(
    lambda row: row.value_counts(normalize=True).max(), axis=1
)

def majority_or_nan(row):
    vc = row.value_counts()
    if vc.empty:
        return None
    max_count = vc.max()
    winners = vc[vc == max_count].index.tolist()
    return winners[0] if len(winners) == 1 else None

merged["HUMAN_LABEL"] = merged[label_cols].apply(majority_or_nan, axis=1)

cats = [0,1,2]
mat = []
for _, row in merged.iterrows():
    counts = [(row[label_cols] == c).sum() for c in cats]
    mat.append(counts)

fkappa = fleiss_kappa(mat)
merged["FLEISS_KAPPA"] = fkappa

for c in cats:
    merged[f"AGREEMENT_ON_{c}"] = (merged[label_cols] == c).sum(axis=1) / n_annotators

# محاسبه اختلاف annotator با اکثریت
diff_scores = {}
for a in label_cols:
    diff_scores[a] = (merged[a] != merged["HUMAN_LABEL"]).mean()

summary = pd.DataFrame({
    "Annotator": list(diff_scores.keys()),
    "Disagreement_with_majority": list(diff_scores.values())
})

with pd.ExcelWriter("FINAL_LABELS.xlsx", engine="openpyxl") as writer:
    merged.to_excel(writer, sheet_name="Labels", index=False)
    summary.to_excel(writer, sheet_name="Annotator_Summary", index=False)

print("[done] Saved FINAL_LABELS.xlsx with 2 sheets")
print(f"Fleiss Kappa = {fkappa:.3f}")
