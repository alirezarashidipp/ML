
import sys
import numpy as np
import pandas as pd
import textstat          

# ─────────────────────────────────────────────────────────────
def normalize(raw, lo, hi, inverse=False):
    """Map raw∈[lo,hi] → 0-10.  Outside → NaN."""
    if pd.isna(raw) or raw < lo or raw > hi:
        return np.nan
    scale = (raw - lo) / (hi - lo)
    return round((1 - scale) * 10 if inverse else scale * 10, 2)






# readability map:  func , lower , upper , inverse?
METRICS = {
    "FRE":  (textstat.flesch_reading_ease,          -20, 100, False),  # higher = plainer
    "FKGL": (textstat.flesch_kincaid_grade, 0, 18, True), # New
    "Fog":  (textstat.gunning_fog,                  0,  25,  True),  # lower = plainer
    "LW":   (textstat.linsear_write_formula,        0,  25,  True),
    "ARI":  (textstat.automated_readability_index,  -10,  25,  True),
    "SMOG": (textstat.smog_index,                   3,  20,  True),
    "CLI":  (textstat.coleman_liau_index,           -10,  20,  True),
    "DC":  (textstat.dale_chall_readability_score,           1,  15,  True),
    "LIX":  (textstat.lix,           1,  80,  True),
    "SPACHE": (textstat.spache_readability, 1, 10, True),   # New

}

def enrich(df: pd.DataFrame, col="Description_clean") -> pd.DataFrame:
    for tag, (func, lo, hi, inv) in METRICS.items():
        raw_col  = tag
        norm_col = f"{tag}_norm"
        rng_col  = f"{tag}_range"

        # raw score
        df[raw_col] = df[col].astype(str).apply(
            lambda t: func(t) if t.strip() else np.nan
        )
        # normalised
        df[norm_col] = df[raw_col].apply(lambda x: normalize(x, lo, hi, inverse=inv))
        # constant range label
        df[rng_col] = f"{lo}-{hi}"
    return df

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python add_readability.py  input.csv  output.csv")

    inp, out = sys.argv[1], sys.argv[2]
    data = pd.read_csv(inp)
    data = enrich(data, col="Description_clean")
    
    norm_cols = [c for c in data.columns if c.endswith("_norm")]
    data = data[["Key", "Description_clean"] + norm_cols]
    
    data.to_csv(out, index=False)
    print(f"✅  Added readability columns → {out}")

