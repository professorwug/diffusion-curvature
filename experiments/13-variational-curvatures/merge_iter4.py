"""Robust merge of iter4 shards (ar1 rows have an extra r_td4_dec field)."""
from pathlib import Path
import pandas as pd

C9 = ["profile", "d", "wseed", "noise", "r_td4", "cv", "r_sent5k",
      "r_ceil5k", "t"]
C10 = ["profile", "d", "wseed", "noise", "r_td4", "cv", "r_td4_dec",
       "r_sent5k", "r_ceil5k", "t"]
rows = []
for p in sorted(Path("processed_data").glob("iter4_w*.csv")):
    for line in p.read_text().splitlines():
        if line.startswith("profile") or not line.strip():
            continue
        f = line.split(",")
        cols = C9 if len(f) == 9 else C10
        rows.append(dict(zip(cols, f)))
df = pd.DataFrame(rows)
for c in df.columns:
    if c not in ("profile", "noise"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.drop_duplicates(["profile", "d", "wseed", "noise"], keep="last")
df.to_csv("processed_data/iter4.csv", index=False)
print(f"{len(df)}/96 units")
pd.set_option("display.width", 240)
print(df.groupby(["noise", "d"])[["r_td4", "r_td4_dec", "r_sent5k",
                                  "r_ceil5k"]].mean().round(3).to_string())
print("\nCV medians:", df.groupby("noise")["cv"].median().round(2).to_dict())
