import polars as pl

df = pl.DataFrame({"hl": [1.0, 2.0, 3.0], "hc": [None, 5.0, 1.0], "lc": [None, 0.5, 2.0]})
out = df.select(pl.max_horizontal("hl", "hc", "lc").alias("tr")).to_series().to_list()
print("max_horizontal with leading nulls:", out)
