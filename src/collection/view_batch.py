import pandas as pd

file = "./data/unprocessed_levels/batch_31.parquet"
df = pd.read_parquet(file)
print(df)