import pandas as pd

file = "./data/unprocessed_levels/batch_31.parquet"
df = pd.read_parquet(file)

pd.set_option('display.max_rows', None)
# print(df)

df = df.drop(labels=range(433,5000))
print(df)

df.to_parquet(file)