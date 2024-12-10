import pandas as pd
import os

source_dir = "./data/unprocessed_levels/"
output_file = "./data/tabular_only.csv"

df = pd.DataFrame(columns=["gamestyle", "theme", "tag1", "tag2", "timer", "likes-norm"])
for filename in os.listdir(source_dir):
    df_b = pd.read_parquet(os.path.join(source_dir, filename))
    df_b['likes-norm'] = df_b["likes"] / (df_b["likes"] + df_b["boos"])
    df_b = df_b.drop(labels=["data_id","level_data","likes","boos"], axis=1)
    df = pd.concat([df, df_b], ignore_index=True)

print(df)
df.to_csv(output_file)