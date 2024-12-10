from datasets import load_dataset
import pandas as pd

# hyperparameters
min_review_count = 100
batch_prefix = "batch_"
batch_size = 5000
debug_wavelength = 2500
output_folder = "./data/unprocessed_levels/"
max_search_count = 0

def main():
    df = create_df()
    item_index = 0
    batch_index = 0
    total_count = 0

    ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")

    for item in iter(ds):
        total_count += 1

        if debug_wavelength > 0 and total_count % debug_wavelength == 0:
            print("Iteration %s" % total_count)
        
        if max_search_count > 0 and total_count >= max_search_count:
            break

        if item["likes"] + item["boos"] < min_review_count:
            continue
        
        for col in df.columns:
            df.at[item_index, col] = item[col]

        item_index += 1
        #if debug_wavelength > 0 and item_index % debug_wavelength == 0:
        #    print("Item index %s" % item_index)

        if item_index >= batch_size:
            save_df(df, batch_index, debug=True)
            df = create_df()

            item_index = 0
            batch_index += 1
    
    if item_index != 0:
        df = df.drop(labels=range(item_index,batch_size))
        save_df(df,batch_index,debug=True)
        batch_index += 1
    print("Searched %s levels and made %s batches." % (total_count, batch_index))

def create_df():
    return pd.DataFrame(index=range(batch_size),columns=["data_id", "gamestyle", "theme", "tag1", "tag2", "timer", "likes", "boos", "level_data"])

def save_df(df: pd.DataFrame, batch_index, debug=False):
    destination = output_folder + batch_prefix + str(batch_index) + ".parquet"
    df.to_parquet(destination)
    if debug:
        print("Saved batch %s to %s" % (batch_index, destination))

if __name__ == "__main__":
    main()