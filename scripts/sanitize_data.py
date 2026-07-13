import polars as pl
import glob
import os

files = glob.glob("data/*.parquet")
for f in files:
    try:
        # Load the whole file into RAM to break the file lock
        df = pl.read_parquet(f)
        
        # 1. Remove rows with 0 or NaN in price
        # 2. Add an epsilon to close to prevent division by zero
        clean_df = df.filter(
            (pl.col("close") > 0.0001) & 
            (pl.col("high") > 0.0001) & 
            (pl.col("low") > 0.0001)
        ).fill_nan(0).fill_null(0)
        
        # Write to a temporary file first, then replace (Atomic Move)
        temp_file = f + ".tmp"
        clean_df.write_parquet(temp_file)
        os.remove(f)
        os.rename(temp_file, f)
        
        print(f"Sanitized {f}: {len(df) - len(clean_df)} bad rows removed.")
    except Exception as e:
        print(f"Could not sanitize {f}: {e}")