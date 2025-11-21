import pandas as pd
from pathlib import Path

# Define path
PARQUET_PATH = Path("heiro_v6_BERT/data/raw/bbaw_huggingface.parquet")

def inspect_parquet():
    if not PARQUET_PATH.exists():
        print(f"Error: {PARQUET_PATH} not found.")
        return

    print(f"Reading {PARQUET_PATH}...")
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print("Columns:", df.columns.tolist())
        print(f"Shape: {df.shape}")
        
        if 'hieroglyphs' in df.columns:
            # Count non-empty hieroglyphs
            non_empty = df[df['hieroglyphs'] != ''].shape[0]
            print(f"Rows with hieroglyphs: {non_empty} / {df.shape[0]}")
            
            # Show a sample
            print("\nSample with hieroglyphs:")
            print(df[df['hieroglyphs'] != ''].head(3)[['transcription', 'hieroglyphs']])
        else:
            print("'hieroglyphs' column not found.")
            
    except Exception as e:
        print(f"Failed to read parquet: {e}")

if __name__ == "__main__":
    inspect_parquet()
