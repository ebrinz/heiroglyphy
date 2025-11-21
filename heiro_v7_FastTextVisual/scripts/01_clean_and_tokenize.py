import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/raw/bbaw_huggingface.parquet"
CLEAN_DATA_PATH = BASE_DIR / "data/processed/cleaned_corpus.txt"

# Ensure output directory exists
CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not PARQUET_PATH.exists():
        print(f"Error: {PARQUET_PATH} not found.")
        return

    print(f"Reading from {PARQUET_PATH}...")
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    print(f"Total rows: {len(df)}")
    
    # Filter for rows with non-empty hieroglyphs
    if 'hieroglyphs' not in df.columns:
        print("Error: 'hieroglyphs' column not found.")
        return
        
    df_glyphs = df[df['hieroglyphs'] != '']
    print(f"Rows with hieroglyphs: {len(df_glyphs)}")
    
    cleaned_lines = []
    
    print("Processing hieroglyphs...")
    for glyph_str in tqdm(df_glyphs['transcription']):
        if not isinstance(glyph_str, str):
            continue
            
        # The glyph string might contain MdC codes or Unicode.
        # Based on the README, it says "Encoding of the hieroglyphs with the Gardiner's sign list"
        # But the sample showed "D21 :Q3 :D36..." which is MdC-like.
        # However, we also saw Unicode in the sample? 
        # Let's look at the sample again: "D21 :Q3 :D36 F4 :D36 L2 -X1 :S19 S29 -U23 -T21..."
        # This is MdC (Manuel de Codage).
        
        # Wait, if it's MdC, FastText needs space-separated tokens.
        # MdC uses space, -, :, *, etc. as separators.
        # We should probably just keep it as is for now, or maybe normalize separators to spaces?
        # FastText splits on whitespace.
        # If we want to learn embeddings for "D21", "Q3", etc., we need them to be tokens.
        # So we should replace -, :, * with spaces.
        
        # Let's refine the cleaning:
        # 1. Replace common MdC separators with space
        clean_line = glyph_str.replace('-', ' ').replace(':', ' ').replace('*', ' ').replace('&', ' ')
        
        # 2. Remove brackets and other markers if we want pure glyphs?
        # The README mentions [], (), {}, <>, etc.
        # Let's keep it simple for now and just tokenize on separators.
        
        # 3. Normalize whitespace
        clean_line = " ".join(clean_line.split())
        
        if clean_line:
            cleaned_lines.append(clean_line)

    print(f"Extracted {len(cleaned_lines)} lines.")
    
    with open(CLEAN_DATA_PATH, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + "\n")

    print(f"Saved cleaned corpus to {CLEAN_DATA_PATH}")

if __name__ == "__main__":
    main()
