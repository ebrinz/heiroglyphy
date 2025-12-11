"""
Script to execute v11 Phase 1.2: Batch Translation
"""
import os
import json
import pandas as pd
import deepl
from pathlib import Path
from tqdm.auto import tqdm

def run_translation():
    # Paths
    PROJECT_ROOT = Path("heiro_v11")
    V11_DATA = PROJECT_ROOT / 'data'
    INPUT_CSV = V11_DATA / 'processed/anchors_to_translate.csv'
    OUTPUT_JSON = V11_DATA / 'processed/v11_anchors.json'
    
    # API Key
    DEEPL_KEY = os.getenv('DEEPL_API_KEY')
    if not DEEPL_KEY:
        DEEPL_KEY = 'DEEPL_API_KEY'
    
    print(f'Project Root: {PROJECT_ROOT}')
    
    # 1. Load Candidates
    df = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df)} candidates')
    
    # 2. Initialize DeepL
    try:
        translator = deepl.Translator(DEEPL_KEY)
        usage = translator.get_usage()
        print(f'✓ DeepL Connected. Usage: {usage.character.count}/{usage.character.limit}')
    except Exception as e:
        print(f'✗ Failed to connect: {e}')
        return

    # 3. Batch Translate
    texts_to_translate = df['current_translation'].fillna('').astype(str).tolist()
    translations = []
    BATCH_SIZE = 50
    
    print(f'Translating {len(texts_to_translate)} texts...')
    
    for i in tqdm(range(0, len(texts_to_translate), BATCH_SIZE)):
        batch = texts_to_translate[i:i+BATCH_SIZE]
        
        try:
            # Check if batch has content
            if any(t.strip() for t in batch):
                results = translator.translate_text(batch, target_lang='EN-US')
                translations.extend([r.text for r in results])
            else:
                translations.extend(batch) # All empty
        except Exception as e:
            print(f'Error in batch {i}: {e}')
            translations.extend(batch)
            
    df['english'] = translations
    print('✓ Translation complete')
    
    # 4. Save Clean Anchors
    clean_anchors = []
    for _, row in df.iterrows():
        if row['english'] and row['english'].strip():
            clean_anchors.append({
                'hieroglyphic': row['hieroglyphic'],
                'english': row['english'].lower().strip(),
                'source': row['source']
            })
            
    print(f'Retained {len(clean_anchors)} valid anchors')
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(clean_anchors, f, indent=2)
        
    print(f'✓ Saved to {OUTPUT_JSON}')

if __name__ == "__main__":
    run_translation()
