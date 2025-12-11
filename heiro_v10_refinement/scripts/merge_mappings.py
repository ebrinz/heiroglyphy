"""
Build a comprehensive mapping by merging Wikipedia + Manual mappings.
"""

import json
from pathlib import Path

def merge_mappings():
    PROJECT_ROOT = Path("heiro_v10_refinement")
    
    # Load Wikipedia mapping
    wiki_path = PROJECT_ROOT / "data/gardiner_mapping.json"
    with open(wiki_path, 'r') as f:
        wiki_mapping = json.load(f)
    
    print(f"Wikipedia mappings: {len(wiki_mapping)}")
    
    # Load manual mapping
    manual_path = PROJECT_ROOT / "data/manual_mappings.json"
    with open(manual_path, 'r') as f:
        manual_data = json.load(f)
    
    # Remove comment keys
    manual_mapping = {k: v for k, v in manual_data.items() if not k.startswith('_')}
    print(f"Manual mappings: {len(manual_mapping)}")
    
    # Build reverse mapping: Trans -> Gardiner Codes
    # From Wikipedia: Code -> Trans
    trans_to_codes = {}
    
    # Add Wikipedia mappings
    for code, trans_str in wiki_mapping.items():
        parts = trans_str.replace('(', '').replace(')', '').split(',')
        for part in parts:
            part = part.strip()
            if part:
                if part not in trans_to_codes:
                    trans_to_codes[part] = []
                if code not in trans_to_codes[part]:
                    trans_to_codes[part].append(code)
    
    # Add/Override with manual mappings
    for trans, codes in manual_mapping.items():
        trans_to_codes[trans] = codes
    
    print(f"Total unique transliterations: {len(trans_to_codes)}")
    
    # Save merged mapping
    output_path = PROJECT_ROOT / "data/merged_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trans_to_codes, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to {output_path}")
    
    # Show coverage on top words
    print("\nSample merged mappings:")
    for k, v in list(trans_to_codes.items())[:20]:
        print(f"  {k} → {v}")
    
    return trans_to_codes

if __name__ == "__main__":
    merge_mappings()
