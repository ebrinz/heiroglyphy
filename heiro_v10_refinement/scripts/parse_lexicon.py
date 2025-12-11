"""
Parse HamdiJr Lexicon.txt to build accurate Gardiner -> Transliteration mappings.
Format: Gardiner_Codes,;transliteration;english;frequency;
Example: A1,;i;I, me, my;258.5;
"""

import json
from pathlib import Path
from collections import defaultdict

def parse_lexicon():
    PROJECT_ROOT = Path("heiro_v10_refinement")
    lexicon_path = PROJECT_ROOT / "data/Lexicon.txt"
    
    # Build mapping: transliteration -> list of Gardiner code sequences
    trans_to_codes = defaultdict(set)
    code_to_trans = defaultdict(set)
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(';')
            if len(parts) < 4:
                continue
            
            gardiner_seq = parts[0]  # e.g., "A1," or "A1,A1,A1,"
            transliteration = parts[1]  # e.g., "i" or "rmT"
            english = parts[2]
            frequency = parts[3]
            
            if not transliteration:
                continue
            
            # Clean Gardiner sequence: remove trailing comma
            gardiner_seq = gardiner_seq.rstrip(',')
            
            # Split into individual codes
            codes = [c.strip() for c in gardiner_seq.split(',') if c.strip()]
            
            if codes:
                # Store as tuple for hashability
                codes_tuple = tuple(codes)
                trans_to_codes[transliteration].add(codes_tuple)
                
                # Also store individual codes
                for code in codes:
                    code_to_trans[code].add(transliteration)
    
    # Convert sets to lists for JSON serialization
    trans_to_codes_list = {k: [list(v) for v in vals] for k, vals in trans_to_codes.items()}
    code_to_trans_list = {k: list(v) for k, v in code_to_trans.items()}
    
    print(f"Parsed {len(trans_to_codes_list)} unique transliterations")
    print(f"Parsed {len(code_to_trans_list)} unique Gardiner codes")
    
    # Save mappings
    output_trans = PROJECT_ROOT / "data/lexicon_trans_to_codes.json"
    with open(output_trans, 'w', encoding='utf-8') as f:
        json.dump(trans_to_codes_list, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved transliteration->codes to {output_trans}")
    
    output_code = PROJECT_ROOT / "data/lexicon_code_to_trans.json"
    with open(output_code, 'w', encoding='utf-8') as f:
        json.dump(code_to_trans_list, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved code->transliteration to {output_code}")
    
    # Sample output
    print("\nSample mappings:")
    for i, (trans, codes_list) in enumerate(list(trans_to_codes_list.items())[:20]):
        # Show first code sequence for each
        first_seq = codes_list[0] if codes_list else []
        print(f"  {trans} → {first_seq}")
    
    return trans_to_codes_list, code_to_trans_list

if __name__ == "__main__":
    parse_lexicon()
