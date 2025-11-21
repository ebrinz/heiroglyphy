#!/usr/bin/env python3
"""
Fix German anchor translations using a dictionary approach for common words.
DeepL doesn't translate single words well without context.
"""

import json
import pickle
from pathlib import Path

# Common German→English dictionary
GERMAN_TO_ENGLISH = {
    'der': 'the',
    'die': 'the',
    'das': 'the',
    'und': 'and',
    'ist': 'is',
    'nicht': 'not',
    'ich': 'i',
    'du': 'you',
    'er': 'he',
    'sie': 'she',
    'es': 'it',
    'wir': 'we',
    'ihr': 'you',
    'von': 'of',
    'zu': 'to',
    'auf': 'on',
    'mit': 'with',
    'für': 'for',
    'an': 'at',
    'bei': 'at',
    'nach': 'after',
    'vor': 'before',
    'über': 'over',
    'unter': 'under',
    'durch': 'through',
    'gegen': 'against',
    'ohne': 'without',
    'um': 'around',
    'zwischen': 'between',
    'während': 'during',
    'wegen': 'because',
    'trotz': 'despite',
    'seit': 'since',
    'bis': 'until',
    'als': 'as',
    'wenn': 'if',
    'ob': 'whether',
    'dass': 'that',
    'weil': 'because',
    'damit': 'so that',
    'obwohl': 'although',
    'gott': 'god',
    'könig': 'king',
    'sohn': 'son',
    'vater': 'father',
    'mutter': 'mother',
    'bruder': 'brother',
    'schwester': 'sister',
    'mann': 'man',
    'frau': 'woman',
    'kind': 'child',
    'leben': 'life',
    'tod': 'death',
    'himmel': 'heaven',
    'erde': 'earth',
    'wasser': 'water',
    'feuer': 'fire',
    'luft': 'air',
    'stein': 'stone',
    'holz': 'wood',
    'gold': 'gold',
    'silber': 'silver',
    'groß': 'great',
    'klein': 'small',
    'gut': 'good',
    'böse': 'evil',
    'schön': 'beautiful',
    'hässlich': 'ugly',
    'alt': 'old',
    'jung': 'young',
    'neu': 'new',
    'alt': 'old',
}

def main():
    # Load anchors
    with open('data/processed/english_anchors.json', 'r') as f:
        anchors = json.load(f)
    
    print(f"Loaded {len(anchors):,} anchors")
    
    # Fix translations
    fixed = 0
    for anchor in anchors:
        german = anchor.get('german', '').lower()
        english = anchor['english'].lower()
        
        # If English == German and we have a translation, fix it
        if english == german and german in GERMAN_TO_ENGLISH:
            anchor['english'] = GERMAN_TO_ENGLISH[german]
            fixed += 1
    
    print(f"Fixed {fixed:,} German words")
    
    # Save
    with open('data/processed/english_anchors.json', 'w') as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)
    
    with open('data/processed/english_anchors.pkl', 'wb') as f:
        pickle.dump(anchors, f)
    
    print("✓ Saved fixed anchors")
    
    # Verify
    german_count = sum(1 for a in anchors if a['english'] in ['der', 'die', 'das', 'und', 'ist', 'ich', 'du'])
    print(f"\nRemaining German words: {german_count:,}")

if __name__ == "__main__":
    main()
