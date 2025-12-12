#!/usr/bin/env python3
"""
Create expanded assets for final_output:
1. Expanded concept vectors (300+ useful English concepts for lookups)
2. Hieroglyph mapping file (hieroglyph → transliteration → English)
"""

import json
import csv
import pickle
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors

REPO_ROOT = Path(__file__).parent

# ============================================
# PART 1: Expanded Concept Vectors
# ============================================

# Comprehensive concept list organized by category
CONCEPTS = {
    # Nature & Elements
    "elements": [
        "fire", "water", "earth", "air", "light", "darkness", "shadow",
        "wind", "storm", "rain", "flood", "drought", "heat", "cold"
    ],
    "celestial": [
        "sun", "moon", "star", "sky", "heaven", "cosmos", "universe",
        "dawn", "dusk", "night", "day", "horizon", "eclipse", "constellation"
    ],
    "geography": [
        "river", "nile", "mountain", "desert", "valley", "oasis", "sea",
        "ocean", "lake", "island", "field", "land", "shore", "bank"
    ],
    "plants": [
        "tree", "flower", "lotus", "papyrus", "reed", "grain", "wheat",
        "barley", "vine", "fruit", "seed", "root", "leaf", "wood"
    ],
    "animals": [
        "lion", "snake", "cobra", "crocodile", "hippopotamus", "bull", "cow",
        "ram", "goat", "dog", "cat", "bird", "falcon", "ibis", "vulture",
        "scarab", "beetle", "fish", "frog", "jackal", "baboon", "monkey"
    ],

    # Divine & Spiritual
    "deities": [
        "god", "goddess", "divine", "sacred", "holy", "deity", "immortal",
        "eternal", "creator", "destroyer", "protector", "guardian"
    ],
    "afterlife": [
        "death", "life", "rebirth", "resurrection", "soul", "spirit",
        "afterlife", "underworld", "paradise", "judgment", "eternity",
        "mummy", "tomb", "coffin", "burial", "funeral"
    ],
    "magic": [
        "magic", "spell", "curse", "blessing", "amulet", "charm", "ritual",
        "ceremony", "offering", "sacrifice", "prayer", "incantation"
    ],

    # Society & Power
    "royalty": [
        "king", "queen", "pharaoh", "prince", "princess", "ruler", "lord",
        "throne", "crown", "scepter", "palace", "dynasty", "reign"
    ],
    "titles": [
        "priest", "priestess", "scribe", "vizier", "governor", "official",
        "servant", "slave", "master", "overseer", "commander", "general"
    ],
    "society": [
        "people", "man", "woman", "child", "elder", "family", "mother",
        "father", "son", "daughter", "brother", "sister", "wife", "husband"
    ],

    # Abstract Concepts
    "virtues": [
        "truth", "justice", "order", "balance", "harmony", "peace",
        "wisdom", "knowledge", "power", "strength", "courage", "honor"
    ],
    "emotions": [
        "love", "hate", "fear", "joy", "sorrow", "anger", "hope",
        "desire", "passion", "happiness", "grief", "terror"
    ],
    "states": [
        "life", "death", "birth", "transformation", "change", "growth",
        "decay", "beginning", "ending", "creation", "destruction"
    ],

    # Actions & Objects
    "actions": [
        "speak", "see", "hear", "eat", "drink", "sleep", "wake",
        "walk", "run", "fight", "kill", "heal", "build", "make",
        "give", "take", "open", "close", "rise", "fall"
    ],
    "body": [
        "head", "face", "eye", "mouth", "ear", "nose", "hand", "arm",
        "foot", "leg", "heart", "blood", "bone", "flesh", "body"
    ],
    "objects": [
        "house", "temple", "pyramid", "obelisk", "statue", "boat", "ship",
        "chariot", "weapon", "sword", "spear", "bow", "arrow", "shield",
        "bread", "beer", "wine", "gold", "silver", "copper", "stone"
    ],

    # Time & Numbers
    "time": [
        "time", "year", "month", "day", "hour", "moment", "forever",
        "past", "present", "future", "ancient", "eternal"
    ],
    "numbers": [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "hundred", "thousand", "million", "first", "last"
    ],

    # Egyptian-specific
    "egyptian": [
        "egypt", "nile", "pyramid", "sphinx", "obelisk", "hieroglyph",
        "cartouche", "scarab", "ankh", "uraeus", "crook", "flail"
    ]
}

def create_concept_vectors():
    """Create expanded concept vectors from GloVe."""
    print("=" * 60)
    print("CREATING EXPANDED CONCEPT VECTORS")
    print("=" * 60)

    # Load GloVe
    glove_path = REPO_ROOT / 'heiro_v5_getdata/data/processed/glove.6B.300d.txt'
    print(f"\nLoading GloVe from {glove_path}...")
    glove = KeyedVectors.load_word2vec_format(str(glove_path), binary=False, no_header=True)
    print(f"  ✓ Loaded {len(glove)} English embeddings")

    # Collect all concepts
    all_concepts = []
    for category, words in CONCEPTS.items():
        all_concepts.extend(words)

    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for word in all_concepts:
        if word not in seen:
            seen.add(word)
            unique_concepts.append(word)

    print(f"\n  Total unique concepts: {len(unique_concepts)}")

    # Get vectors for concepts that exist in GloVe
    valid_words = []
    valid_vectors = []
    missing = []

    for word in unique_concepts:
        if word in glove:
            valid_words.append(word)
            valid_vectors.append(glove[word])
        else:
            missing.append(word)

    print(f"  Found in GloVe: {len(valid_words)}")
    if missing:
        print(f"  Missing ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Convert to arrays
    vectors = np.array(valid_vectors, dtype=np.float16)
    words = np.array(valid_words)

    # Save
    output_path = REPO_ROOT / 'final_output/concept_vectors.npz'
    np.savez_compressed(output_path, vectors=vectors, words=words)
    print(f"\n  ✓ Saved {len(words)} concept vectors to {output_path}")

    # Also save category metadata
    category_map = {}
    for category, cat_words in CONCEPTS.items():
        category_map[category] = [w for w in cat_words if w in valid_words]

    meta_path = REPO_ROOT / 'final_output/concept_categories.json'
    with open(meta_path, 'w') as f:
        json.dump(category_map, f, indent=2)
    print(f"  ✓ Saved category metadata to {meta_path}")

    return len(words)


# ============================================
# PART 2: Hieroglyph Mapping File
# ============================================

def parse_lexicon_line(line):
    """Parse a line from the HamdiJr Lexicon.txt file."""
    # Format: CODES,;transliteration;english;occurrence;
    parts = line.strip().split(';')
    if len(parts) >= 3:
        codes_part = parts[0].rstrip(',')
        transliteration = parts[1].strip()
        english = parts[2].strip()
        occurrence = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0

        # Parse Gardiner codes (comma-separated)
        codes = [c.strip() for c in codes_part.split(',') if c.strip()]

        return {
            'gardiner_codes': codes,
            'transliteration': transliteration,
            'english': english,
            'occurrence': occurrence
        }
    return None

def create_hieroglyph_mapping():
    """Create comprehensive hieroglyph mapping file."""
    print("\n" + "=" * 60)
    print("CREATING HIEROGLYPH MAPPING FILE")
    print("=" * 60)

    # Load Unicode → Character mapping
    unicode_map = {}
    lexicon_csv = REPO_ROOT / 'heiro_v6_BERT/data/processed/hieroglyph_lexicon.csv'
    print(f"\nLoading Unicode mapping from {lexicon_csv}...")

    with open(lexicon_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gardiner = row['glyph_name'].upper()
            unicode_map[gardiner] = {
                'unicode': row['unicode'],
                'character': row['character']
            }
    print(f"  ✓ Loaded {len(unicode_map)} Unicode mappings")

    # Load HamdiJr Lexicon
    lexicon_path = REPO_ROOT / 'heiro_v10_refinement/data/Lexicon.txt'
    print(f"\nParsing lexicon from {lexicon_path}...")

    entries = []
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parsed = parse_lexicon_line(line)
                if parsed and parsed['transliteration'] and parsed['english']:
                    entries.append(parsed)

    print(f"  ✓ Parsed {len(entries)} lexicon entries")

    # Build mapping with hieroglyph characters
    mapping = []
    for entry in entries:
        # Get hieroglyph characters for the Gardiner codes
        hieroglyphs = []
        for code in entry['gardiner_codes']:
            code_upper = code.upper()
            if code_upper in unicode_map:
                hieroglyphs.append(unicode_map[code_upper]['character'])

        hieroglyph_str = ''.join(hieroglyphs) if hieroglyphs else ''

        mapping.append({
            'hieroglyph': hieroglyph_str,
            'gardiner_codes': ','.join(entry['gardiner_codes']),
            'transliteration': entry['transliteration'],
            'english': entry['english'],
            'occurrence': entry['occurrence']
        })

    # Sort by occurrence (most common first)
    mapping.sort(key=lambda x: x['occurrence'], reverse=True)

    # Save as TSV (tab-separated for easy viewing)
    output_path = REPO_ROOT / 'final_output/hieroglyph_dictionary.tsv'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('hieroglyph\tgardiner_codes\ttransliteration\tenglish\toccurrence\n')
        for entry in mapping:
            f.write(f"{entry['hieroglyph']}\t{entry['gardiner_codes']}\t{entry['transliteration']}\t{entry['english']}\t{entry['occurrence']}\n")

    print(f"\n  ✓ Saved {len(mapping)} entries to {output_path}")

    # Also save as JSON for programmatic access
    json_path = REPO_ROOT / 'final_output/hieroglyph_dictionary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved JSON version to {json_path}")

    # Show sample
    print("\n  Sample entries (top 20 by occurrence):")
    for entry in mapping[:20]:
        print(f"    {entry['hieroglyph']:6} {entry['transliteration']:15} → {entry['english'][:40]}")

    return len(mapping)


def main():
    print("\n" + "=" * 60)
    print("CREATING EXPANDED FINAL OUTPUT ASSETS")
    print("=" * 60)

    n_concepts = create_concept_vectors()
    n_mappings = create_hieroglyph_mapping()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nCreated:")
    print(f"  - concept_vectors.npz ({n_concepts} concepts)")
    print(f"  - concept_categories.json (category metadata)")
    print(f"  - hieroglyph_dictionary.tsv ({n_mappings} entries)")
    print(f"  - hieroglyph_dictionary.json (programmatic access)")


if __name__ == '__main__':
    main()
