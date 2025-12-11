"""
V9 Step 1: Filter Anchors

Removes function words and low-confidence anchors from the anchor dictionary.
This addresses the "function word pollution" issue identified in V7 analysis.

Filters:
- Remove function words (articles, pronouns, prepositions, etc.)
- Keep only anchors with confidence >= 0.60
- Output: filtered_anchors.json
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ANCHORS_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/anchors.json"
OUTPUT_PATH = BASE_DIR / "data/processed/filtered_anchors.json"
STATS_PATH = BASE_DIR / "data/processed/filter_statistics.json"

# Confidence threshold
MIN_CONFIDENCE = 0.60

# Function words to filter out (English and German since some anchors use German)
FUNCTION_WORDS = {
    # Articles
    'the', 'a', 'an',
    # Pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'who', 'whom', 'which', 'what', 'whose',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
    # Be verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    # Auxiliary verbs
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'shall',
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'under', 'over',
    'between', 'among', 'upon',
    # Conjunctions
    'and', 'but', 'or', 'nor', 'so', 'yet', 'if', 'then', 'because', 'although',
    'while', 'when', 'where', 'as', 'than',
    # Other function words
    'not', 'no', 'yes', 'there', 'here', 'also', 'only', 'just', 'even', 'very',
    'too', 'now', 'then',
    # German function words
    'der', 'die', 'das', 'er', 'sie', 'es', 'ich', 'du', 'wir', 'ihr', 'sein',
    'ist', 'sind', 'war', 'nicht', 'wie', 'man', 'zu', 'und', 'auf', 'im', 'dem',
    'den', 'ein', 'eine', 'einer', 'des', 'dem',
}


def main():
    print(f"Loading anchors from {ANCHORS_PATH}...")
    with open(ANCHORS_PATH, 'r') as f:
        anchors = json.load(f)

    print(f"Total anchors: {len(anchors)}")

    # Filter anchors
    filtered = []
    removed_low_conf = 0
    removed_function = 0

    for anchor in anchors:
        eng = anchor['english'].lower()
        conf = anchor['confidence']

        # Check confidence threshold
        if conf < MIN_CONFIDENCE:
            removed_low_conf += 1
            continue

        # Check function words
        if eng in FUNCTION_WORDS:
            removed_function += 1
            continue

        filtered.append(anchor)

    # Sort by confidence (descending)
    filtered.sort(key=lambda x: -x['confidence'])

    # Statistics
    stats = {
        "original_count": len(anchors),
        "filtered_count": len(filtered),
        "removed_low_confidence": removed_low_conf,
        "removed_function_words": removed_function,
        "confidence_threshold": MIN_CONFIDENCE,
        "retention_rate": len(filtered) / len(anchors) * 100,
        "avg_confidence_before": sum(a['confidence'] for a in anchors) / len(anchors),
        "avg_confidence_after": sum(a['confidence'] for a in filtered) / len(filtered) if filtered else 0,
    }

    print(f"\nFiltering Results:")
    print(f"  Original anchors: {stats['original_count']}")
    print(f"  Removed (low confidence < {MIN_CONFIDENCE}): {removed_low_conf}")
    print(f"  Removed (function words): {removed_function}")
    print(f"  Remaining anchors: {stats['filtered_count']}")
    print(f"  Retention rate: {stats['retention_rate']:.1f}%")
    print(f"  Avg confidence before: {stats['avg_confidence_before']:.3f}")
    print(f"  Avg confidence after: {stats['avg_confidence_after']:.3f}")

    # Save filtered anchors
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"\nSaved filtered anchors to {OUTPUT_PATH}")

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {STATS_PATH}")

    # Show top 20 anchors
    print(f"\nTop 20 filtered anchors:")
    for i, a in enumerate(filtered[:20]):
        print(f"  {i+1:2}. {a['hieroglyphic']:15} -> {a['english']:15} conf={a['confidence']:.2f} freq={a['frequency']}")


if __name__ == "__main__":
    main()
