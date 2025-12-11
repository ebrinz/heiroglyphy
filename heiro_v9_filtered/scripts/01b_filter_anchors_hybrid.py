"""
V9 Step 1b: Hybrid Anchor Filtering

A more balanced approach:
- Remove LOW-CONFIDENCE function words (< 0.50)
- Keep HIGH-CONFIDENCE function words (they still provide useful signal)
- Keep all content words with confidence >= 0.50

This balances quality vs quantity better than aggressive filtering.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ANCHORS_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/anchors.json"
OUTPUT_PATH = BASE_DIR / "data/processed/filtered_anchors_hybrid.json"
STATS_PATH = BASE_DIR / "data/processed/filter_statistics_hybrid.json"

# Thresholds
MIN_CONFIDENCE_CONTENT = 0.50  # Content words need 50%+ confidence
MIN_CONFIDENCE_FUNCTION = 0.70  # Function words need 70%+ confidence (stricter)

# Function words (be stricter with these)
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

    # Filter anchors with hybrid approach
    filtered = []
    removed_low_conf_content = 0
    removed_low_conf_function = 0
    kept_function = 0
    kept_content = 0

    for anchor in anchors:
        eng = anchor['english'].lower()
        conf = anchor['confidence']
        is_function = eng in FUNCTION_WORDS

        if is_function:
            # Stricter threshold for function words
            if conf >= MIN_CONFIDENCE_FUNCTION:
                filtered.append(anchor)
                kept_function += 1
            else:
                removed_low_conf_function += 1
        else:
            # More lenient for content words
            if conf >= MIN_CONFIDENCE_CONTENT:
                filtered.append(anchor)
                kept_content += 1
            else:
                removed_low_conf_content += 1

    # Sort by confidence (descending)
    filtered.sort(key=lambda x: -x['confidence'])

    # Statistics
    stats = {
        "original_count": len(anchors),
        "filtered_count": len(filtered),
        "kept_content_words": kept_content,
        "kept_function_words": kept_function,
        "removed_low_conf_content": removed_low_conf_content,
        "removed_low_conf_function": removed_low_conf_function,
        "content_threshold": MIN_CONFIDENCE_CONTENT,
        "function_threshold": MIN_CONFIDENCE_FUNCTION,
        "retention_rate": len(filtered) / len(anchors) * 100,
        "avg_confidence_before": sum(a['confidence'] for a in anchors) / len(anchors),
        "avg_confidence_after": sum(a['confidence'] for a in filtered) / len(filtered) if filtered else 0,
    }

    print(f"\nHybrid Filtering Results:")
    print(f"  Original anchors: {stats['original_count']}")
    print(f"  Content words kept (conf >= {MIN_CONFIDENCE_CONTENT}): {kept_content}")
    print(f"  Function words kept (conf >= {MIN_CONFIDENCE_FUNCTION}): {kept_function}")
    print(f"  Removed low-conf content: {removed_low_conf_content}")
    print(f"  Removed low-conf function: {removed_low_conf_function}")
    print(f"  Total remaining: {stats['filtered_count']}")
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

    # Show sample anchors
    print(f"\nTop 10 content word anchors:")
    content_anchors = [a for a in filtered if a['english'].lower() not in FUNCTION_WORDS]
    for i, a in enumerate(content_anchors[:10]):
        print(f"  {i+1:2}. {a['hieroglyphic']:15} -> {a['english']:15} conf={a['confidence']:.2f}")

    print(f"\nTop 10 function word anchors (high confidence):")
    function_anchors = [a for a in filtered if a['english'].lower() in FUNCTION_WORDS]
    for i, a in enumerate(function_anchors[:10]):
        print(f"  {i+1:2}. {a['hieroglyphic']:15} -> {a['english']:15} conf={a['confidence']:.2f}")


if __name__ == "__main__":
    main()
