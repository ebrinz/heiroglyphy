"""
Vocabulary Normalization Utilities

This module provides functions to normalize Ancient Egyptian transliterations
by removing grammatical markers, suffixes, and inflections to improve matching
with the Gardiner code mappings.
"""

import re
from typing import Dict, List, Tuple

def normalize_word(word: str) -> List[str]:
    """
    Normalize a transliterated Egyptian word by removing grammatical markers.
    
    Returns a list of normalized variants to try for mapping.
    
    Args:
        word: Original transliterated word (e.g., '=f', 'n(,j)', 'jri̯.n')
        
    Returns:
        List of normalized variants to try, ordered by likelihood
        
    Examples:
        normalize_word('=f') → ['f', '=f']
        normalize_word('n(,j)') → ['n', 'n(,j)']
        normalize_word('jri̯.n') → ['jri̯', 'jri̯.n']
        normalize_word('[m]') → ['m', '[m]']
    """
    variants = []
    
    # Original word
    original = word
    
    # Remove leading '=' (suffix markers)
    if word.startswith('='):
        variants.append(word[1:])
    
    # Remove parenthetical content (grammatical notes)
    if '(' in word:
        cleaned = re.sub(r'\([^)]*\)', '', word)
        if cleaned and cleaned != word:
            variants.append(cleaned)
    
    # Remove square brackets (editorial marks)
    if '[' in word or ']' in word:
        cleaned = word.replace('[', '').replace(']', '')
        if cleaned and cleaned != word:
            variants.append(cleaned)
    
    # Remove inflection markers (. followed by letters)
    if '.' in word:
        # Try removing everything after dot
        parts = word.split('.')
        if parts[0] and parts[0] != word:
            variants.append(parts[0])
    
    # Add original last (fallback)
    if original not in variants:
        variants.append(original)
    
    return variants

def build_normalized_mapping(
    original_mapping: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build a normalized mapping from Gardiner codes to transliterations.
    
    Args:
        original_mapping: Dict of {gardiner_code: transliteration_string}
        
    Returns:
        Tuple of:
        - trans_to_code: Dict mapping transliteration → gardiner_code
        - trans_variants: Dict mapping normalized_form → list of gardiner_codes
    """
    trans_to_code = {}
    trans_variants = {}
    
    for code, trans_str in original_mapping.items():
        # Split by comma or semicolon
        parts = re.split(r'[,;]', trans_str)
        
        for part in parts:
            # Clean up parentheses and whitespace
            clean_part = re.sub(r'\(.*?\)', '', part).strip()
            
            if clean_part:
                # Store direct mapping
                if clean_part not in trans_to_code:
                    trans_to_code[clean_part] = code
                
                # Store in variants list
                if clean_part not in trans_variants:
                    trans_variants[clean_part] = []
                if code not in trans_variants[clean_part]:
                    trans_variants[clean_part].append(code)
    
    return trans_to_code, trans_variants

def lookup_with_normalization(
    word: str,
    trans_to_code: Dict[str, str],
    trans_variants: Dict[str, List[str]]
) -> List[str]:
    """
    Lookup a word in the mapping, trying normalized variants.
    
    Args:
        word: Word to look up
        trans_to_code: Direct transliteration → code mapping
        trans_variants: Normalized → list of codes mapping
        
    Returns:
        List of Gardiner codes (may be empty if no match found)
    """
    # Try direct lookup first
    if word in trans_variants:
        return trans_variants[word]
    
    # Try normalized variants
    variants = normalize_word(word)
    for variant in variants:
        if variant in trans_variants:
            return trans_variants[variant]
    
    return []

if __name__ == "__main__":
    # Test normalization
    test_words = [
        '=f', '=k', '=j', '=s', '=sn', '=tn',
        'n(,j)', '(j)ḫ,t', '(ḥr)',
        'jri̯.n', 'jri̯.t', 'k.t', 's,t',
        '[m]', '[n]', '=[f]', '=(j)'
    ]
    
    print("Normalization Test:")
    print("-" * 60)
    print(f"{'Original':<15} → {'Normalized Variants'}")
    print("-" * 60)
    
    for word in test_words:
        variants = normalize_word(word)
        print(f"{word:<15} → {variants}")
