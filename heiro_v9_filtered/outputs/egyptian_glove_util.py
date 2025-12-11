"""
Egyptian-GloVe Utility Module

Finds Egyptian hieroglyphic words that semantically match English concepts.
Works with any GloVe model since Egyptian vectors are pre-aligned to GloVe space.

Usage:
    from egyptian_glove_util import EgyptianLookup

    lookup = EgyptianLookup(
        egyptian_vectors_path="egyptian_aligned_vectors.npy",
        egyptian_vocab_path="egyptian_aligned_vocab.pkl",
        glove_model=your_glove_keyed_vectors  # or path to glove file
    )

    # Single concept
    results = lookup.find("death")

    # Relationship between concepts (vector addition)
    results = lookup.find_relationship(["death", "rebirth"])

    # Analogy: A is to B as C is to ?
    results = lookup.find_analogy("king", "queen", "god")  # god - king + queen

    # Blend with weights
    results = lookup.find_blend({"sun": 0.7, "power": 0.3})
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


class EgyptianLookup:
    def __init__(
        self,
        egyptian_vectors_path: Union[str, Path],
        egyptian_vocab_path: Union[str, Path],
        glove_model  # KeyedVectors or path to .txt file
    ):
        """
        Initialize the Egyptian lookup utility.

        Args:
            egyptian_vectors_path: Path to egyptian_aligned_vectors.npy
            egyptian_vocab_path: Path to egyptian_aligned_vocab.pkl
            glove_model: Either a gensim KeyedVectors object or path to GloVe .txt file
        """
        # Load Egyptian vectors
        self.vectors = np.load(egyptian_vectors_path)

        # Load vocab
        with open(egyptian_vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Reverse vocab for index -> word lookup
        self.idx_to_word = {v: k for k, v in self.vocab.items()}

        # Load or use GloVe
        if isinstance(glove_model, (str, Path)):
            from gensim.models import KeyedVectors
            self.glove = KeyedVectors.load_word2vec_format(
                str(glove_model), binary=False, no_header=True
            )
        else:
            self.glove = glove_model

        print(f"Loaded {len(self.vocab)} Egyptian words, {len(self.glove)} English words")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _get_english_vec(self, word: str) -> Optional[np.ndarray]:
        """Get vector for an English word"""
        if word in self.glove:
            return self.glove[word]
        return None

    def _find_nearest_egyptian(
        self,
        vec: np.ndarray,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Find Egyptian words nearest to a vector"""
        vec = self._normalize(vec)
        sims = self.vectors @ vec
        top_indices = np.argsort(sims)[-topn:][::-1]
        return [(self.idx_to_word[i], float(sims[i])) for i in top_indices]

    def find(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching a single English concept.

        Args:
            word: English word
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity_score) tuples
        """
        vec = self._get_english_vec(word.lower())
        if vec is None:
            return []
        return self._find_nearest_egyptian(vec, topn)

    def find_relationship(
        self,
        words: List[str],
        topn: int = 10,
        operation: str = "add"
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching the relationship between concepts.

        Args:
            words: List of English words to combine
            topn: Number of results
            operation: "add" (sum vectors) or "mean" (average vectors)

        Returns:
            List of (egyptian_word, similarity_score) tuples

        Example:
            # Death + Rebirth (transformation concept)
            find_relationship(["death", "rebirth"])

            # The Lovers card concept
            find_relationship(["love", "choice", "union", "duality"])
        """
        vecs = []
        for w in words:
            vec = self._get_english_vec(w.lower())
            if vec is not None:
                vecs.append(vec)

        if not vecs:
            return []

        if operation == "add":
            combined = np.sum(vecs, axis=0)
        else:  # mean
            combined = np.mean(vecs, axis=0)

        return self._find_nearest_egyptian(combined, topn)

    def find_analogy(
        self,
        a: str,
        b: str,
        c: str,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching an analogy: A is to B as C is to ?

        Uses vector arithmetic: result = C - A + B

        Args:
            a, b, c: English words forming the analogy
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity_score) tuples

        Example:
            # king is to queen as god is to ? (goddess)
            find_analogy("king", "queen", "god")

            # sun is to day as moon is to ? (night)
            find_analogy("sun", "day", "moon")
        """
        vec_a = self._get_english_vec(a.lower())
        vec_b = self._get_english_vec(b.lower())
        vec_c = self._get_english_vec(c.lower())

        if any(v is None for v in [vec_a, vec_b, vec_c]):
            return []

        # c - a + b
        result = vec_c - vec_a + vec_b
        return self._find_nearest_egyptian(result, topn)

    def find_blend(
        self,
        word_weights: Dict[str, float],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching a weighted blend of concepts.

        Args:
            word_weights: Dict mapping English words to weights
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity_score) tuples

        Example:
            # Mostly sun energy with some death/transformation
            find_blend({"sun": 0.7, "death": 0.2, "rebirth": 0.1})

            # Two tarot cards blended
            find_blend({"tower": 0.5, "star": 0.5})  # destruction + hope
        """
        combined = np.zeros(self.glove.vector_size)
        total_weight = 0

        for word, weight in word_weights.items():
            vec = self._get_english_vec(word.lower())
            if vec is not None:
                combined += weight * vec
                total_weight += weight

        if total_weight == 0:
            return []

        combined /= total_weight
        return self._find_nearest_egyptian(combined, topn)

    def find_contrast(
        self,
        positive: List[str],
        negative: List[str],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching positive concepts minus negative concepts.

        Args:
            positive: Words to include
            negative: Words to subtract
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity_score) tuples

        Example:
            # Power without destruction
            find_contrast(["power", "strength"], ["destruction", "violence"])

            # Love without loss
            find_contrast(["love", "union"], ["loss", "separation"])
        """
        pos_vecs = []
        for w in positive:
            vec = self._get_english_vec(w.lower())
            if vec is not None:
                pos_vecs.append(vec)

        neg_vecs = []
        for w in negative:
            vec = self._get_english_vec(w.lower())
            if vec is not None:
                neg_vecs.append(vec)

        if not pos_vecs:
            return []

        combined = np.mean(pos_vecs, axis=0)
        if neg_vecs:
            combined -= np.mean(neg_vecs, axis=0)

        return self._find_nearest_egyptian(combined, topn)

    def find_midpoint(
        self,
        word1: str,
        word2: str,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words at the semantic midpoint between two concepts.

        Args:
            word1, word2: Two English words
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity_score) tuples

        Example:
            # Between life and death
            find_midpoint("life", "death")

            # Between sun and moon
            find_midpoint("sun", "moon")
        """
        return self.find_relationship([word1, word2], topn, operation="mean")

    def get_egyptian_vector(self, egyptian_word: str) -> Optional[np.ndarray]:
        """Get the aligned vector for an Egyptian word"""
        if egyptian_word in self.vocab:
            return self.vectors[self.vocab[egyptian_word]]
        return None

    def egyptian_similarity(self, word1: str, word2: str) -> float:
        """Compute similarity between two Egyptian words"""
        vec1 = self.get_egyptian_vector(word1)
        vec2 = self.get_egyptian_vector(word2)
        if vec1 is None or vec2 is None:
            return 0.0
        return float(np.dot(self._normalize(vec1), self._normalize(vec2)))


# Example usage
if __name__ == "__main__":
    # Demo with local files
    lookup = EgyptianLookup(
        egyptian_vectors_path="egyptian_aligned_vectors.npy",
        egyptian_vocab_path="egyptian_aligned_vocab.pkl",
        glove_model="/Users/crashy/Development/heiroglyphy/heiro_v5_getdata/data/processed/glove.6B.300d.txt"
    )

    print("\n=== Single Concept ===")
    print("'death':", lookup.find("death", topn=5))

    print("\n=== Relationship (Death + Rebirth) ===")
    print(lookup.find_relationship(["death", "rebirth"], topn=5))

    print("\n=== Analogy (king:queen :: god:?) ===")
    print(lookup.find_analogy("king", "queen", "god", topn=5))

    print("\n=== Blend (Tower + Star cards) ===")
    print(lookup.find_blend({"destruction": 0.5, "hope": 0.5}, topn=5))

    print("\n=== Midpoint (life <-> death) ===")
    print(lookup.find_midpoint("life", "death", topn=5))
