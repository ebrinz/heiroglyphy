"""
Egyptian Hieroglyphic Semantic Lookup

Maps English concepts to semantically related Egyptian hieroglyphic words.
Egyptian vectors are pre-aligned to GloVe 300d space, enabling direct
vector arithmetic between English and Egyptian vocabularies.

Requirements:
    - numpy
    - gensim (for loading GloVe)
    - pickle (standard library)

Files Required:
    - egyptian_aligned_vectors.npy (92 MB) - 80,662 Egyptian words in GloVe space
    - egyptian_aligned_vocab.pkl (1.5 MB) - word -> index mapping

Usage:
    from egyptian_lookup import EgyptianLookup

    lookup = EgyptianLookup(
        vectors_path="egyptian_aligned_vectors.npy",
        vocab_path="egyptian_aligned_vocab.pkl",
        glove=your_glove_model  # gensim KeyedVectors or path to .txt
    )

    # Find Egyptian words for a concept
    results = lookup.find("sun")

    # Combine multiple concepts
    results = lookup.find_relationship(["power", "wisdom"])

    # Vector analogy: A is to B as C is to ?
    results = lookup.find_analogy("king", "queen", "god")
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


class EgyptianLookup:
    """
    Semantic lookup for Egyptian hieroglyphic words using GloVe-aligned vectors.

    The Egyptian vocabulary consists of transliterated hieroglyphic words from
    ancient Egyptian texts (primarily religious and funerary literature).
    """

    def __init__(
        self,
        vectors_path: Union[str, Path],
        vocab_path: Union[str, Path],
        glove  # KeyedVectors instance or path to GloVe .txt file
    ):
        """
        Initialize the lookup utility.

        Args:
            vectors_path: Path to egyptian_aligned_vectors.npz (or .npy)
            vocab_path: Path to egyptian_aligned_vocab.pkl
            glove: gensim KeyedVectors object, or path to GloVe text file
        """
        # Support both .npz (compressed) and .npy formats
        vectors_path = Path(vectors_path)
        if vectors_path.suffix == '.npz':
            data = np.load(vectors_path)
            self.vectors = data['vectors'].astype(np.float32)
        else:
            self.vectors = np.load(vectors_path)

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.idx_to_word = {v: k for k, v in self.vocab.items()}

        if isinstance(glove, (str, Path)):
            from gensim.models import KeyedVectors
            self.glove = KeyedVectors.load_word2vec_format(
                str(glove), binary=False, no_header=True
            )
        else:
            self.glove = glove

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _get_glove_vec(self, word: str) -> Optional[np.ndarray]:
        """Get GloVe vector for an English word."""
        word = word.lower()
        return self.glove[word] if word in self.glove else None

    def _search(self, vec: np.ndarray, topn: int) -> List[Tuple[str, float]]:
        """Find nearest Egyptian words to a vector."""
        vec = self._normalize(vec)
        sims = self.vectors @ vec
        indices = np.argsort(sims)[-topn:][::-1]
        return [(self.idx_to_word[i], float(sims[i])) for i in indices]

    def find(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching a single English concept.

        Args:
            word: English word to look up
            topn: Number of results to return

        Returns:
            List of (egyptian_word, similarity) tuples, sorted by similarity
        """
        vec = self._get_glove_vec(word)
        return self._search(vec, topn) if vec is not None else []

    def find_relationship(
        self,
        words: List[str],
        topn: int = 10,
        aggregate: str = "sum"
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching combined concepts.

        Combines multiple English word vectors to find Egyptian words
        that relate to the overall semantic space of the concepts.

        Args:
            words: List of English words to combine
            topn: Number of results
            aggregate: "sum" or "mean" - how to combine vectors

        Returns:
            List of (egyptian_word, similarity) tuples
        """
        vecs = [self._get_glove_vec(w) for w in words]
        vecs = [v for v in vecs if v is not None]

        if not vecs:
            return []

        combined = np.sum(vecs, axis=0) if aggregate == "sum" else np.mean(vecs, axis=0)
        return self._search(combined, topn)

    def find_analogy(
        self,
        a: str,
        b: str,
        c: str,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words via analogy: A is to B as C is to ?

        Uses vector arithmetic: result = C - A + B

        Args:
            a: First term of the analogy
            b: Second term (what A relates to)
            c: Third term (find what this relates to like A->B)
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity) tuples

        Example:
            find_analogy("king", "queen", "god")  # finds goddess-like concepts
            find_analogy("sun", "day", "moon")    # finds night-like concepts
        """
        vec_a, vec_b, vec_c = [self._get_glove_vec(w) for w in [a, b, c]]

        if any(v is None for v in [vec_a, vec_b, vec_c]):
            return []

        result = vec_c - vec_a + vec_b
        return self._search(result, topn)

    def find_blend(
        self,
        weights: Dict[str, float],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching a weighted blend of concepts.

        Args:
            weights: Dict mapping English words to weights (e.g., {"sun": 0.7, "moon": 0.3})
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity) tuples
        """
        combined = np.zeros(self.glove.vector_size)
        total = 0.0

        for word, weight in weights.items():
            vec = self._get_glove_vec(word)
            if vec is not None:
                combined += weight * vec
                total += weight

        if total == 0:
            return []

        return self._search(combined / total, topn)

    def find_contrast(
        self,
        positive: List[str],
        negative: List[str],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words matching positive concepts minus negative concepts.

        Args:
            positive: Concepts to include
            negative: Concepts to subtract/avoid
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity) tuples

        Example:
            find_contrast(["power", "wisdom"], ["destruction"])
        """
        pos = [self._get_glove_vec(w) for w in positive]
        neg = [self._get_glove_vec(w) for w in negative]
        pos = [v for v in pos if v is not None]
        neg = [v for v in neg if v is not None]

        if not pos:
            return []

        combined = np.mean(pos, axis=0)
        if neg:
            combined -= np.mean(neg, axis=0)

        return self._search(combined, topn)

    def find_midpoint(
        self,
        word1: str,
        word2: str,
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find Egyptian words at the semantic midpoint between two concepts.

        Args:
            word1: First concept
            word2: Second concept
            topn: Number of results

        Returns:
            List of (egyptian_word, similarity) tuples
        """
        return self.find_relationship([word1, word2], topn, aggregate="mean")

    def get_vector(self, egyptian_word: str) -> Optional[np.ndarray]:
        """Get the aligned vector for an Egyptian word."""
        if egyptian_word in self.vocab:
            return self.vectors[self.vocab[egyptian_word]].copy()
        return None

    def similarity(self, eg_word1: str, eg_word2: str) -> float:
        """Compute cosine similarity between two Egyptian words."""
        v1, v2 = self.get_vector(eg_word1), self.get_vector(eg_word2)
        if v1 is None or v2 is None:
            return 0.0
        return float(np.dot(self._normalize(v1), self._normalize(v2)))

    @property
    def vocabulary_size(self) -> int:
        """Number of Egyptian words in the vocabulary."""
        return len(self.vocab)

    def __contains__(self, word: str) -> bool:
        """Check if an Egyptian word is in the vocabulary."""
        return word in self.vocab

    def __len__(self) -> int:
        return len(self.vocab)
