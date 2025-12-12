"""
Egyptian Lookup - Lite Version for Edge Devices

Instead of loading a full GloVe model, uses a pre-computed vocabulary
of concept words. Suitable for React Native / mobile deployment.

Storage requirements:
  - egyptian_aligned_vectors.npz: 43 MB
  - esoteric_glove_vectors.npz: 62 KB
  - Total: ~43 MB

Usage:
    from egyptian_lookup_lite import EgyptianLookupLite

    lookup = EgyptianLookupLite(
        egyptian_vectors_path="egyptian_aligned_vectors.npz",
        egyptian_vocab_path="egyptian_aligned_vocab.pkl",
        concept_vectors_path="esoteric_glove_vectors.npz"
    )

    # Works the same as full version
    results = lookup.find("sun")
    results = lookup.find_relationship(["death", "rebirth"])
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional


class EgyptianLookupLite:
    """
    Lightweight Egyptian lookup using pre-computed concept vectors.
    No gensim dependency required.
    """

    def __init__(
        self,
        egyptian_vectors_path: Union[str, Path],
        egyptian_vocab_path: Union[str, Path],
        concept_vectors_path: Union[str, Path]
    ):
        """
        Initialize the lite lookup.

        Args:
            egyptian_vectors_path: Path to egyptian_aligned_vectors.npz
            egyptian_vocab_path: Path to egyptian_aligned_vocab.pkl
            concept_vectors_path: Path to esoteric_glove_vectors.npz (or similar)
        """
        # Load Egyptian vectors
        egyptian_vectors_path = Path(egyptian_vectors_path)
        if egyptian_vectors_path.suffix == '.npz':
            data = np.load(egyptian_vectors_path)
            self.egyptian_vectors = data['vectors'].astype(np.float32)
        else:
            self.egyptian_vectors = np.load(egyptian_vectors_path)

        # Load Egyptian vocab
        with open(egyptian_vocab_path, 'rb') as f:
            self.egyptian_vocab = pickle.load(f)
        self.idx_to_egyptian = {v: k for k, v in self.egyptian_vocab.items()}

        # Load concept vectors (pre-computed GloVe subset)
        concept_data = np.load(concept_vectors_path, allow_pickle=True)
        self.concept_vectors = concept_data['vectors'].astype(np.float32)
        self.concept_words = list(concept_data['words'])
        self.concept_vocab = {w: i for i, w in enumerate(self.concept_words)}

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _get_concept_vec(self, word: str) -> Optional[np.ndarray]:
        word = word.lower()
        if word in self.concept_vocab:
            return self.concept_vectors[self.concept_vocab[word]]
        return None

    def _search(self, vec: np.ndarray, topn: int) -> List[Tuple[str, float]]:
        vec = self._normalize(vec)
        sims = self.egyptian_vectors @ vec
        indices = np.argsort(sims)[-topn:][::-1]
        return [(self.idx_to_egyptian[i], float(sims[i])) for i in indices]

    def available_concepts(self) -> List[str]:
        """Return list of available concept words."""
        return self.concept_words.copy()

    def has_concept(self, word: str) -> bool:
        """Check if a concept word is available."""
        return word.lower() in self.concept_vocab

    def find(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find Egyptian words for a concept."""
        vec = self._get_concept_vec(word)
        return self._search(vec, topn) if vec is not None else []

    def find_relationship(
        self,
        words: List[str],
        topn: int = 10,
        aggregate: str = "sum"
    ) -> List[Tuple[str, float]]:
        """Find Egyptian words matching combined concepts."""
        vecs = [self._get_concept_vec(w) for w in words]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            return []
        combined = np.sum(vecs, axis=0) if aggregate == "sum" else np.mean(vecs, axis=0)
        return self._search(combined, topn)

    def find_blend(
        self,
        weights: Dict[str, float],
        topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Find Egyptian words for weighted concept blend."""
        combined = np.zeros(300, dtype=np.float32)
        total = 0.0
        for word, weight in weights.items():
            vec = self._get_concept_vec(word)
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
        """Find Egyptian words: positive concepts minus negative."""
        pos = [self._get_concept_vec(w) for w in positive]
        neg = [self._get_concept_vec(w) for w in negative]
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
        """Find Egyptian words at semantic midpoint."""
        return self.find_relationship([word1, word2], topn, aggregate="mean")

    @property
    def egyptian_vocabulary_size(self) -> int:
        return len(self.egyptian_vocab)

    @property
    def concept_vocabulary_size(self) -> int:
        return len(self.concept_words)
