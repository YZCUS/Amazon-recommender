"""Approximate nearest neighbor search via simple random hyperplane hashing.

This is a lightweight, dependency-free approximation of LSH suitable for small to
medium-sized datasets. It is not a replacement for Spark-based LSH but provides
an easy-to-run baseline that mirrors the configuration concepts from the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .cosine_similarity import compute_cosine_similarity


@dataclass
class LSHConfig:
    """Configuration for the simple LSH model.

    Attributes
    ----------
    num_hyperplanes : int
        Number of random hyperplanes to form signatures (larger is more precise).
    bucket_prefix_bits : int
        Number of leading bits of the signature used for hashing into buckets.
    distance_threshold : float | None
        Optional threshold to filter candidates by cosine distance. If provided,
        items with (1 - cosine_similarity) > distance_threshold are filtered out.
    """

    num_hyperplanes: int = 16
    bucket_prefix_bits: int = 12
    distance_threshold: float | None = None


class RandomHyperplaneLSH:
    """Simple random hyperplane LSH for cosine similarity.

    The model creates a set of random hyperplanes and encodes each vector by the
    sign pattern (+/-) with respect to those hyperplanes. Buckets are formed using
    the leading bits of the signature. Queries are answered by scanning only the
    vectors in matching buckets, and then scoring candidates via cosine similarity.
    """

    def __init__(self, config: LSHConfig | None = None) -> None:
        self.config = config or LSHConfig()
        self._hyperplanes: np.ndarray | None = None
        self._buckets: Dict[int, List[int]] = {}
        self._matrix: np.ndarray | None = None

    def fit(self, matrix: np.ndarray, random_seed: int | None = 42) -> "RandomHyperplaneLSH":
        """Fit the LSH index on the provided matrix.

        Parameters
        ----------
        matrix : np.ndarray
            2D array of shape (num_items, num_features)
        random_seed : int | None
            Seed for reproducible hyperplane generation.
        """
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2D")

        rng = np.random.default_rng(random_seed)
        num_features = matrix.shape[1]
        self._hyperplanes = rng.normal(size=(self.config.num_hyperplanes, num_features))
        self._matrix = matrix.astype(np.float32, copy=False)

        signatures = self._compute_signatures(self._matrix)
        self._buckets.clear()
        for index, sig in enumerate(signatures):
            key = self._prefix_key(sig)
            self._buckets.setdefault(key, []).append(index)
        return self

    def query(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top_k nearest neighbors as (index, score) pairs.

        The method filters candidates by the matching bucket. If the bucket is too
        small, it expands to scanning all items as a safe fallback for small datasets.
        """
        if self._hyperplanes is None or self._matrix is None:
            raise RuntimeError("LSH index is not fitted. Call fit(matrix) first.")

        sig = self._compute_signatures(query_vector[None, :])[0]
        key = self._prefix_key(sig)
        candidate_indices = self._buckets.get(key, [])

        # Safe fallback: if the bucket is tiny, scan all to keep results meaningful
        if len(candidate_indices) < max(10, top_k):
            candidate_indices = list(range(self._matrix.shape[0]))

        candidate_matrix = self._matrix[candidate_indices]
        scores = compute_cosine_similarity(candidate_matrix, query_vector)

        results: List[Tuple[int, float]] = list(zip(candidate_indices, scores.tolist()))

        if self.config.distance_threshold is not None:
            threshold = self.config.distance_threshold
            results = [(i, s) for (i, s) in results if (1.0 - s) <= threshold]

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _compute_signatures(self, matrix: np.ndarray) -> np.ndarray:
        assert self._hyperplanes is not None
        projections = matrix @ self._hyperplanes.T
        signatures = (projections >= 0).astype(np.uint8)
        return signatures

    def _prefix_key(self, signature_bits: np.ndarray) -> int:
        prefix_len = min(self.config.bucket_prefix_bits, signature_bits.shape[-1])
        key = 0
        for bit in signature_bits[:prefix_len]:
            key = (key << 1) | int(bit)
        return key


__all__ = ["LSHConfig", "RandomHyperplaneLSH"]


