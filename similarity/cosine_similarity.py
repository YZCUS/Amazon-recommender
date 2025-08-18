"""Cosine similarity utilities.

This module provides cosine similarity computations for dense vectors.
"""

from __future__ import annotations

import numpy as np


def compute_cosine_similarity(matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of ``matrix`` and ``query_vector``.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D numpy array of shape (num_items, num_features).
    query_vector : np.ndarray
        A 1D numpy array of shape (num_features,).

    Returns
    -------
    np.ndarray
        A 1D array of cosine similarity scores with length equal to ``num_items``.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if query_vector.ndim != 1:
        raise ValueError("query_vector must be 1D")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError("Feature dimension mismatch between matrix and query_vector")

    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(query_vector)

    # Avoid division by zero by replacing zeros with ones (results will become zero similarity)
    safe_matrix_norms = np.where(matrix_norms == 0.0, 1.0, matrix_norms)
    safe_vector_norm = 1.0 if vector_norm == 0.0 else vector_norm

    dots = matrix @ query_vector
    similarities = dots / (safe_matrix_norms * safe_vector_norm)
    return similarities


__all__ = ["compute_cosine_similarity"]


