"""Review data cleaning utilities and CLI.

This module loads raw review data, normalizes textual fields, and computes helpfulness
ratios in a robust way to support downstream sentiment analysis and ranking.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")


def _to_string(value: object) -> str:
    """Convert arbitrary review text-like value to a clean lowercase string."""
    if isinstance(value, str):
        text = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        text = ""
    else:
        text = str(value)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _parse_helpful(value: object) -> Tuple[int, int]:
    """Parse helpful votes into (helpful_yes, total_votes).

    Supports multiple formats commonly seen in Amazon datasets:
    - Python list string like "[3, 5]"
    - Two integers in a tuple-like string
    - Two separate columns handled upstream
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0, 0
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0] or 0), int(value[1] or 0)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return int(parsed[0] or 0), int(parsed[1] or 0)
        except Exception:
            pass
    return 0, 0


def load_reviews(path: str | None = None) -> pd.DataFrame:
    """Load reviews from CSV; if not found, return a tiny demo dataset."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    # Minimal demo rows to keep CLI functional without external data
    return pd.DataFrame(
        [
            {"asin": "B08R39MRDW", "reviewText": "Great sound and battery.", "overall": 5, "helpful": "[4, 5]"},
            {"asin": "B07PZR3PVB", "reviewText": "Decent, but fit is not perfect.", "overall": 3, "helpful": "[2, 4]"},
            {"asin": "B08R39MRDW", "reviewText": "Bass is strong.", "overall": 4, "helpful": "[1, 2]"},
        ]
    )


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned reviews DataFrame with text and helpfulness fields normalized."""
    required_cols = ["asin", "overall"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input reviews must contain column '{col}'")

    cleaned = df.copy()
    text_col = "reviewText" if "reviewText" in cleaned.columns else ("review_text" if "review_text" in cleaned.columns else None)
    if text_col is None:
        cleaned["review_text_clean"] = ""
    else:
        cleaned["review_text_clean"] = cleaned[text_col].map(_to_string)

    # Parse helpful into two columns
    helpful_yes = np.zeros(len(cleaned), dtype=np.int64)
    total_votes = np.zeros(len(cleaned), dtype=np.int64)

    if "helpful" in cleaned.columns:
        pairs = cleaned["helpful"].map(_parse_helpful)
        helpful_yes = np.array([p[0] for p in pairs], dtype=np.int64)
        total_votes = np.array([p[1] for p in pairs], dtype=np.int64)
    else:
        if "helpful_yes" in cleaned.columns:
            helpful_yes = cleaned["helpful_yes"].fillna(0).astype(int).to_numpy()
        if "total_votes" in cleaned.columns:
            total_votes = cleaned["total_votes"].fillna(0).astype(int).to_numpy()

    total_safe = np.where(total_votes == 0, 1, total_votes)
    helpful_ratio = helpful_yes / total_safe

    cleaned["helpful_yes"] = helpful_yes
    cleaned["total_votes"] = total_votes
    cleaned["helpful_ratio"] = helpful_ratio.astype(np.float32)
    cleaned["overall"] = pd.to_numeric(cleaned["overall"], errors="coerce").fillna(0).astype(np.float32)
    return cleaned[["asin", "review_text_clean", "overall", "helpful_yes", "total_votes", "helpful_ratio"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Review data cleaning CLI")
    parser.add_argument("--reviews-csv", default=os.path.join("data", "reviews.csv"), help="Path to raw reviews CSV")
    parser.add_argument("--output-csv", default=os.path.join("data", "cleaned_reviews.csv"), help="Output path for cleaned reviews")
    args = parser.parse_args()

    df = load_reviews(args.reviews_csv)
    cleaned = clean_reviews(df)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    cleaned.to_csv(args.output_csv, index=False)
    print(f"Saved cleaned reviews to: {args.output_csv}")


if __name__ == "__main__":
    main()


