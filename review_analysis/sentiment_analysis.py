"""Sentiment analysis over cleaned reviews using NLTK VADER.

The module aggregates sentiment metrics per ASIN to support ranking.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import nltk
import pandas as pd


def _ensure_vader() -> None:
    """Ensure the VADER lexicon is available; download if missing."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def analyze_sentiment(cleaned_reviews: pd.DataFrame) -> pd.DataFrame:
    """Compute per-review sentiment and aggregate by ASIN."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    _ensure_vader()
    sia = SentimentIntensityAnalyzer()

    if "review_text_clean" not in cleaned_reviews.columns:
        raise ValueError("Expected column 'review_text_clean' in cleaned reviews")

    scores = cleaned_reviews["review_text_clean"].map(lambda t: sia.polarity_scores(t)["compound"])
    df = cleaned_reviews.copy()
    df["sentiment_compound"] = scores.astype(float)

    agg = (
        df.groupby("asin", as_index=False)
        .agg(
            avg_sentiment=("sentiment_compound", "mean"),
            avg_rating=("overall", "mean"),
            avg_helpful_ratio=("helpful_ratio", "mean"),
            review_count=("sentiment_compound", "size"),
        )
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Review sentiment analysis CLI")
    parser.add_argument("--cleaned-reviews-csv", default=os.path.join("data", "cleaned_reviews.csv"), help="Path to cleaned reviews CSV")
    parser.add_argument("--output-csv", default=os.path.join("data", "review_sentiment_summary.csv"), help="Output path for aggregated sentiment")
    args = parser.parse_args()

    cleaned = pd.read_csv(args.cleaned_reviews_csv)
    summary = analyze_sentiment(cleaned)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"Saved review sentiment summary to: {args.output_csv}")


if __name__ == "__main__":
    main()


