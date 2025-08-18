"""Final recommendation ranking engine.

Combines metadata similarity scores (from PySpark LSH) with review-based signals
to produce a final ranking for an input ASIN.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from similarity.metadata_analysis import find_similar_products_spark
from pyspark.sql import SparkSession


@dataclass
class RankingWeights:
    """Weights for each component in the final score."""

    similarity_weight: float = 0.6
    sentiment_weight: float = 0.2
    rating_weight: float = 0.15
    helpful_weight: float = 0.05


def _normalize_series(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v - min_v < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def rank_candidates(
    input_asin: str,
    meta_json_path: str,
    review_summary: pd.DataFrame,
    top_k: int = 5,
    weights: RankingWeights | None = None,
) -> List[Tuple[str, float]]:
    """Return final ranked candidates as (asin, score)."""
    weights = weights or RankingWeights()
    similar = find_similar_products_spark(
        meta_json_path=meta_json_path,
        query_asin=input_asin,
        top_k=max(50, top_k),
    )
    sim_df = pd.DataFrame(similar, columns=["asin", "similarity_score"])

    merged = sim_df.merge(review_summary, on="asin", how="left")
    # Normalize components to [0, 1]
    merged["similarity_norm"] = _normalize_series(merged["similarity_score"])
    merged["sentiment_norm"] = _normalize_series(merged.get(
        "avg_sentiment", pd.Series(0.0, index=merged.index)))
    merged["rating_norm"] = _normalize_series(merged.get(
        "avg_rating", pd.Series(0.0, index=merged.index)))
    merged["helpful_norm"] = _normalize_series(merged.get(
        "avg_helpful_ratio", pd.Series(0.0, index=merged.index)))

    merged["final_score"] = (
        weights.similarity_weight * merged["similarity_norm"]
        + weights.sentiment_weight * merged["sentiment_norm"]
        + weights.rating_weight * merged["rating_norm"]
        + weights.helpful_weight * merged["helpful_norm"]
    )

    ranked = merged.sort_values("final_score", ascending=False).head(top_k)
    return list(zip(ranked["asin"].tolist(), ranked["final_score"].astype(float).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Final ranking engine")
    parser.add_argument("--input-asin", required=True,
                        help="ASIN to recommend alternatives for")
    parser.add_argument("--meta-json", default=os.path.join("data",
                        "meta_Amazon_Fashion.jsonl"), help="Path to product metadata JSONL")
    parser.add_argument("--reviews-summary-csv", default=os.path.join("data",
                        "review_sentiment_summary.csv"), help="Path to aggregated review summary CSV")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of final recommendations")
    args = parser.parse_args()

    # Load review summary written by Spark (directory CSV) or single CSV file
    def _load_review_summary(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=[
                "asin", "avg_sentiment", "avg_rating", "avg_helpful_ratio", "review_count"
            ])
        if os.path.isdir(path):
            spark = SparkSession.builder.appName(
                "RankingEngineLoad").getOrCreate()
            sdf = spark.read.option("header", "true").csv(path)
            pdf = sdf.toPandas()
            # Cast expected numeric columns
            for col in ["avg_sentiment", "avg_rating", "avg_helpful_ratio", "review_count"]:
                if col in pdf.columns:
                    pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
            if "asin" in pdf.columns:
                pdf["asin"] = pdf["asin"].astype(str)
            return pdf
        return pd.read_csv(path)

    review_summary = _load_review_summary(args.reviews_summary_csv)

    results = rank_candidates(
        input_asin=args.input_asin,
        meta_json_path=args.meta_json,
        review_summary=review_summary,
        top_k=args.top_k,
    )

    for asin, score in results:
        print(f"{asin}\t{score:.4f}")


if __name__ == "__main__":
    main()
