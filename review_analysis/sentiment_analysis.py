"""Sentiment analysis over cleaned reviews using NLTK VADER (PySpark).

This module reads cleaned reviews (CSV written by data_cleaning.py), computes
VADER compound sentiment per review via mapInPandas, and aggregates by ASIN.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import nltk
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def _ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def get_spark(app_name: str = "ReviewSentimentAnalysis") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def compute_sentiment_map(pdf_iter):
    from nltk.sentiment import SentimentIntensityAnalyzer
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    for pdf in pdf_iter:
        if not len(pdf):
            yield pdf
            continue
        pdf = pdf.copy()
        texts = pdf.get("review_text_clean", pd.Series(
            [], dtype=str)).astype(str)
        pdf["sentiment_compound"] = texts.map(
            lambda t: sia.polarity_scores(t)["compound"]).astype(float)
        yield pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review sentiment analysis CLI (PySpark)")
    parser.add_argument("--cleaned-reviews-csv", default=os.path.join("data",
                        "cleaned_reviews.csv"), help="Path to cleaned reviews CSV")
    parser.add_argument("--output-csv", default=os.path.join("data",
                        "review_sentiment_summary.csv"), help="Output path for aggregated sentiment")
    args = parser.parse_args()

    spark = get_spark()
    df = spark.read.option("header", "true").csv(args.cleaned_reviews_csv)

    # Ensure expected columns
    for c, typ in [("asin", T.StringType()), ("rating", T.FloatType()), ("helpful_yes", T.IntegerType()), ("total_votes", T.IntegerType()), ("review_text_clean", T.StringType())]:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(typ))
        else:
            df = df.withColumn(c, F.col(c).cast(typ))

    # helpful ratio
    total_safe = F.when(F.col("total_votes").isNull() | (
        F.col("total_votes") == 0), F.lit(1)).otherwise(F.col("total_votes"))
    df = df.withColumn("helpful_ratio", (F.col(
        "helpful_yes").cast("float") / total_safe).cast("float"))

    # compute sentiment with mapInPandas
    schema = T.StructType(
        [*(df.schema.fields), T.StructField("sentiment_compound", T.FloatType(), True)])
    df = df.mapInPandas(compute_sentiment_map, schema=schema)

    # aggregate by ASIN
    summary = (
        df.groupBy("asin").agg(
            F.avg("sentiment_compound").alias("avg_sentiment"),
            F.avg("rating").alias("avg_rating"),
            F.avg("helpful_ratio").alias("avg_helpful_ratio"),
            F.count("sentiment_compound").alias("review_count"),
        )
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary.coalesce(1).write.mode("overwrite").option(
        "header", "true").csv(args.output_csv)
    print(f"Saved review sentiment summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
