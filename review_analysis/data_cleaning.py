"""Review data cleaning utilities and CLI (PySpark).

This module loads raw review data (JSONL or CSV) using Spark, deduplicates rows
following the notebook logic, and standardizes columns for downstream analysis.
"""

from __future__ import annotations

import argparse
import os
from typing import List

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


def get_spark(app_name: str = "ReviewDataCleaning") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def read_reviews(spark: SparkSession, path: str) -> "F.DataFrame":
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        # Flexible schema covering common Amazon review fields
        schema = T.StructType(
            [
                T.StructField("asin", T.StringType(), True),
                T.StructField("parent_asin", T.StringType(), True),
                T.StructField("user_id", T.StringType(), True),
                T.StructField("rating", T.FloatType(), True),
                T.StructField("overall", T.FloatType(), True),
                T.StructField("timestamp", T.LongType(), True),
                T.StructField("verified_purchase", T.BooleanType(), True),
                T.StructField("helpful_vote", T.IntegerType(), True),
                T.StructField("helpful", T.ArrayType(T.IntegerType()), True),
                T.StructField("text", T.StringType(), True),
                T.StructField("reviewText", T.StringType(), True),
            ]
        )
        return spark.read.schema(schema).json(path)
    # CSV fallback with header inference
    return spark.read.option("header", "true").csv(path)


def deduplicate_reviews(df: "F.DataFrame") -> "F.DataFrame":
    cols_for_exact = [c for c in ["asin", "user_id", "timestamp", "text"] if c in df.columns]
    if cols_for_exact:
        df = df.dropDuplicates(cols_for_exact)

    # Prefer user_id without underscore suffix when duplicates exist for same (text, timestamp)
    if set(["text", "timestamp", "user_id"]).issubset(set(df.columns)):
        win = Window.partitionBy("text", "timestamp").orderBy(F.expr("user_id LIKE '%\\_%'"))
        df = df.withColumn("_rank", F.row_number().over(win)).filter(F.col("_rank") == 1).drop("_rank")
    return df


def standardize_columns(df: "F.DataFrame") -> "F.DataFrame":
    # Unify rating field
    rating_col = "rating" if "rating" in df.columns else ("overall" if "overall" in df.columns else None)
    if rating_col is None:
        df = df.withColumn("rating", F.lit(None).cast("float"))
    elif rating_col != "rating":
        df = df.withColumn("rating", F.col(rating_col).cast("float"))

    # Unify review text
    text_col = "text" if "text" in df.columns else ("reviewText" if "reviewText" in df.columns else None)
    if text_col is None:
        df = df.withColumn("review_text_clean", F.lit(""))
    else:
        df = df.withColumn("review_text_clean", F.regexp_replace(F.lower(F.col(text_col)), r"\s+", " ").cast("string"))

    # Helpful votes (support either helpful_vote or [helpful_yes, total] in 'helpful')
    if "helpful_vote" in df.columns:
        df = df.withColumn("helpful_yes", F.col("helpful_vote").cast("int"))
        df = df.withColumn("total_votes", F.lit(None).cast("int"))
    elif "helpful" in df.columns:
        df = df.withColumn("helpful_yes", F.when(F.size(F.col("helpful")) >= 1, F.col("helpful")[0]).otherwise(0).cast("int"))
        df = df.withColumn("total_votes", F.when(F.size(F.col("helpful")) >= 2, F.col("helpful")[1]).otherwise(0).cast("int"))
    else:
        df = df.withColumn("helpful_yes", F.lit(0).cast("int"))
        df = df.withColumn("total_votes", F.lit(0).cast("int"))

    # Choose product identifier as 'asin'
    if "asin" not in df.columns and "parent_asin" in df.columns:
        df = df.withColumn("asin", F.col("parent_asin"))

    # Verified purchase flag normalization
    if "verified_purchase" not in df.columns:
        df = df.withColumn("verified_purchase", F.lit(None).cast("boolean"))

    return df.select(
        "asin",
        "review_text_clean",
        "rating",
        "helpful_yes",
        "total_votes",
        "verified_purchase",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Review data cleaning CLI (PySpark)")
    parser.add_argument("--reviews-path", default=os.path.join("data", "Amazon_Fashion.jsonl"), help="Path to raw reviews (JSONL/JSON/CSV)")
    parser.add_argument("--output-csv", default=os.path.join("data", "cleaned_reviews.csv"), help="Output path for cleaned reviews CSV")
    args = parser.parse_args()

    spark = get_spark()
    df = read_reviews(spark, args.reviews_path)
    df = deduplicate_reviews(df)
    cleaned = standardize_columns(df)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    cleaned.coalesce(1).write.mode("overwrite").option("header", "true").csv(args.output_csv)
    print(f"Saved cleaned reviews to: {args.output_csv}")


if __name__ == "__main__":
    main()


