"""PySpark-based Similarity Analysis Engine.

Pipeline: Tokenizer → StopWordsRemover → HashingTF → IDF → VectorAssembler → Normalizer → BucketedRandomProjectionLSH
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    VectorAssembler,
    Normalizer,
    BucketedRandomProjectionLSH,
)


def get_spark(app_name: str = "SimilarityAnalysisEngine") -> SparkSession:
    """Create or get a SparkSession with sensible defaults."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def read_metadata_json(spark: SparkSession, path: str) -> "F.DataFrame":
    """Read product metadata JSONL with a focused schema."""
    schema = T.StructType(
        [
            T.StructField("parent_asin", T.StringType(), True),
            T.StructField("title", T.StringType(), True),
            T.StructField("price", T.FloatType(), True),
            T.StructField("description", T.ArrayType(T.StringType()), True),
            T.StructField("main_category", T.StringType(), True),
            T.StructField("features", T.StringType(), True),
            T.StructField("categories", T.ArrayType(T.StringType()), True),
        ]
    )
    return spark.read.schema(schema).json(path)


def build_feature_pipeline(num_features: int = 2000) -> Pipeline:
    """Construct the text feature pipeline with IDF, assembling and normalization."""
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words",
                           outputCol="rawFeatures", numFeatures=num_features)
    idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures")
    assembler = VectorAssembler(
        inputCols=["idfFeatures"], outputCol="features_vec")
    normalizer = Normalizer(inputCol="features_vec",
                            outputCol="normalized_features")
    return Pipeline(stages=[tokenizer, remover, hashing_tf, idf, assembler, normalizer])


def find_similar_products_spark(
    meta_json_path: str,
    query_asin: str,
    top_k: int = 10,
    bucket_length: float = 0.5,
    distance_threshold: float = 5.0,
    num_features: int = 2000,
) -> List[Tuple[str, float]]:
    """Compute top-k similar products for a given ASIN using Spark LSH.

    Returns a list of (asin, similarity_score) sorted by score desc.
    """
    spark = get_spark()
    meta_df = read_metadata_json(spark, meta_json_path)

    # Combine relevant text columns into a single text column
    meta_df = meta_df.select(
        "parent_asin",
        "title",
        "description",
        "price",
        "categories",
        "features",
    )

    meta_df = meta_df.withColumn(
        "text",
        F.concat_ws(" ", F.col("title"), F.col("categories"),
                    F.col("description"), F.col("features")),
    )

    # Build and fit the pipeline
    pipeline = build_feature_pipeline(num_features=num_features)
    model = pipeline.fit(meta_df)
    featured_df = model.transform(meta_df)

    # LSH for approximate nearest neighbors
    lsh = BucketedRandomProjectionLSH(
        inputCol="normalized_features",
        outputCol="hashes",
        bucketLength=bucket_length,
    )
    lsh_model = lsh.fit(featured_df)

    joined = lsh_model.approxSimilarityJoin(
        featured_df, featured_df, distance_threshold, distCol="distance"
    )

    # Filter for the target item and convert distance to similarity
    filtered = (
        joined.filter(F.col("datasetA.parent_asin") == F.lit(query_asin))
        .withColumn("similarity", 1 / (1 + F.col("distance")))
        .select(
            F.col("datasetB.parent_asin").alias("recommended_asin"),
            F.col("similarity"),
        )
        .orderBy(F.col("similarity").desc())
    )

    # Collect top_k + 1 to exclude self-match if present
    rows = filtered.limit(top_k + 1).collect()
    results: List[Tuple[str, float]] = []
    for r in rows:
        asin = r["recommended_asin"]
        score = float(r["similarity"]) if r["similarity"] is not None else 0.0
        if asin == query_asin:
            continue
        results.append((asin, score))
        if len(results) >= top_k:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Similarity Analysis Engine (PySpark)")
    parser.add_argument("--input-asin", required=True, help="Query ASIN")
    parser.add_argument(
        "--meta-json",
        default=os.path.join("data", "meta_Amazon_Fashion.jsonl"),
        help="Path to product metadata JSONL",
    )
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of similar products to return")
    parser.add_argument("--bucket-length", type=float,
                        default=0.5, help="LSH bucket length")
    parser.add_argument("--distance-threshold", type=float,
                        default=5.0, help="Max LSH distance to consider")
    parser.add_argument("--num-features", type=int,
                        default=2000, help="HashingTF number of features")
    args = parser.parse_args()

    results = find_similar_products_spark(
        meta_json_path=args.meta_json,
        query_asin=args.input_asin,
        top_k=args.top_k,
        bucket_length=args.bucket_length,
        distance_threshold=args.distance_threshold,
        num_features=args.num_features,
    )

    for asin, score in results:
        print(f"{asin}\t{score:.6f}")


if __name__ == "__main__":
    main()
