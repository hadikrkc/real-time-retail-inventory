"""
Phase 3 — Spark Structured Streaming feature pipeline.

Reads from retail.sales.events, computes 7-day rolling window features,
and writes to TimescaleDB sales_features via psycopg2 upsert.

Run inside Docker (recommended):
  docker exec spark spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    /opt/pipeline/stream_processor/feature_pipeline.py \
    --kafka-server kafka1:19092 \
    --db-host timescaledb \
    --starting-offsets latest
"""

import argparse
import os

import psycopg2
from psycopg2.extras import execute_values

# Empty string → JARs already on classpath via spark-submit --jars (Docker mode)
KAFKA_PACKAGE = os.environ.get(
    "KAFKA_PACKAGE",
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    avg, col, count, from_json, max, min, sum, to_timestamp, window,
)
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# ── Schema ────────────────────────────────────────────────────────────────────

EVENT_SCHEMA = StructType([
    StructField("event_id",  StringType(),  True),
    StructField("timestamp", StringType(),  True),
    StructField("item_id",   StringType(),  True),
    StructField("dept_id",   StringType(),  True),
    StructField("cat_id",    StringType(),  True),
    StructField("store_id",  StringType(),  True),
    StructField("state_id",  StringType(),  True),
    StructField("sales_qty", IntegerType(), True),
    StructField("day",       StringType(),  True),
])

# ── Spark session ─────────────────────────────────────────────────────────────

def make_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("retail-feature-pipeline")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.streaming.kafka.maxRatePerPartition", "1000")
    )
    if KAFKA_PACKAGE:
        builder = builder.config("spark.jars.packages", KAFKA_PACKAGE)
    return builder.getOrCreate()

# ── Feature computation ───────────────────────────────────────────────────────

def build_features(spark: SparkSession, kafka_server: str, starting_offsets: str) -> DataFrame:
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_server)
        .option("subscribe", "retail.sales.events")
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    events = (
        raw
        .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("event_time", to_timestamp("timestamp"))
        .withWatermark("event_time", "1 day")
    )

    features = (
        events
        .groupBy(
            col("store_id"),
            col("item_id"),
            window(col("event_time"), "7 days", "1 day"),
        )
        .agg(
            avg("sales_qty").alias("rolling_avg_7d"),
            sum("sales_qty").alias("rolling_sum_7d"),
            count("*").alias("event_count_7d"),
            max("sales_qty").alias("max_qty_7d"),
            min("sales_qty").alias("min_qty_7d"),
        )
        .select(
            col("window.end").alias("time"),
            col("store_id"),
            col("item_id"),
            col("rolling_avg_7d"),
            col("rolling_sum_7d"),
            col("event_count_7d"),
            col("max_qty_7d"),
            col("min_qty_7d"),
        )
    )

    return features

# ── Sink: TimescaleDB via psycopg2 upsert ────────────────────────────────────

def make_write_batch(db_host: str):
    def write_batch(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.isEmpty():
            return
        rows = batch_df.collect()
        with psycopg2.connect(
            host=db_host, port=5432, dbname="retail",
            user="retail", password="retail",
        ) as conn:
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO sales_features
                        (time, store_id, item_id, rolling_avg_7d, rolling_sum_7d,
                         event_count_7d, max_qty_7d, min_qty_7d)
                    VALUES %s
                    ON CONFLICT (time, store_id, item_id) DO UPDATE SET
                        rolling_avg_7d = EXCLUDED.rolling_avg_7d,
                        rolling_sum_7d = EXCLUDED.rolling_sum_7d,
                        event_count_7d = EXCLUDED.event_count_7d,
                        max_qty_7d     = EXCLUDED.max_qty_7d,
                        min_qty_7d     = EXCLUDED.min_qty_7d,
                        inserted_at    = NOW()
                """, [
                    (r.time, r.store_id, r.item_id,
                     r.rolling_avg_7d, r.rolling_sum_7d,
                     r.event_count_7d, r.max_qty_7d, r.min_qty_7d)
                    for r in rows
                ])
            conn.commit()
        print(f"[Batch {batch_id}] {len(rows)} feature rows upserted to TimescaleDB")
    return write_batch

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Spark Structured Streaming — feature pipeline")
    parser.add_argument("--kafka-server", default="kafka1:19092",
                        help="Kafka bootstrap server (default: kafka1:19092 for Docker)")
    parser.add_argument("--db-host", default="timescaledb",
                        help="TimescaleDB host (default: timescaledb for Docker)")
    parser.add_argument("--starting-offsets", default="latest",
                        choices=["latest", "earliest"])
    args = parser.parse_args()

    checkpoint_dir = "/tmp/spark-checkpoints/features"

    spark = make_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version     : {spark.version}")
    print(f"Kafka server      : {args.kafka_server}")
    print(f"TimescaleDB       : {args.db_host}:5432/retail")
    print(f"Starting offsets  : {args.starting_offsets}")
    print(f"Kafka package     : {KAFKA_PACKAGE or '(pre-loaded via --jars)'}")
    print("Feature pipeline running. Ctrl+C to stop.\n")

    features = build_features(spark, args.kafka_server, args.starting_offsets)

    query = (
        features.writeStream
        .foreachBatch(make_write_batch(args.db_host))
        .outputMode("update")
        .option("checkpointLocation", checkpoint_dir)
        .trigger(processingTime="10 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
