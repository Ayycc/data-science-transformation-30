"""Local-mode SparkSession with Delta Lake."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def get_spark(app: str = "clickstream", shuffle_partitions: int = 64, driver_memory: str = "8g", ui: bool = True) -> SparkSession:
    """Create (or reuse) a local SparkSession with the Delta extensions.

    The Delta jars are resolved by ``configure_spark_with_delta_pip`` from Maven Central on first use
    (cached in ~/.ivy2 afterwards). Needs Java 17+ on PATH or JAVA_HOME.
    """
    from delta import configure_spark_with_delta_pip

    # Python workers must run the same interpreter as the driver (venvs, CI)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    cores = os.environ.get("SPARK_LOCAL_CORES", "*")
    builder = (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "true" if ui else "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.databricks.delta.snapshotPartitions", "4")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "4")
        .config("spark.driver.extraJavaOptions", "-Dlog4j2.level=ERROR")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
