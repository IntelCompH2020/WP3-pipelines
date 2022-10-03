import sys
from pathlib import Path

from pyspark.sql import SparkSession

sys.path.insert(0, Path(__file__).parent.parent.resolve().as_posix())
print(sys.path)

from OpenAIRE import gen_OA_metadata
from PATSTAT import gen_PT_metadata
from SemanticScholar import gen_SS_metadata

if __name__ == "__main__":
    # Create session
    spark = SparkSession.builder.appName("WP3pipeline").getOrCreate()
    sc = spark.sparkContext
    print(sc.version)

    gen_OA_metadata(spark)
    gen_PT_metadata(spark)
    gen_SS_metadata(spark)
