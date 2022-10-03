from configparser import ConfigParser
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window


def gen_PT_metadata(spark):
    # Define directories
    #

    cf = ConfigParser()
    cf.read("config.cf")

    dir_pt = Path(cf.get("data", "patstat"))

    #################################################
    #### Load info
    #################################################
    pt = spark.read.parquet(dir_pt.joinpath("patstat_appln.parquet").as_posix())

    # Key bibliographical data elements relevant to identify patent publications
    pt_pub = spark.read.parquet(dir_pt.joinpath("tls211.parquet").as_posix()).select(
        "appln_id", "pat_publn_id", "publn_kind"
    )
    # Links between publications, applications and non-patent literature documents with regards to citations.
    pt_cit = spark.read.parquet(dir_pt.joinpath("tls212.parquet").as_posix()).select(
        "pat_publn_id", "cited_pat_publn_id", "cited_appln_id", "cited_npl_publn_id"
    )
    cit_appln = pt_cit.select("pat_publn_id", "cited_appln_id").where(
        "cited_appln_id>0"
    )
    cit_publn = pt_cit.select("pat_publn_id", "cited_pat_publn_id").where(
        "cited_pat_publn_id>0"
    )

    #################################################
    #### Citation info
    #################################################
    # First, join appln_ids with docdb_family_id in publications
    pt_fam = pt.select(
        F.col("appln_id").alias("appln_id_f"), "docdb_family_id"
    )  # appln_id_f <-> docdb_family_id
    pub_fam = pt_pub.join(pt_fam, pt_pub.appln_id == pt_fam.appln_id_f, "left").drop(
        "appln_id_f"
    )  # pat_publn_id <-> appln_id <-> docdb_family_id

    # Then, convert citations through appln_ids to citations using docdb_family_id
    pbf_src = pub_fam.select(*(F.col(el).alias(el + "_src") for el in pub_fam.columns))
    pbf_dst = pub_fam.select(*(F.col(el).alias(el + "_dst") for el in pub_fam.columns))

    # These are the publiactions citing other publications (include family identifiers)
    cit_publn_fam = (
        cit_publn.join(
            pbf_src, cit_publn.pat_publn_id == pbf_src.pat_publn_id_src, "left"
        )
        .join(pbf_dst, cit_publn.cited_pat_publn_id == pbf_dst.pat_publn_id_dst, "left")
        .drop("pat_publn_id", "cited_pat_publn_id")
        .select(
            "appln_id_src",
            "pat_publn_id_src",
            "publn_kind_src",
            "docdb_family_id_src",
            "appln_id_dst",
            "pat_publn_id_dst",
            "publn_kind_dst",
            "docdb_family_id_dst",
        )
        .withColumn(
            "autoCit", F.col("docdb_family_id_src") == F.col("docdb_family_id_dst")
        )
    )

    # These are the publications citing appln_id (include family identifiers)
    cit_appln_fam = (
        cit_appln.alias("df")
        .join(
            pbf_src.alias("df1"),
            F.col("df.pat_publn_id") == F.col("df1.pat_publn_id_src"),
            "left",
        )
        .join(
            pt_fam.select(
                "appln_id_f", F.col("docdb_family_id").alias("docdb_family_id_dst")
            ).alias("df2"),
            F.col("df.cited_appln_id") == F.col("df2.appln_id_f"),
            "left",
        )
        .drop("pat_publn_id_src", "appln_id_f")
        .select(
            "appln_id_src",
            "pat_publn_id",
            "publn_kind_src",
            "docdb_family_id_src",
            "cited_appln_id",
            "docdb_family_id_dst",
        )
        .withColumn(
            "autoCit", F.col("docdb_family_id_src") == F.col("docdb_family_id_dst")
        )
    )
    # cit_fam.write.parquet(
    #     "/export/usuarios01/joseantem/Proyectos/out/fam_citations.parquet",
    #     mode="overwrite",
    # )

    # Concat cit_publn_fam and cit_appln_fam
    full_cit = cit_publn_fam.select(
        "docdb_family_id_src", "docdb_family_id_dst", "autoCit"
    ).unionByName(
        cit_appln_fam.select("docdb_family_id_src", "docdb_family_id_dst", "autoCit")
    )
    # Get unique total citations
    window = Window.partitionBy("docdb_family_id_src", "docdb_family_id_dst").orderBy(
        "docdb_family_id_src", "docdb_family_id_dst"
    )
    full_cit_un = (
        full_cit.withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") == 1)
        .drop("rank")
    )

    # Auto citation
    auto_cit = full_cit_un.where("autoCit")

    #################################################
    #### Get counts
    #################################################
    tot_cit_appln = cit_appln_fam.count()
    tot_cit_publn = cit_publn_fam.count()
    tot_full_cit = full_cit.count()
    full_cit_unique = full_cit_un.count()
    auto_cit = auto_cit.count()

    print("Citations by doc-db family")
    print(f"Number of citations (applications): {tot_cit_appln}")
    print(f"Number of citations (publications): {tot_cit_publn}")
    print(f"Number of combined citations: {tot_full_cit}")
    print(f"Number of unique citations: {full_cit_unique} ({full_cit_unique/tot_full_cit*100:.3f}%)")
    print(f"Number of auto citations: {auto_cit} ({auto_cit/auto_cit*100:.3f}%)")

    #################################################
    #### Output data
    #################################################
    columns = ["Num_cit", "Num_cit_wo_self"]
    row = [full_cit_unique, full_cit_unique-auto_cit]
    data = [row]
    df = spark.createDataFrame(data=data, schema=columns)
    df.printSchema()
    df.show(truncate=False)
    df.write.parquet(
        "/export/ml4ds/IntelComp/Datalake/PATSTAT/metadata.parquet",
        mode="overwrite",
    )


if __name__ == "__main__":
    # Create session
    spark = SparkSession.builder.appName("WP3pipeline").getOrCreate()
    sc = spark.sparkContext
    print(sc.version)

    gen_PT_metadata(spark)
