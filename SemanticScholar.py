from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pandas as pd
# import plotly.express as px
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window


def gen_SS_metadata(spark):
    # Define directories
    #

    cf = ConfigParser()
    cf.read("config.cf")

    dir_ss = Path(cf.get("data", "semanticscholar"))

    #################################################
    #### Load info
    #################################################
    ss = spark.read.parquet(dir_ss.joinpath("papers.parquet").as_posix())
    ss_cit = spark.read.parquet(dir_ss.joinpath("citations.parquet").as_posix())

    #################################################
    #### Citation information
    #################################################

    window = Window.partitionBy("source", "dest").orderBy("source", "dest")
    ss_cit = ss_cit.withColumn("dup", F.row_number().over(window)).withColumn(
        "autoCit", F.col("source") == F.col("dest")
    )
    ss_cit_unique = ss_cit.where(F.col("dup") == 1)

    #################################################
    #### Quartile information
    #################################################
    df = pd.read_csv("scimagojr 2021.csv", sep=";")
    df = df.fillna(np.nan).replace([np.nan], [None])
    scimago = spark.createDataFrame(df)

    ss = ss.join(scimago, F.lower(ss.journalName) == F.lower(scimago.Title), "left")
    ss_journ = ss.where(F.length("journalName") > 0)
    ss_journ_val = ss_journ.where(F.col("Rank").isNotNull())

    #################################################
    #### Get counts
    #################################################
    tot_ss = ss.count()
    tot_journ = ss_journ.count()
    tot_journ_val = ss_journ_val.count()
    ss_tot = ss_cit.count()
    ss_tot_unique = ss_cit_unique.count()
    ss_tot_unique_self = ss_cit_unique.where(F.col("autoCit")).count()
    print(f"Total papers in SS: {tot_ss}")
    print(f"Total papers in SS with journal: {tot_journ} ({tot_journ/tot_ss * 100:.3f}%)")
    print(f"Total papers in SS with valid journal: {tot_journ_val} ({tot_journ_val/tot_ss * 100:.3f}%)")
    print(f"Number citations: {ss_tot}")
    print(f"Number unique citations: {ss_tot_unique} ({ss_tot_unique/ss_tot*100:.3f}%)")
    print(f"Number self-citations: {ss_tot_unique_self} ({ss_tot_unique_self/ss_tot*100:.3f}%)")

    df_year_SJR = (
        ss_journ.select("year", "SJR Best Quartile")
        .groupBy("year", "SJR Best Quartile")
        .count()
        .toPandas()
    )
    df = df_year_SJR.groupby(["year", "SJR Best Quartile"]).agg(lambda x: x)
    df = df_year_SJR.rename(
        columns={"year": "Year", "SJR Best Quartile": "Quartile", "count": "Count"}
    )
    # df = df[df["Year"] > 1950]
    quartile_order = sorted(
        df_year_SJR["SJR Best Quartile"].unique(),
        key=lambda x: "zz" if x is None else x,
    )
    quarts = df[["Quartile", "Count"]].groupby(["Quartile"]).sum().to_dict()["Count"]
    # fig = px.bar(
    #     df,
    #     x="Year",
    #     y="Count",
    #     color="Quartile",
    #     #     barmode="group",
    #     #     log_y=True,
    #     category_orders={"Quartile": quartile_order},
    #     title="",
    # )
    # fig.update_layout(
    #     {
    #         "xaxis.rangeslider.visible": True,
    #         "yaxis.fixedrange": True,
    #     }
    # )
    # fig.show()

    #################################################
    #### Output data
    #################################################
    columns = ["Num_cit", "Num_cit_wo_self", "quartiles"]
    row = [ss_tot, ss_tot - ss_tot_unique_self, quarts]
    data = [row]
    df = spark.createDataFrame(data=data, schema=columns)
    df.printSchema()
    df.show(truncate=False)
    df.write.parquet(
        "/export/ml4ds/IntelComp/Datalake/SemanticScholar/metadata.parquet",
        mode="overwrite",
    )


if __name__ == "__main__":
    # Create session
    spark = SparkSession.builder.appName("WP3pipeline").getOrCreate()
    sc = spark.sparkContext
    print(sc.version)

    gen_SS_metadata(spark)
