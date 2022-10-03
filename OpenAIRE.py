import re
from configparser import ConfigParser
from pathlib import Path

import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, Row, StringType, StructType

plt.rc("font", size=20)  # controls default text size
plt.rc("axes", titlesize=20)  # fontsize of the title
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=16)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=16)  # fontsize of the y tick labels
plt.rc("legend", fontsize=16)  # fontsize of the legend


#################################################
#### Auxiliary functions
#################################################


def flattenStructSchema(schema, prefix=None):
    """
    Converts the structure:
        root
        | - el1
        |   | - sub1
        |        | - sub_sub
        |        | - sub_sub2
        |   | - sub2
        | - el2

    into:
        root.el1.sub1.sub_sub
        root.el1.sub1.sub_sub2
        root.el1.sub2
        root.el2
    """
    columnName = []
    for f in schema.fields:
        if prefix:
            name = f"{prefix}.{f.name}"
        else:
            name = f.name

        if isinstance(f.dataType, StructType):
            columnName.extend(flattenStructSchema(f.dataType, name))
        elif isinstance(f.dataType, ArrayType):
            if isinstance(f.dataType.elementType, StructType):
                columnName.extend(flattenStructSchema(f.dataType.elementType, name))
            elif isinstance(f.dataType.elementType, ArrayType):
                columnName.extend(
                    flattenStructSchema(f.dataType.elementType.elementType, name)
                )
            else:
                columnName.append(name)
        else:
            columnName.append(name)
    return columnName


def get_value(x):
    """
    Get specific element
    """
    quals = []
    if x is None:
        return None
    for el in x:
        if el is None:
            quals.append(None)
        else:
            if isinstance(el, list):
                for i in el:
                    try:
                        quals.append(f"{i['value']}")
                    except:
                        quals.append(None)
            elif isinstance(el, Row):
                quals.append(el.asDict().get("value", None))
            elif isinstance(el, str):
                quals.append(el)
            else:
                quals.append(f"{el}")
    quals = list(set(quals) - set([None, ""]))
    if all([el is None for el in quals]) or not quals:
        quals = None
    return quals


get_value_udf = F.udf(get_value, ArrayType(StringType()))


def get_qual(x):
    """
    Get type and value of specific element
    """
    quals = []
    if x is None:
        return None
    for el in x:
        if el is None:
            quals.append(None)
        else:
            if isinstance(el, list):
                for i in el:
                    try:
                        quals.append(f"{i['qualifier']['classid']}:{i['value']}")
                    except:
                        quals.append(None)
            elif el:
                try:
                    quals.append(f"{el['value']}")
                except:
                    quals.append(f"{el}")
            else:
                quals.append(None)
    if all([el is None for el in quals]) or not quals:
        quals = None
    return quals


get_qual_udf = F.udf(get_qual, ArrayType(StringType()))


def combine_array(cols, remove_dup=True):
    """
    Combine two or more column arrays (including null) as in:
    F.array_union(
        F.when(col1.isNotNull(), col1).otherwise(F.array()),
        F.when(col2.isNotNull(), col2).otherwise(F.array())
    )
    """
    comb = []
    for c in cols:
        if c:
            comb.extend(c)
    if remove_dup:
        comb = list(set(comb) - set([None, ""]))
    if not comb:
        return None
    return comb


udf_combine_array = F.udf(combine_array, ArrayType(StringType()))


def get_author_pid(x):
    """
    Get author identifiers:
        - Microsoft graph
        - ORCID
    """
    pids = []
    if x is None:
        return None
    for el in x:
        if el:
            if "microsoft" in el.lower():
                pids.append(f"microsoft:{el.split('/')[-1]}".strip())
            elif "orcid" in el.lower():
                pids.append(f"orcid:{el.split(':')[-1]}".strip())
            else:
                pids.append(el)
        else:
            pids.append(None)
    return pids


udf_get_author_pid = F.udf(get_author_pid, ArrayType(StringType()))


def valid_doi(doi):
    """
    Returns the passed DOI if valid, else None
    """
    match = re.match("^10\.\d{4,9}\/[-._;()/:\w]+$", doi)
    if match:
        return match.string
    return None


def get_doi(x):
    """
    Obtain the DOI from a column
    """
    if x is None:
        return None
    for el in x:
        if el is not None:
            info = el.lower()
            info = info.split("doi:")[-1]
            info = info.split("doi.org")[-1]
            info = info.strip("/").strip()
            if valid_doi(info):
                return info
    return None


udf_get_doi = F.udf(get_doi, StringType())


def hasOrcid(x):
    """
    Obtain the orcid
    """
    if x is None:
        return None
    orcids = [el for el in x if "orcid" in el]
    if not orcids:
        return None
    return orcids


udf_hasOrcid = F.udf(hasOrcid, ArrayType(StringType()))


def hasMicrosoft(x):
    """
    Obtain the orcid
    """
    if not x:
        return None
    microsoft = [el for el in x if "microsoft" in el]
    if not microsoft:
        return None
    return microsoft


udf_hasMicrosoft = F.udf(hasMicrosoft, ArrayType(StringType()))


def gen_OA_metadata(spark):
    # Define directories
    #

    cf = ConfigParser()
    cf.read("config.cf")

    dir_oa = Path(cf.get("data", "openaire"))

    #################################################
    #### Load info
    #################################################
    df = spark.read.parquet(dir_oa.joinpath("publication").as_posix())
    df_ref = spark.read.parquet(dir_oa.joinpath("relation").as_posix())

    #################################################
    #### Filter OpenAIRE
    #################################################
    # Only the following columns are relevant to us
    cols = [
        F.col("id"),
        get_value_udf(F.col("author.affiliation")),
        F.col("author.fullname"),
        get_qual_udf(F.col("author.pid")),
        #  get_value_udf(F.col('collectedfrom')),
        #  get_value_udf(F.col('contributor')),
        F.col("country.classid"),
        #  F.col('country.classname'),
        #  get_value_udf(F.col('description')),
        #  get_value_udf(F.col('format')),
        #  get_value_udf(F.col('fulltext')),
        get_value_udf("dateofacceptance"),
        get_value_udf("instance.dateofacceptance"),
        get_qual_udf(F.col("instance.alternateIdentifier")),
        #  F.col('instance.distributionlocation'),
        get_value_udf(F.col("instance.hostedby")),
        get_qual_udf(F.col("instance.pid")),
        F.col("instance.url"),
        #  F.col('instance.instancetype.classname'),
        F.array_distinct(F.col("instance.instancetype.classname")),
        F.col("language.classid"),
        #  F.col('language.classname'),
        get_qual_udf(F.col("pid")),
        get_value_udf(F.col("publisher")),
        get_value_udf(F.col("title")).getItem(0),
    ]
    cols = [
        el.alias(re.sub("\).*|.*\(", "", el._jc.toString()).replace(".", "_"))
        for el in cols
    ]

    # Filter DataFrame to obtain only relevant fields
    oa_filtered = (
        (
            df.select(cols)
            .withColumn("auth_pid", udf_get_author_pid("author_pid"))
            .withColumn(
                "doi",
                udf_get_doi(
                    udf_combine_array(
                        F.array(
                            udf_combine_array(F.col("instance_url")),
                            F.col("instance_alternateIdentifier"),
                            F.col("instance_pid"),
                        )
                    )
                )
                # ).drop(
                #     "instance_alternateIdentifier", "instance_pid", "author_pid"
            )
            .withColumn(
                "acceptance",
                F.array_sort(
                    udf_combine_array(
                        F.array(
                            F.col("dateofacceptance"),
                            F.col("instance_dateofacceptance"),
                        )
                    )
                ).getItem(0),
            )
        )
        .select(
            "id",
            "instance_instancetype_classname",
            "country_classid",
            "language_classid",
            "auth_pid",
            "doi",
            "acceptance",
            "title",
        )
        .dropna(
            subset=[
                "instance_instancetype_classname",
                "country_classid",
                "language_classid",
                "auth_pid",
                "doi",
            ],
            how="all",
        )
    )

    # oa_filtered.write.parquet(
    #     "/export/ml4ds/IntelComp/Datalake/OpenAIRE/oa_filtered.parquet",
    #     mode="overwrite",
    # )

    # oa_filtered = spark.read.parquet(
    #     "/export/ml4ds/IntelComp/Datalake/OpenAIRE/oa_filtered.parquet"
    # )

    #################################################
    #### Merge with SemanticScholar
    #################################################

    # Read SS
    dir_ss = Path(cf.get("data", "semanticscholar"))
    ss = spark.read.parquet(dir_ss.joinpath("papers.parquet").as_posix())

    # Merge with filtered DataFrame using DOI
    oa_ss = (
        oa_filtered.where(F.col("doi").isNotNull())
        .dropDuplicates(subset=["doi"])
        .join(
            ss.select(F.col("id").alias("ssid"), F.lower("doi").alias("doi"))
            .where(F.length("doi") > 0)
            .dropDuplicates(subset=["doi"]),
            on="doi",
            how="left",
        )
    )

    # oa_ss.write.parquet(
    #     "/export/ml4ds/IntelComp/Datalake/OpenAIRE/oa_ss.parquet", mode="overwrite"
    # )

    # oa_ss = spark.read.parquet("/export/ml4ds/IntelComp/Datalake/OpenAIRE/oa_ss.parquet")

    #################################################
    #### Citations
    #################################################
    oa_cites = df_ref.where(F.col("relClass") == "cites").count()
    oa_cited = df_ref.where(F.col("relClass") == "isCitedBy").count()
    oa_cit = df_ref.where(F.col("subRelType") == "citation").count()

    # Self citations
    window = Window.partitionBy("source", "target").orderBy("source", "target")
    df_self_cit = (
        df_ref.where(F.col("subRelType") == "citation")
        .withColumn("dup", F.row_number().over(window))
        .withColumn("autoCit", F.col("source") == F.col("target"))
    )
    ss_cit_unique = df_self_cit.where(F.col("dup") == 1)
    cit_wo_self = ss_cit_unique.count()

    print("Citations")
    print(f"Source -> Dest: {oa_cites}")
    print(f"Dest -> Sourc: {oa_cited}")
    print(f"Total: {oa_cit}")
    print(f"Total (w/o self citations): {cit_wo_self}")

    #################################################
    #### Counts
    #################################################

    # General counts
    # Total
    n_oa_filt = oa_filtered.count()
    n_oa_filt_u = oa_filtered.select("id").distinct().count()
    # With country
    n_oa_ctry = oa_filtered.where(F.size("country_classid") > 0).count()
    # With doi
    n_oa_doi = oa_filtered.where(F.col("doi").isNotNull()).count()
    n_oa_doi_u = (
        oa_filtered.select("doi").where(F.col("doi").isNotNull()).distinct().count()
    )
    # In SemanticScholar
    n_oa_ss = oa_ss.where(F.col("ssid").isNotNull()).count()
    n_oa_ss_u = oa_ss.select("ssid").where(F.col("ssid").isNotNull()).distinct().count()
    # With DOI & country
    n_oa_C_DOI = (
        oa_ss.where(F.col("ssid").isNotNull())
        .where(F.size("country_classid") > 0)
        .count()
    )
    print("--1--")

    # Publication types
    oa_withType = oa_filtered.where(
        F.col("instance_instancetype_classname").isNotNull()
    )
    n_oa_withType = oa_withType.count()
    instance_types = (
        oa_withType.select(F.explode("instance_instancetype_classname"))
        .distinct()
        .collect()
    )
    instance_types = [el["col"] for el in instance_types]
    print(f"Instance types: {', '.join(i for i in instance_types)}")

    oa_article = oa_withType.where(
        F.array_contains("instance_instancetype_classname", "Article")
    )
    n_oa_article = oa_article.count()
    n_art_withDOI = oa_article.where(F.col("doi").isNotNull()).count()

    oa_journal = oa_withType.where(
        F.array_contains("instance_instancetype_classname", "Journal")
    )
    n_oa_journal = oa_journal.count()

    oa_patent = oa_withType.where(
        F.array_contains("instance_instancetype_classname", "Patent")
    )
    n_oa_patent = oa_patent.count()
    print("--2--")

    print("_" * 110)
    print(
        f"{'Total':>45}\
            {'Percentage':>15}\
            {'Unique':>8}\
            {'P. Unique':>10}"
    )
    print("-" * 110)
    print(
        f"{'Number of elements:':<30}\
            {n_oa_filt:>10}\
            {n_oa_filt/n_oa_filt*100:>10.2f}%\
            {n_oa_filt_u:>10}\
            {n_oa_filt_u/n_oa_filt_u*100:>7.2f}%"
    )
    print(
        f"{'Elements with country info:':<30}\
            {n_oa_ctry:>10}\
            {n_oa_ctry/n_oa_filt*100:>10.2f}%\
            {'':>10}\
            {'':>10}"
    )
    print(
        f"{'Documents with DOI:':<30}\
            {n_oa_doi:>10}\
            {n_oa_doi/n_oa_filt*100:>10.2f}%\
            {n_oa_doi_u:>10}\
            {n_oa_doi_u/n_oa_filt_u*100:>7.2f}%"
    )
    print(
        f"{'Documents in SemanticScholar:':<30}\
            {n_oa_ss:>10}\
            {n_oa_ss/n_oa_filt*100:>10.2f}%\
            {n_oa_ss_u:>10}\
            {n_oa_ss_u/n_oa_filt_u*100:>7.2f}%"
    )
    print(
        f"{'Documents in SS with Country:':<30}\
            {n_oa_C_DOI:>10}\
            {n_oa_C_DOI/n_oa_filt*100:>10.2f}%"
    )
    print("_" * 110)
    print()
    print("_" * 80)
    print(
        f"{'Documents types':<15}\
            {'Articles':>13}\
            {'Journals':>13}\
            {'Patents':>13}"
    )
    print("-" * 80)
    print(
        f"{'Total':>15}\
            {n_oa_article:>13}\
            {n_oa_journal:>13}\
            {n_oa_patent:>13}"
    )
    print(
        f"{'% of total':>15}\
            {n_oa_article/n_oa_withType*100:>12.2f}%\
            {n_oa_journal/n_oa_withType*100:>12.2f}%\
            {n_oa_patent/n_oa_withType*100:>12.2f}%"
    )
    print(
        f"{'with DOI':>15}\
            {n_art_withDOI:>13}"
    )

    #################################################
    #### Counts
    #################################################

    # Country info
    oa_ctry = (
        oa_filtered.select("country_classid")
        .where(F.size("country_classid") > 0)
        .withColumn("num_countries", F.size(F.col("country_classid")))
    )

    df_n_ctry = oa_ctry.groupBy("num_countries").count().toPandas()
    df_n_byCtry = (
        oa_ctry.select(F.explode("country_classid").alias("country"))
        .groupBy("country")
        .count()
        .toPandas()
    )

    # #################################################
    # #### Plots
    # #################################################

    # plt.figure(figsize=(20, 8))
    # data = df_n_ctry.sort_values(by="num_countries").to_dict(orient="list")
    # plt.bar(data["num_countries"], data["count"], log=True)

    # plt.xticks(data["num_countries"])
    # plt.xlabel("Number of countries")
    # plt.ylabel("Publications")
    # plt.title("Publications by number of countries")
    # plt.show()

    # #################################################

    # plt.figure(figsize=(20, 8))
    # data = df_n_byCtry.sort_values(by="count").to_dict(orient="list")
    # topn = 10
    # plt.barh(data["country"][-topn:], data["count"][-topn:])

    # plt.xlabel("Publications")
    # plt.ylabel("Country")
    # plt.title(f"Publications by country (top {topn})")
    # plt.show()

    # #################################################

    # # Year info

    # # OA
    # oa_year = oa_filtered.dropDuplicates(subset=["doi"]).select(
    #     F.col("country_classid").alias("country"), F.year("acceptance").alias("year")
    # )
    # # Pub/Year
    # df_n_year = oa_year.groupBy("year").count().alias("count").toPandas()
    # # Pub(with country)/year
    # df_n_year_country = (
    #     oa_year.where(F.size("country") > 0)
    #     .groupBy("year")
    #     .agg(
    #         F.count("year").alias("count"),
    #         udf_combine_array(F.collect_list("country"), F.lit(False)).alias(
    #             "countries"
    #         ),
    #     )
    #     .toPandas()
    # )

    # # OA-SS
    # oa_year_ss = (
    #     oa_ss.select(
    #         "doi",
    #         "id",
    #         "ssid",
    #         F.col("country_classid").alias("country"),
    #         F.year("acceptance").alias("year"),
    #     )
    #     .dropDuplicates(subset=["ssid"])
    #     .dropDuplicates(subset=["id"])
    #     .dropDuplicates(subset=["doi"])
    #     .where(F.col("ssid").isNotNull())
    # )
    # # Pub/year
    # df_n_year_ss = (
    #     oa_year_ss.select("country", "year")
    #     .groupBy("year")
    #     .count()
    #     .alias("count")
    #     .toPandas()
    # )

    # plt.figure(figsize=(20, 8))

    # data = (
    #     df_n_year[df_n_year["year"].between(1980, 2022)]
    #     .sort_values(by="year")
    #     .dropna()
    #     .to_dict(orient="list")
    # )
    # plt.bar(
    #     np.array(data["year"]) - 0.2,
    #     data["count"],
    #     width=0.4,
    #     label="OpenAIRE",
    #     log=False,
    # )
    # data = (
    #     df_n_year_country[df_n_year_country["year"].between(1980, 2022)]
    #     .sort_values(by="year")
    #     .dropna()
    # )
    # plt.bar(
    #     np.array(data["year"]) + 0.2,
    #     data["count"],
    #     width=0.4,
    #     label="OA (with country)",
    #     log=False,
    # )

    # plt.legend()
    # plt.xlim([1979, 2023])
    # plt.xlabel("Year")
    # plt.ylabel("Publications")
    # plt.title(f"Publications by year")
    # plt.show()

    # #################################################

    # plt.figure(figsize=(20, 8))

    # df_ss_years = (
    #     ss.select("year", "doi")
    #     .dropDuplicates(subset=["doi"])
    #     .groupBy("year")
    #     .count()
    #     .toPandas()
    # )
    # data = (
    #     df_ss_years[df_ss_years["year"].between(1980, 2022)]
    #     .sort_values(by="year")
    #     .dropna()
    #     .to_dict(orient="list")
    # )
    # plt.bar(
    #     np.array(data["year"]) - 0.3, data["count"], width=0.3, label="SS", log=False
    # )

    # data = (
    #     df_n_year[df_n_year["year"].between(1980, 2022)]
    #     .sort_values(by="year")
    #     .dropna()
    #     .to_dict(orient="list")
    # )
    # plt.bar(
    #     np.array(data["year"]), data["count"], width=0.3, label="OpenAIRE", log=False
    # )

    # data = (
    #     df_n_year_ss[df_n_year_ss["year"].between(1980, 2022)]
    #     .sort_values(by="year")
    #     .dropna()
    #     .to_dict(orient="list")
    # )
    # plt.bar(
    #     np.array(data["year"]) + 0.3,
    #     data["count"],
    #     width=0.3,
    #     label="Intersection",
    #     log=False,
    # )

    # plt.legend()
    # plt.xlim([1979, 2023])
    # plt.xlabel("Year")
    # plt.ylabel("Publications")
    # plt.title(f"Publications by year")
    # plt.show()

    #################################################
    #### Output data
    #################################################
    columns = ["Num_cit", "Num_cit_wo_self"]
    row = [oa_cit, cit_wo_self]
    data = [row]
    df = spark.createDataFrame(data=data, schema=columns)
    df.printSchema()
    df.show(truncate=False)
    df.write.parquet(
        "/export/ml4ds/IntelComp/Datalake/OpenAIRE/metadata.parquet",
        mode="overwrite",
    )


if __name__ == "__main__":
    # Create session
    spark = SparkSession.builder.appName("WP3pipeline").getOrCreate()
    sc = spark.sparkContext
    print(sc.version)

    gen_OA_metadata(spark)
