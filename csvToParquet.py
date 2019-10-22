import os
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession \
    .builder \
    .appName("spark-parquet-example") \
    .config("spark.master", "local") \
    .config("spark.executor.heartbeatInterval", "10000") \
    .config("spark.network.timeout", "10001") \
    .getOrCreate()

# Create a variable to store the base path to the data directory
base_path = "/mnt/499f5b4c-132a-4775-a414-ce6eba504896/Downloads/"


# Create a variable to store the list of names for the data sample files
files = [
'58fba123-ee58-4fa5-b8a5-8d94be579593.csv'
]

# Read a data set from file
for i in files:
    data = spark \
        .read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .format("csv") \
        .load("{}{}".format(base_path, i))

    data \
        .write \
        .parquet("{}/Parquet/{}.parquet".format(base_path.replace('Downloads/', ''), i[:-4].strip()))

    # Get the file size of the original csv format file
    print("File size (csv): {}".format(os.path.getsize("{}{}".format(base_path, files[0]))))

    # Get the size of the parquet format file structure
    print("File size (parquet): {}".format(os.path.getsize("{}/Parquet/{}.parquet".format(base_path, files[0][:-4].strip()))))
