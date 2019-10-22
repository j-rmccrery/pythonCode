from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local') \
    .appName('spark-parquet-example') \
    .config('spark.executor.memory', '10gb') \
    .config("spark.cores.max", "8") \
    .getOrCreate()

sc = spark.sparkContext

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

files = ["4a1f94d6-1257-4b5d-baed-6f468822713c"]
data = sqlContext.read.parquet('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/Parquet/'+ files[0] +'.parquet')

from pyspark.sql.functions import lower
data.user_id = lower(data.user_id)

import os, glob
dNew = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/mparticle_/parquet'
os.chdir(dNew)
doneFiles = glob.glob('*')

os.chdir('/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/mparticle_/2018-09-10')
files = glob.glob('*')
newPath = '/mnt/499f5b4c-132a-4775-a414-ce6eba504896/data/mparticle_/2018-09-10'

for i in files:
    if i.replace('csv', 'parquet') in doneFiles:
        continue
    newData = spark \
        .read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .format("csv") \
        .load("{}/{}".format(newPath, i))
        
    newData.user_id = lower(newData.user_id)

    x = data.join(newData, "user_id")    
    x \
    .write \
    .parquet("{}/{}.parquet".format(dNew, i[:-4].strip()))