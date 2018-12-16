# coding:utf-8

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("benchmarkALS") \
        .getOrCreate()

    lines = spark.read.option("header", "true").csv("../users_items_20percent.csv").rdd
    # this read do not read any data in, just create a placeholder
    # and told the RDD where it will read the data from

    isPlayedRDD = lines.map(lambda p: Row(userId=int(p[5]), itemId=int(p[6]),isPlayed=float(p[3])))

    # convert RDD into a dataset

    ratings = spark.createDataFrame(isPlayedRDD)

    (training, test) = ratings.randomSplit([0.8, 0.2])  # until now, nothing is happening here

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="itemId", ratingCol="isPlayed",
              coldStartStrategy="drop")
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="isPlayed",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))


