# coding:utf-8

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import numpy as np

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("benchmarkALS") \
        .config("spark.executor.cores", '4') \
        .getOrCreate()

    lines = spark.read.option("header", "true").csv("../users_items_30percent.csv").rdd
    # this read do not read any data in, just create a placeholder
    # and told the RDD where it will read the data from

    isPlayedRDD = lines.map(lambda p: Row(user_id=int(p[0]), item_index=int(p[1]),playtime_forever=float(np.log(float(p[2])+1))))

    # convert RDD into a dataset

    ratings = spark.createDataFrame(isPlayedRDD)

    (training, test) = ratings.randomSplit([0.8, 0.2])  # until now, nothing is happening here

    als = ALS(maxIter=5, regParam=0.1, userCol="user_id", itemCol="item_index", ratingCol="playtime_forever",
              coldStartStrategy="drop")

    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="playtime_forever",
                                    predictionCol="prediction")

    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    userRecs = model.recommendForAllUsers(10)  # top_N recommendation

    user85Recs = userRecs.filter(userRecs['user_id'] == 85).collect()

    spark.stop()


    for row in user85Recs:
        for rec in row.recommendations:
            print(rec.item_index)
