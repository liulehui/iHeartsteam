# coding:utf-8

import pandas as pd
import scipy as sp
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyspark.sql import *
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors,SparseMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix,MatrixEntry
from pyspark.sql import SparkSession
import time
from math import sqrt

inputfile = '../users_items_100_LSH.csv'
itemfile = '../itemid2itemindex_100_LSH.csv'

def loadItemIndex(itemfile):

    itemIndexs = []
    with open(itemfile) as csvfile:
        itemReader = csv.reader(csvfile)
        next(itemReader)
        for row in itemReader:
            itemIndex = int(row[0])
            itemID = row[1]
            # itemindex_to_itemId[itemIndex] = itemID
            itemIndexs.append(itemIndex)
    return itemIndexs

def sparseify(users_num,user_index,ratings):
    feature = Vectors.sparse(users_num, user_index, ratings)
    return feature

def ANN(model: object, K: object, item: object, df: object) -> object:
    df_i = df[df['item_id'] == item]
    keys = []
    for index,row in df_i.iterrows():
        feature = sparseify(users_num, row["user_index"], row["ratings"])
        keys.append(feature)

    print("Approximately searching dfA for "+str(K)+" nearest neighbors of the key:")

    neighbors_index = []
    similarity = []

    for key in keys:
        #model.approxNearestNeighbors(dataB, key, K).show()
        neighbors = model.approxNearestNeighbors(dataB, key, K).toPandas()
        for index,row in neighbors.iterrows():
            neighbors_index.append(row['id'])
            similarity.append(1-row['distCol'])
    neighbors_index = neighbors_index[1:]
    similarity = similarity[1:]
    print(neighbors_index)
    print(similarity)
    similaritymap = dict(zip(neighbors_index,similarity))
    return neighbors_index,similarity,similaritymap

def getrating(userid,neighbors,similaritymap,df):
    rating = 0
    df_u = df[df['user_id'] == userid]
    bought_items = set(df_u['item_index'].unique().tolist())

    interset = bought_items.intersection(set(neighbors))
    print(len(interset))
    if len(interset) == 0:
        return df_u['playtime_forever'].mean()
    if len(interset) > 0:
        sum_similarity = 0
        for i in interset:
            sum_similarity += similaritymap[i]
            df_ui = df_u[df_u['item_index']==i]
            print(similaritymap[i])
            print(df_ui['playtime_forever'].mean())
            rating += similaritymap[i]*float(df_ui['playtime_forever'].mean())
        rating = rating/sum_similarity
        return rating


if __name__ == '__main__':

    data = []
    df = pd.read_csv(inputfile)
    df['playtime_forever'] = round(np.log(df['playtime_forever']+1),2)

    itemIndexs = loadItemIndex(itemfile)
    item_num = len(itemIndexs)


    users_num = len(df['user_id'].unique().tolist())

    count = 0
    for i in itemIndexs:
        count += 1
        if count % 1000 == 0:
            print(count)

        df_i = df[df['item_index'] == i]
        item_i_users = sorted(df_i['user_id'].unique().tolist())



        len_i = len(item_i_users)
        rating = [1.0]*len_i

        obervation = [i,item_i_users,rating]
        data.append(obervation)

    df = pd.DataFrame(data)
    df.columns = ['item_id','user_index','ratings']


    spark = SparkSession \
        .builder \
        .appName("LSH") \
        .getOrCreate()

    sqlCtx = SQLContext(spark)
    dataA = sqlCtx.createDataFrame(df)
    print("This is before sparsefify.\n")
    dataA.show()

    Item = Row('id','features')
    Item_seq = []
    for index,row in df.iterrows():
        print(index)
        feature = sparseify(users_num,row["user_index"],row["ratings"])
        row = Item(row['item_id'],feature)
        Item_seq.append(row)

    dataB = spark.createDataFrame(Item_seq)
    dataB.show()

    start = time.time()
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(dataB)

    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dataB).show()

    # start experiment
    ratingdata = pd.read_csv('../users_items_100.csv')
    ratingdata['playtime_forever'] = round(np.log(ratingdata['playtime_forever'] + 1), 2)
    y = ratingdata['playtime_forever']
    X = ratingdata[['user_id','item_index']]
    print(X.shape)
    print(y.shape)
    traindata, testdata = train_test_split(ratingdata,train_size=0.9999)
    X_train = traindata[['user_id','item_index']]
    y_train = traindata['playtime_forever']
    X_test = testdata[['user_id', 'item_index']]
    y_test = testdata['playtime_forever']
    print('Test_size = '+str(len(y_test)))
    predictions = []


    for index,row in X_test.iterrows():
        user_id = row['user_id']
        item_index = row['item_index']
        K = 20
        similarityMAP = {}
        if item_index in similarityMAP:
            prediction = getrating(user_id, similarityMAP[item_index]["neighbors_index"],similarityMAP[item_index]["similaritymap"], traindata)
            print('Prediction = ' + str(prediction))
            predictions.append(prediction)
        else:
            neighbors_index,similarity,similaritymap = ANN(model,K,item_index,df)

            similarityMAP[item_index] = {}
            similarityMAP[item_index]["neighbors_index"] = neighbors_index
            similarityMAP[item_index]["similaritymap"] = similaritymap

            prediction = getrating(user_id,similarityMAP[item_index]["neighbors_index"],similarityMAP[item_index]["similaritymap"],traindata)
            print('Prediction = '+str(prediction))
            predictions.append(prediction)

    print(len(predictions))

    # rmse

    rmse = sqrt(mean_squared_error(list(y_test),predictions))

    print('=====================final result=====================')

    print('RMSE: {}'.format(rmse))



    end = time.time()
    time_taken = str(int(end - start)) + " sec"

    print('Time: {}'.format(time_taken))

