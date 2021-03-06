# coding:utf-8
import json
import ast
import pandas as pd
import numpy as np
import scipy as sp
import csv

np.random.seed(4571)

def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)

def sampledata(filepath,mypercent):

    data = []
    steamID_to_userID = {}


    counter = 0

    for i in parse(filepath):
        dump = json.dumps(i)
        load = json.loads(dump)
        counter += 1
        if counter % 5000 == 0:
            print(counter) # number of rows i.e. number of users in the dataset

        # generate mapping between steam_id and user_id in the original data json file
        if load['steam_id'] not in steamID_to_userID:
            steamID_to_userID[load['steam_id']] = load['user_id']

        data_i = []  # all the games user i played

        for j in range(load['items_count']):

            observation = [load['steam_id'],load['items'][j]['item_id'],round(load['items'][j]['playtime_forever']/60,2),round(load['items'][j]['playtime_2weeks']/60,2),1]

            data_i.append(observation)

        if len(data_i) == 0:
            continue

        df_i = pd.DataFrame(data_i)
        # print(df_i.shape)
        df_i.columns = ['steam_id', 'item_id','playtime_forever','playtime_2weeks','isplayed']
        if len(data_i) > 10:
            df_i = df_i.sample(frac = mypercent) # sample data to train set
        data.append(df_i)

    df = pd.concat(data) # data we sampledR
    df.columns = ['steam_id', 'item_id','playtime_forever','playtime_2weeks','isplayed']

    user_id = pd.factorize(df.steam_id)
    item_index = pd.factorize(df.item_id)
    steamid2userid = dict(zip(user_id[0],df.steam_id))
    itemid2itemindex = dict(zip(item_index[0],df.item_id))

    df['user_id'] = user_id[0]
    df['item_index'] = item_index[0]

    df = df.drop(['steam_id'], axis=1)
    df = df.drop(['item_id'], axis=1)



    return df,steamid2userid,itemid2itemindex,steamID_to_userID


if __name__ == '__main__':
    filepath = '../australian_users_items.json'
    percent = 1
    dataset, steamid2userid, itemid2itemindex,steamID2userID= sampledata(filepath,percent)

    playdata = dataset[['user_id','item_index','playtime_forever']]
    print(playdata.shape)
    # playdata = playdata[playdata['playtime_forever'] > 0]
    # print(playdata.shape)


    with open('../steamid2userid_100_LSH.csv', 'w') as f:
        for key in steamid2userid.keys():
            f.write("%s,%s\n" % (key, steamid2userid[key]))
    with open('../itemid2itemindex_100_LSH.csv', 'w') as f:
        for key in itemid2itemindex.keys():
            f.write("%s,%s\n" % (key, itemid2itemindex[key]))
    with open('../steamID2userID_full_LSH.csv', 'w') as f:
        for key in steamID2userID.keys():
            f.write("%s,%s\n" % (key, steamID2userID[key]))



    playdata.to_csv('../users_items_100_LSH.csv',index=False)





