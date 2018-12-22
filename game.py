# coding:utf-8

import csv
import json
import pandas as pd
import numpy as np

def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)

class Game:

    itemID_to_name = {}
    name_to_itemID = {}
    itemindex_to_itemId = {}
    itemID_to_genre = {}

    itemIndexPath = '../itemid2itemindex_100.csv'
    genrePath = '../bundle_data.json'
    itemNamePath = '../australian_users_items.json'

    def loadItemName(self):

        counter = 0

        for i in parse(self.itemNamePath):
            dump = json.dumps(i)
            load = json.loads(dump)
            counter += 1
            if counter % 5000 == 0:
                print('Loading itemName')
                print(counter)

            for j in range(load['items_count']):

                #observation = [load['steam_id'],load['items'][j]['item_id'],round(load['items'][j]['playtime_forever']/60,2),round(load['items'][j]['playtime_2weeks']/60,2),1]
                if load['items'][j]['item_id'] not in self.itemID_to_name:
                    self.itemID_to_name[load['items'][j]['item_id']] = load['items'][j]['item_name']
                    self.name_to_itemID[load['items'][j]['item_name']] = load['items'][j]['item_id']

    def loadItemIndex(self):
        with open(self.itemIndexPath) as csvfile:
            itemReader = csv.reader(csvfile)
            next(itemReader)
            for row in itemReader:
                itemIndex = int(row[0])
                itemID = row[1]
                self.itemindex_to_itemId[itemIndex] = itemID

    def loadgenre(self):
        count = 0
        for i in parse(self.genrePath):
            dump = json.dumps(i)
            load = json.loads(dump)
            count += 1
            if count % 100 == 0:
                print('Loading genre')
                print(count)
            for item in load['items']:
                if item['item_id'] not in self.itemID_to_genre:
                    self.itemID_to_genre[item['item_id']] = item['genre']
                if item['item_id'] not in self.itemID_to_name:
                    self.itemID_to_name[item['item_id']] = item['item_name']

    def getNameandGenre(self,itemIndex):
        if itemIndex in self.itemindex_to_itemId:
            itemID = self.itemindex_to_itemId[itemIndex]

            if itemID in self.itemID_to_name:
                itemName = self.itemID_to_name[itemID]
            else:
                itemName = 'do not find Name'

            if itemID in self.itemID_to_genre:
                itemGenre = self.itemID_to_genre[itemID]
            else:
                itemGenre = 'do not find Genre'
            return itemName,itemGenre
        else:
            return 'do not find ID.'

if  __name__ == '__main__':
    item_index = 5690

    game = Game()
    game.loadItemName()
    game.loadItemIndex()
    game.loadgenre()

    name, genre = game.getNameandGenre(item_index)
    print(name)
    print(genre)