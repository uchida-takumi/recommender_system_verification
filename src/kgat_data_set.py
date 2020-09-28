#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knowledge_graph_attention_network が実行可能なデータファイルを生成する。
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pickle
import os
import sys
import random
from collections import defaultdict
from sklearn import preprocessing

from src.module import util

############################
# Set validable
if len(sys.argv) > 1:
    k_hold = int(sys.argv[1])
else:
    k_hold = 0


# set_random_seed
random_seed = 12345
random.seed(random_seed)
np.random.seed(random_seed)

# set validation parameteres
train_test_days = 30
n_hold = 10
topN = [5,10,20]

# set DataDirectory
DIR_SAVE = 'src/module/knowledge_graph_attention_network/Data/ml'
if not os.path.exists(DIR_SAVE):
    os.makedirs(DIR_SAVE)

# load data
rating, user, movie = util.read_ml20m_data()

####################################
# split train and test indexes
one_day_sec = 60 * 60 * 24
max_seconds = rating.Timestamp.max()

seps = sorted([max_seconds - i * one_day_sec * train_test_days for i in range(n_hold+2)])
sep_indexs = [(seps[i-1]<=rating.Timestamp)&(rating.Timestamp<seps[i]) for i in range(1, n_hold+2)]
train_index, test_index = sep_indexs[k_hold], sep_indexs[k_hold+1]
train_rating, test_rating = rating[train_index], rating[test_index]

#####################################
# index replace to 0incliment

user_le, item_le = preprocessing.LabelEncoder(), preprocessing.LabelEncoder()

user_ids = list(set(list(train_rating['UserID']) + list(test_rating['UserID'])))
user_le.fit(user_ids)
item_ids = list(set(list(train_rating['MovieID']) + list(test_rating['MovieID'])))
item_le.fit(item_ids)
max_item_id = item_le.transform(item_ids).max()

train_rating['UserID'] = user_le.transform(train_rating['UserID'])
train_rating['MovieID'] = item_le.transform(train_rating['MovieID'])
test_rating['UserID'] = user_le.transform(test_rating['UserID'])
test_rating['MovieID'] = item_le.transform(test_rating['MovieID'])
movie = movie.loc[movie['MovieID'].isin(item_ids), :]
movie['MovieID'] = item_le.transform(movie['MovieID'])


######################################
# make train.txt
def write_rating(rating, filename):
    """
    train.txt あるいは test.txt に変換する。
    """
    lines = []
    for u in set(rating['UserID']):
        _item_ids = rating.loc[rating['UserID']==u, 'MovieID']
        line = str(u)+' ' + ' '.join([str(i) for i in _item_ids])
        lines.append(line)
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
        
write_rating(train_rating, filename=os.path.join(DIR_SAVE, 'train.txt'))            
write_rating(test_rating, filename=os.path.join(DIR_SAVE, 'test.txt'))            

#######################################    
# make kg_final.txt
## ジャンルリレーション用のデータを作成する
Genres = set()
for genres in movie['Genres']:
    for genre in genres.split('|'):
        Genres.add(genre)
Genres = [g for g in sorted(list(Genres)) if g!='(no genres listed)']
get_attributes = lambda genres: [int(Ge in genres.split('|')) for Ge in Genres]
item_attributes = {row['MovieID']:get_attributes(row['Genres']) for idx, row in movie.iterrows()}
item_attributes = {i:list(np.where(np.array(l)==1)[0]) for i,l in item_attributes.items()}
## ジャンルのIDはitemエンティティのID番号の追加で生成する。（エンティティ集合として扱う）
item_attributes = {i:[l+1+max_item_id for l in L] for i,L in item_attributes.items()}

## ジャンルというrelationはrelation_id=0として扱う。
lines = []
for i,A in item_attributes.items():
    pre_str = str(i) + ' 0 '  
    for a in A: 
        lines.append(pre_str + str(a))

## kg_final.txt として出力する
filename = os.path.join(DIR_SAVE, 'kg_final.txt')
with open(filename, 'w') as f:
    f.write('\n'.join(lines))

#######################################
# train_rating と test_rating をcsvで保存する。
filename = os.path.join(DIR_SAVE, 'train_rating.csv')
train_rating.to_csv(filename, index=False)
filename = os.path.join(DIR_SAVE, 'test_rating.csv')
test_rating.to_csv(filename, index=False)

########################################
# 出力設定ファイルを出力
import pickle
config = dict(k_hold=k_hold)
filename = os.path.join(DIR_SAVE, 'config.pickle')
pickle.dump(config, open(filename, 'wb'))




