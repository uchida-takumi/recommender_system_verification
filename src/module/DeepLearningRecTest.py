#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:40:23 2020

@author: takumi_uchida
"""

import tensorflow as tf
import numpy as np
import pandas as pd

from src.module.knowledge_graph_attention_network.Model.utility.helper import *
from src.module.knowledge_graph_attention_network.Model.utility.batch_test import *
import os
from src.module.DeepLearningRec import KGAT
tf.set_random_seed(2019)
np.random.seed(2019)


config = dict()
config['n_users'] = data_generator.n_users
config['n_items'] = data_generator.n_items
config['n_relations'] = data_generator.n_relations
config['n_entities']  = data_generator.n_entities

if args.model_type in ['kgat', 'cfkg']:
    "Load the laplacian matrix."
    config['A_in'] = sum(data_generator.lap_list)

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['all_v_list'] = data_generator.all_v_list

args = parse_args()

self = KGAT()
data_config = config
pretrain_data = None
self.fit()

# test と train を読み込む
DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
filename = os.path.join(DIR_DATA, 'train_rating.csv')
train_rating = pd.read_csv(filename)
filename = os.path.join(DIR_DATA, 'test_rating.csv')
test_rating = pd.read_csv(filename)

train_user_ids = train_rating['UserID']
train_item_ids = train_rating['MovieID']
train_values   = train_rating['Rating']
test_user_ids = test_rating['UserID']
test_item_ids = test_rating['MovieID']
test_values   = test_rating['Rating']


random_user_ids = np.random.choice(train_user_ids, size=100000)
random_item_ids = np.random.choice(train_item_ids, size=100000)

#  各種スコアを確認する
train_predicted = self.predict(train_user_ids, train_item_ids)
print( pd.DataFrame(train_predicted).describe() )
test_predicted = self.predict(test_user_ids, test_item_ids)
print( pd.DataFrame(test_predicted).describe() )
random_predicted = self.predict(random_user_ids, random_item_ids)
print( pd.DataFrame(random_predicted).describe() )
""" test のスコアがrandom よりも低いといえる """

# どのテストサンプルの結果が悪かったか、分析するためにテストサンプルに情報を付加する。
user_info = train_rating.groupby(by='UserID').agg({'MovieID':len,'Rating':np.mean})
user_info.columns = ['user_cnt', 'user_mean_rating']
user_info['user_cnt_bins'] = pd.cut(user_info['user_cnt'], 3)
user_info = user_info.reset_index()

item_info = train_rating.groupby(by='MovieID').agg({'UserID':len,'Rating':np.mean})
item_info.columns = ['item_cnt', 'item_mean_rating']
item_info['item_cnt_bins'] = pd.cut(item_info['item_cnt'], 3)
item_info = item_info.reset_index()

test_rating['predicted'] = test_predicted
test_rating = test_rating.merge(user_info, how='left', on='UserID')
test_rating = test_rating.merge(item_info, how='left', on='MovieID')

test_rating.to_csv('check.csv')

# あああ
all_item_ids = np.unique(np.concatenate([train_item_ids, test_item_ids]))
all_user_ids = np.unique(np.concatenate([train_user_ids, test_user_ids]))
random_user_ids = sorted(list(set( np.random.choice(all_user_ids, size=50) )))

user_ids_, item_ids_ = [], []
for u in random_user_ids:
    user_ids_ += [u]*len(all_item_ids)
    item_ids_ += list(all_item_ids)

predicted_ = self.predict(user_ids_, item_ids_)

df = pd.DataFrame({'UserID':user_ids_, 'MovieID':item_ids_, 'predicted':predicted_})

train_rating['train_flg'] = 1
df = df.merge(train_rating[['UserID','MovieID','train_flg']], how='left', on=['UserID', 'MovieID'], suffixes=['', '_train'])
test_rating['test_flg'] = 1
df = df.merge(test_rating[['UserID','MovieID','test_flg']], how='left', on=['UserID', 'MovieID'], suffixes=['', '_test'])

df = df.merge(user_info, how='left', on='UserID')
df = df.merge(item_info, how='left', on='MovieID')

df.to_csv('check2.csv', index=False)


# モデル内部のembeddingを取得する
sess = tf.Session()
sess.run(tf.global_variables_initializer())

user_embed = sess.run( self.model.weights['user_embed'] )
user_embed = pd.DataFrame(user_embed)
user_embed.columns = ["embed%02d"%i for i in user_embed.columns]
user_embed = user_embed.reset_index().rename(columns={'index':'UserID'})
user_embed = user_embed.merge(user_info, how='left', on='UserID')
user_embed.to_csv('check_user_embed.csv', index=False)

item_embed = sess.run( self.model.weights['entity_embed'] )
item_embed = pd.DataFrame(item_embed)
item_embed.columns = ["embed%02d"%i for i in item_embed.columns]
item_embed = item_embed.reset_index().rename(columns={'index':'MovieID'})
item_embed = item_embed.merge(item_info, how='left', on='MovieID')
item_embed.to_csv('check_item_embed.csv', index=False)






if __name__ == 'how_to_use_this':
    import tensorflow as tf
    from src.module.DeepLearningRecTest import tfclass

    sess = tf.Session()
    self = tfclass()
    self.train(10)
    feed_dict = {self.y:100}
    self.predict(sess, feed_dict)



