#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KGAT用に特化して、評価pickleを出力する。
データを指定したn_holdに分割し、modelsごとにテストを行う。
それぞれのテスト結果をpickleファイルにして出力する。

kgat_data_set.py によって、すでに学習データと予測データは分離されている。
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pandas as pd
import pickle
import os
import sys
import random
import pickle

from src.module import util
from src.module.get_CF_varidation_arrays import get_CF_varidation_arrays

############################
# Set validable
random_seed = 12345

# set_random_seed
random.seed(random_seed)
np.random.seed(random_seed)

topN = [5,10,20]
DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
config = pickle.load(open(os.path.join(DIR_DATA, 'config.pickle'), 'rb'))
k_hold = config['k_hold']
#train_test_days = config['train_test_days']
train_test_days = 30
############################
# Read DataSet
# データセットはすでに分離しているため処理を省略する
pass

####################################
# set dataset
pass

####################################
# split train and test indexes
pass

####################################
# --- Set models ---
 
# DeepLearningRecをimportした時点で、kgat_data_set.pyが出力した学習セットを読み込んでいるので、そのまま学習が可能。
from src.module.DeepFM import DeepFM
import pandas as pd

DIR_output = 'pickle'
DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_rating.csv'))
df_test = pd.read_csv(os.path.join(DIR_DATA, 'test_rating.csv'))

train_user_ids = df_train['UserID']
train_item_ids = df_train['MovieID']
train_values   = df_train['Rating']
test_user_ids  = df_test['UserID']
test_item_ids  = df_test['MovieID']
test_values    = df_test['Rating']

set_train_test_users = set(np.concatenate([train_user_ids, test_user_ids]))
set_train_test_items = set(np.concatenate([train_item_ids, test_item_ids]))
dict_genre = pickle.load(open(os.path.join(DIR_DATA, 'genre.pickle'), 'rb'))



####################################
# --- varidation ---
n_random_selected_item_ids = 1000

model_name = 'deepfm_genre'
model = DeepFM(set_train_test_users, set_train_test_items, dict_genre)
#model.dfm_params['epoch'] = 5
model.fit(df_train['UserID'], df_train['UserID'], df_train['Rating'])
validation_arrays =  get_CF_varidation_arrays(
            train_user_ids, train_item_ids, train_values,
            test_user_ids, test_item_ids, test_values,
            model,
            n_random_selected_item_ids=n_random_selected_item_ids,
            remove_still_interaction_from_test=True,
            random_seed=random_seed,
            topN=topN, need_fit=False
        )
file_name = os.path.join(DIR_output, 'validation__model_name={}__random_seed={}__train_test_days={}__topN={}__hold={}.pickle'.format(model_name, random_seed, train_test_days, topN, k_hold))
pickle.dump(validation_arrays, open(file_name, 'wb'))


# ジャンル情報なし
model_name = 'deepfm'
model = DeepFM(set_train_test_users, set_train_test_items, dict_genre=None)
#model.dfm_params['epoch'] = 5
model.fit(df_train['UserID'], df_train['UserID'], df_train['Rating'])
validation_arrays =  get_CF_varidation_arrays(
            train_user_ids, train_item_ids, train_values,
            test_user_ids, test_item_ids, test_values,
            model,
            n_random_selected_item_ids=n_random_selected_item_ids,
            remove_still_interaction_from_test=True,
            random_seed=random_seed,
            topN=topN, need_fit=False
        )
file_name = os.path.join(DIR_output, 'validation__model_name={}__random_seed={}__train_test_days={}__topN={}__hold={}.pickle'.format(model_name, random_seed, train_test_days, topN, k_hold))
pickle.dump(validation_arrays, open(file_name, 'wb'))

print("END on {}".format(file_name))



