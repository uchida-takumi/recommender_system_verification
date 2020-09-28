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

DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
DIR_output = 'pickle'


############################
# Set validable
random_seed = 12345

# set_random_seed
random.seed(random_seed)
np.random.seed(random_seed)

model_name = 'kgat'
train_test_days = 30
topN = [5,10,20]

config = pickle.load(open(os.path.join(DIR_DATA, 'config.pickle'), 'rb'))
k_hold = config['k_hold']


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
# 
# DeepLearningRecをimportした時点で、kgat_data_set.pyが出力した学習セットを読み込んでいるので、そのまま学習が可能。
from src.module.DeepLearningRec import KGAT
kgat = KGAT()
kgat.fit()

####################################
# --- varidation ---

n_random_selected_item_ids = 1000

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

validation_arrays =  get_CF_varidation_arrays(
            train_user_ids, train_item_ids, train_values,
            test_user_ids, test_item_ids, test_values,
            kgat,
            n_random_selected_item_ids=n_random_selected_item_ids,
            remove_still_interaction_from_test=True,
            random_seed=random_seed,
            topN=topN, need_fit=False
        )
file_name = os.path.join(DIR_output, 'validation__model_name={}__random_seed={}__train_test_days={}__topN={}__hold={}.pickle'.format(model_name, random_seed, train_test_days, topN, i))
pickle.dump(validation_arrays, open(file_name, 'wb'))
print("END on {}".format(arg_dict))



