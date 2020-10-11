"""
# install the package
pip install deepctr

# tutorial
https://deepctr-doc.readthedocs.io/en/latest/Quick-Start.html#getting-started-4-steps-to-deepctr

# github
https://github.com/shenweichen/DeepCTR

しかし、これは binary しか出来ないので適応不可能。
binary を無理矢理適応させるばあいは、非クリックデータを何らかの方法で生成する必要がある。

# ---- 次のアイデア ----
# github
https://github.com/ChenglongChen/tensorflow-DeepFM
"""

import tensorflow as tf
import os
import pickle
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from src.module.tensorflow_DeepFM.DeepFM import DeepFM as DeepFM_



# インターフェース
class DeepFM:
    def __init__(self, set_train_test_users, set_train_test_items, dict_genre=None):
        """
        import pandas as pd
        DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
        df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_rating.csv'))
        df_test = pd.read_csv(os.path.join(DIR_DATA, 'test_rating.csv'))

        set_train_test_users = set(np.concatenate([df_train['UserID'], df_test['UserID']]))
        set_train_test_items = set(np.concatenate([df_train['MovieID'], df_test['MovieID']]))
        dict_genre = pickle.load(open(os.path.join(DIR_DATA, 'genre.pickle'), 'rb'))
        
        self = DeepFM(set_train_test_users, set_train_test_items, dict_genre)
        self.dfm_params['epoch'] = 5
        self.fit(df_train['UserID'], df_train['UserID'], df_train['Rating'])
        
        predicted = self.predict(df_test['UserID'], df_test['UserID'])
        

        # MAE of test-set
        print( np.mean(np.abs(predicted - df_test['Rating'])) )

        # MAE of mean-prediction
        print( np.mean(np.abs(df_test['Rating'].mean() - df_test['Rating'])) ) 
        ## 平均予測よりもMAEが悪い。。。
        ## まぁ、実際のテストをクリアできればOKとする。
        """
        
        """
        参考として、Movielens1Mデータで検証されたハイパーパラメータは以下の通り
        Deep Matrix Factorization Approach for
        Collaborative Filtering Recommender Systems
        
        k(hidden-factor) = 8, γ(learning-rate) = 0.01, λ(regularization) = 0.045
        K = [9, 3, 3]; Γ= [0.01, 0.01, 0.01]; Λ = [0.1, 0.01, 0.1]
        """
        self.dict_genre = dict_genre
        self.data_manager = Data_manager(set_train_test_users, set_train_test_items, dict_genre)
        feature_size, field_size = self.data_manager.get_feature_size_field_size()
        self.dfm_params = {
            "feature_size" : feature_size,
            "field_size" : field_size,
            "loss_type" : "mse", # "logloss" なら {0,1} の判別問題。 "mse" なら　regression。
            "use_fm": True,
            "use_deep": True,
            "embedding_size": 8,
            "dropout_fm": [1.0, 1.0],
            "deep_layers": [32, 32],
            "dropout_deep": [0.5, 0.5, 0.5],
            "deep_layers_activation": tf.nn.relu,
            "epoch": 30,
            "batch_size": 512,
            "learning_rate": 0.01,
            "optimizer_type": "adam",
            "batch_norm": 1,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.045,
            "verbose": True,
            "eval_metric": mean_absolute_error,
            "random_seed": 2017,
        }

    def fit(self, users, items, ratings, *args, **kargs):
        """
        users = [0,0,1]
        items = [0,3,3]
        ratings = [3.,4.,5.]
        """
        self.model = DeepFM_(**self.dfm_params)
        Xi, Xv = self.data_manager.transform_users_and_items_to_Xi_Xv(users, items)
        # load data
        self.model.fit(Xi, Xv, ratings)
        self.trained_users = list(set(users))
        self.trained_items = list(set(items))
        self.global_mean = self.model.predict(Xi, Xv).mean()

    def predict(self, users, items, *args, **kargs):
        Xi, Xv = self.data_manager.transform_users_and_items_to_Xi_Xv(users, items)
        predicted = self.model.predict(Xi, Xv)
        
        trained_user_index = np.in1d(np.array(users), self.trained_users)
        trained_item_index = np.in1d(np.array(items), self.trained_items)
        if self.dict_genre:
            predicted[~trained_user_index] = self.global_mean
        else:
            predicted[~trained_user_index] = self.global_mean
            predicted[~trained_item_index] = self.global_mean
        return predicted

# prepare training and validation data in the required format
class Data_manager:
    def __init__(self, users, items, dict_genre=None):
        """
        users [array like object]:
            train, test set に含まれる user_id
        items [array like object]:
            train, test set に含まれる item_id
        dict_genre [dictionary]:
            ex) {item_id: [genre_id1, genre_id2]}
        
        tensorflow_DeepFM/example 内部のプログラム、特にDataReader.pyを読み、データの形式を解読した。
        結論として、 item, user, genre  の各IDは以下のように変換すればよい。
        1) user = {0,1,2} → {0,1,2} *未変更
        2) item = {0,1} → {3,4} *userからのインクリメントID
        3) genre = {0,1} → {5,6} *itemからのインクリメントID
        4) a interaction-sample [u,i,g] = [0,1,0]→[0,4,5]
        5) Xi_train (X-index trainset) = [変換した[u,i,g]1, 変換した[u,i,g]2, ...]
        6) Xv_train (X-value trainset) = [[1.,1.,1.], [1.,1.,1.], ...]
          user,item,genre はカテゴリ変数なのですべて1.となる。
        7) y_train = [rating-score1, rating-score2, ...] *変換不要
    
        EXAMPLE
        -------------
        import pandas as pd
        df_rating = pd.read_csv(os.path.join(DIR_DATA, 'train_rating.csv'))
        dict_genre = pickle.load(open(os.path.join(DIR_DATA, 'genre.pickle'), 'rb'))
        users = df_rating['UserID']
        items = df_rating['MovieID']

        self = Data_manager(users, items, dict_genre=dict_genre)
        """        
        self.dict_genre = dict_genre
        # インクリメントインデックスを生成するオブジェクト self.inclement_index を生成する。
        if dict_genre:
            dict_genre = {i:gs for i,gs in dict_genre.items() if i in items}
            n_genre = max([max(gs) for i,gs in dict_genre.items() if gs]) + 1
            genres = list(range(n_genre))
        else:
            dict_genre = {}
            n_genre = 0
            genres  = []
            
        self.inclement_index = inclement_index(users, items, genres)

        # userとitemをインクリメントIDに変更する
        dict_genre = {self.inclement_index.transform([i], field='item')[0]:gs for i,gs in dict_genre.items()}

        # user, itemはそれぞれで2フィールド、ジャンルはジャンルラベルごとに別々のフィールドにわける。
        self.re_dict_genre = {} 
        for i,gs in dict_genre.items():
            # re_dict は　{item_id:(field_id, genru_id)}となる。
            genre_one_hot_vec = [0] * n_genre
            for g in gs:
                genre_one_hot_vec[g] = 1 # カテゴリ変数はかならず整数の1とする。
            self.re_dict_genre[i] = genre_one_hot_vec
                
        self.genre_indexes = self.inclement_index.transform(genres, field='genre')
        self.feature_size = self.inclement_index.get_feature_size()
        self.field_size = 2 + n_genre
        
    def get_feature_size_field_size(self):
        return self.feature_size, self.field_size

    def transform_users_and_items_to_Xi_Xv(self, users, items):
        """
        users = [0,0,1]
        items = [1,5,5]
        
        """
        Xi, Xv = [], []
        users = self.inclement_index.transform(users, field='user')
        items = self.inclement_index.transform(items, field='item')
        for u,i in zip(users, items):
            if self.dict_genre:
                Xi.append([u, i] + self.genre_indexes)
                Xv.append([1, 1] + self.re_dict_genre[i])
            else:
                Xi.append([u, i])
                Xv.append([1, 1])                
        return Xi, Xv
        
    

class inclement_index:
    def __init__(self, users, items, genres=[]):
        """
        users = ['u0','u1',3]
        items = ['i0', 3]
        genres = ['pop', 'sf']
        
        self = inclement_index(users, items, genres)
        self.transform(['u0', 'u1', 3], field='user', inverse=False)
        self.transform(['i0', 3], field='item', inverse=False)
        self.transform(['pop', 'sf'], field='genre', inverse=False)
        
        transformed = self.transform(['u0', 'u1', 3], field='user', inverse=False)
        self.transform(transformed, field='user', inverse=True)

        """
        users = set(users)
        items = set(items)
        genres = set(genres)
        self.increment_cnt = 0
        self.user_dict = {u:self.get_incremate_index() for u in users}
        self.user_inverse_dict = {v:k for k,v in self.user_dict.items()}
        self.item_dict = {i:self.get_incremate_index() for i in items}
        self.item_inverse_dict = {v:k for k,v in self.item_dict.items()}
        self.genre_dict = {g:self.get_incremate_index() for g in genres}
        self.genre_inverse_dict = {v:k for k,v in self.genre_dict.items()}

    def transform(self, xs, field='user', inverse=False):
        """
        xs = [0,2]

        self.transform(xs, type='user')
        """
        if inverse:
            if field == 'user':
                _dict = self.user_inverse_dict
            elif field == 'item':
                _dict = self.item_inverse_dict
            elif field == 'genre':
                _dict = self.genre_inverse_dict
        else:
            if field == 'user':
                _dict = self.user_dict
            elif field == 'item':
                _dict = self.item_dict
            elif field == 'genre':
                _dict = self.genre_dict

        return [_dict[x] for x in xs]        
    
    def get_incremate_index(self):
        now_index = copy.deepcopy(self.increment_cnt)
        self.increment_cnt += 1
        return now_index
    
    def get_feature_size(self):
        return self.increment_cnt
        


