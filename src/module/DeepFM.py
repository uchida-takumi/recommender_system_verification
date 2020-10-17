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
    def __init__(self, set_train_test_users, set_train_test_items, dict_genre=None, first_half_fit_only_fm=False, ctr_prediction=True):
        """
        import pandas as pd
        DIR_DATA = 'src/module/knowledge_graph_attention_network/Data/ml'
        df_train = pd.read_csv(os.path.join(DIR_DATA, 'train_rating.csv'))
        df_test = pd.read_csv(os.path.join(DIR_DATA, 'test_rating.csv'))

        set_train_test_users = set(np.concatenate([df_train['UserID'], df_test['UserID']]))
        set_train_test_items = set(np.concatenate([df_train['MovieID'], df_test['MovieID']]))
        dict_genre = pickle.load(open(os.path.join(DIR_DATA, 'genre.pickle'), 'rb'))
        
        self = DeepFM(set_train_test_users, set_train_test_items, dict_genre)
        self.dfm_params['epoch'] = 10
        self.dfm_params['batch_size'] = 64
        
        users   = df_train['UserID'].values
        items   = df_train['UserID'].values
        ratings = df_train['Rating'].values
        self.fit(users, items, ratings)
        predicted = self.predict(df_test['UserID'].values, df_test['UserID'].values)
        

        # MAE of test-set
        print( np.mean(np.abs(predicted - df_test['Rating'])) )

        # MAE of mean-prediction
        print( np.mean(np.abs(df_test['Rating'].mean() - df_test['Rating'])) ) 


        ## まぁ、実際のテストをクリアできればOKとする。
        """
        
        """
        参考として、Movielens1Mデータで検証されたハイパーパラメータは以下の通り
        Deep Matrix Factorization Approach for
        Collaborative Filtering Recommender Systems
        
        k(hidden-factor) = 8, γ(learning-rate) = 0.01, λ(regularization) = 0.045
        K = [9, 3, 3]; Γ= [0.01, 0.01, 0.01]; Λ = [0.1, 0.01, 0.1]
        """
        self.set_train_test_users = set(set_train_test_users)
        self.set_train_test_items = set(set_train_test_items)
        
        self.dict_genre = dict_genre
        self.first_half_fit_only_fm = first_half_fit_only_fm
        self.data_manager = Data_manager(set_train_test_users, set_train_test_items, dict_genre)
        feature_size, field_size = self.data_manager.get_feature_size_field_size()
        self.dfm_params = {
            "feature_size" : feature_size,
            "field_size" : field_size,
            "loss_type" : "mse", # "logloss" なら {0,1} の判別問題。 "mse" なら　regression。
            "use_fm": True, # fm-layer を使用
            "use_deep": True, # deep-layer を使用
            "embedding_size": 8,
            "dropout_fm": [1.0, 1.0],
            "deep_layers": [32, 32],
            "dropout_deep": [0.5, 0.5, 0.5],
            "deep_layers_activation": tf.nn.relu,
            "epoch": 30,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer_type": "adam",
            "batch_norm": 1,
            "batch_norm_decay": 0.995,
            "l2_reg": 0.0001,
            "l2_reg_embedding": 0.0001,
            "l2_reg_bias": 0.0001,
            "verbose": True,
            "eval_metric": mean_absolute_error,
            "greater_is_better": False, # 学習における損失スコアが大きい方が良いかどうか
            "random_seed": 2017,
            }
        self.ctr_prediction = ctr_prediction
        if self.ctr_prediction:
            self.dfm_params["loss_type"] = "logloss"
            

    def fit(self, users, items, ratings, test_users=[], test_items=[], test_ratings=[], **kargs):
        """
        users = [0,0,1]
        items = [0,3,3]
        ratings = [3.,4.,5.]
        """
        global_mean_bias_init = np.float32(np.mean(ratings))
        global_mean_bias_init = 0.01
        self.model = DeepFM_(**self.dfm_params, global_mean_bias_init=global_mean_bias_init, first_half_fit_only_fm=self.first_half_fit_only_fm)
        
        # もし、CTR予測の場合は、y=0のデータをランダム生成する。
        if self.ctr_prediction:
            users = list(users) + list(np.random.choice(list(set(users)), size=len(users)))
            items = list(items) + list(np.random.choice(list(set(items)), size=len(items)))
            ratings      = list((np.array(ratings)>0).astype(int)) + [0]*len(ratings)
            test_ratings = list((np.array(test_ratings)>0).astype(int))
        
        Xi, Xv = self.data_manager.transform_users_and_items_to_Xi_Xv(users, items)
        
        if len(test_users)>0:
            test_Xi, test_Xv = self.data_manager.transform_users_and_items_to_Xi_Xv(test_users, test_items)
            self.model.fit(Xi, Xv, ratings, test_Xi, test_Xv, test_ratings, early_stopping=True)
        else:
            self.model.fit(Xi, Xv, ratings, early_stopping=True, **kargs)
            
        # load data
        self.trained_users = list(set(users))
        self.trained_items = list(set(items))
        self.global_mean = self.model.predict(Xi, Xv).mean()
        
                
    def predict(self, users, items, *args, **kargs):
        Xi, Xv = self.data_manager.transform_users_and_items_to_Xi_Xv(users, items)
        predicted = self.model.predict(Xi, Xv)
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


        
if __name__ == 'how to use it.':
    ###########################
    # --- かなりシンプルなテスト ---
    sample_size = 1000
    users = np.random.choice(range(100), size=sample_size) 
    items = np.random.choice(range(100), size=sample_size)  
    genre_dict = None
    ratings = users - items
    
    self = DeepFM(set(users), set(items))
    self.dfm_params['batch_size'] = 64
    self.dfm_params['epoch'] = 100
    self.fit(users, items, ratings)   
    self.predict([10, 5, 10], [10, 10, 2]) # 正解は [0, -5, 8] である
    # 十分に小さなbatch_sizeかどうかは非常に重要
    # これは学習テストのロス減少によって確認できる。
    
    ###########################
    # --- シンプルなテスト1 ---
    sample_size = 1000
    n_user = 500
    n_item = 20
    users = np.random.choice(range(n_user), size=sample_size) 
    items = np.random.choice(range(n_item), size=sample_size)  
    
    user_embedding = {u:np.random.rand(5)-0.5 for u in range(n_user)}
    item_embedding = {i:np.random.rand(5)-0.5 for i in range(n_item)}
    
    def rating(u, i):
        return 10*sum(user_embedding[u] * item_embedding[i]) + 3
    
    ratings = [rating(u, i) for u,i in zip(users, items)]
    
    self = DeepFM(list(range(n_user)), list(range(n_item)))
    self.dfm_params['epoch'] = 100
    self.dfm_params['embedding_size'] = 200
    self.dfm_params['l2_reg'] = 0.0045
    self.fit(users, items, ratings)
    
    test_users = np.random.choice(range(n_user), size=sample_size) 
    test_items = np.random.choice(range(n_item), size=sample_size)  
    test_ratings = [rating(u, i) for u,i in zip(users, items)]

    predicted = self.predict(test_users, test_items)
    print( np.mean(np.abs(test_ratings - predicted)) )    
    print( np.mean(np.abs(test_ratings - np.mean(ratings))) )
    
    # scaler を導入すると改善されるか？　→　特に改善はされていない。
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit([[r] for r in ratings])
    s_ratings = scaler.transform([[r] for r in ratings])[:,0]
    
    self.fit(users, items, s_ratings)  
    predicted = self.predict(test_users, test_items)
    predicted = scaler.inverse_transform(predicted[:,None])
    print( np.mean(np.abs(test_ratings - predicted)) )    
    print( np.mean(np.abs(test_ratings - np.mean(ratings))) )

    ###########################
    # --- シンプルなテスト2　bias とembedding あり ---
    sample_size = 1000
    n_user = 500
    n_item = 20
    users = np.random.choice(range(n_user), size=sample_size) 
    items = np.random.choice(range(n_item), size=sample_size)  
    
    user_embedding = {u:np.random.rand(5)-0.5 for u in range(n_user)}
    item_embedding = {i:np.random.rand(5)-0.5 for i in range(n_item)}
    user_bias = {u:u/10 for u in range(n_user)} # 単純にidが大きいほどバイアスが大きい
    item_bias = {i:i for i in range(n_item)} # 単純にidが大きいほどバイアスが大きい
    
    def rating(u, i):
        return 10*sum(user_embedding[u] * item_embedding[i]) + user_bias[u] + item_bias[i]    
    ratings = [rating(u, i) for u,i in zip(users, items)]

    test_users = np.random.choice(range(n_user), size=sample_size) 
    test_items = np.random.choice(range(n_item), size=sample_size)  
    test_ratings = [rating(u, i) for u,i in zip(users, items)]
    
    self = DeepFM(list(range(n_user)), list(range(n_item)))
    self.dfm_params['epoch'] = 100
    self.dfm_params['embedding_size'] = 200
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    
    # 平均性能との比較
    predicted = self.predict(test_users, test_items)
    print( np.mean(np.abs(test_ratings - predicted)) )    
    print( np.mean(np.abs(test_ratings - np.mean(ratings))) )
    
    # オラクルとの比較
    predicted = self.predict([200]*n_item, list(range(n_item)))
    answer = [rating(200,i) for i in range(n_item)]
    print(predicted)
    print(answer)
    print(predicted - answer)
    
    ## 内部の embedding を確認する。
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    

    ###########################
    # --- シンプルなテスト3　head-tail-new ID ---
    sample_size = 1000
    n_user = 200
    n_item = 50
    ## id が後半になるほど学習セット中の出現率が低くなる。
    p_user = 1/np.array(range(1, n_user+1)); p_user /= p_user.sum()
    p_item = 1/np.array(range(1, n_item+1)); p_item /= p_item.sum()
    users = np.random.choice(range(n_user), size=sample_size, p=p_user) 
    items = np.random.choice(range(n_item), size=sample_size, p=p_item)  

    user_embedding = {u:np.random.rand(5)-0.5 for u in range(n_user)}
    item_embedding = {i:np.random.rand(5)-0.5 for i in range(n_item)}
    user_bias = {u:u/10 for u in range(n_user)} # 単純にidが大きいほどバイアスが大きい
    item_bias = {i:i for i in range(n_item)} # 単純にidが大きいほどバイアスが大きい
    def rating(u, i):
        return 10*sum(user_embedding[u] * item_embedding[i]) + user_bias[u] + item_bias[i]    
    ratings = [rating(u, i) for u,i in zip(users, items)]

    ## user=500 と item=20 はそれぞれ新規IDとなる
    test_users = np.random.choice(range(n_user), size=sample_size) 
    test_items = np.random.choice(range(n_item), size=sample_size)  
    test_ratings = [rating(u, i) for u,i in zip(users, items)]

    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 300
    self.dfm_params['embedding_size'] = 4
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    # 平均値予測との比較
    predicted = self.predict(test_users, test_items)
    print( np.mean(np.abs(test_ratings - predicted)) )    
    print( np.mean(np.abs(test_ratings - np.mean(ratings))) )

    ## 内部の embedding を確認する。
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])

    ## 可視化する(ID=500 まではユーザーで、それ以降はアイテム)
    import pandas as pd
    # [正常] 一部のembeddingがIDの増加に合わせて線形に変化している。これらはバイアス効果を一部学習している。
    pd.DataFrame(feature_embeddings).plot() 
    # [成功] DeepFM のバイアスの初期値を0付近にすることで、userのバイアスはオラクルに近くなった。
    # [?] itemのバイアスはオラクルと逆にidが増加するほど減少している → おそらくembeddingがバイアスを学習してしまったゆえか？
    pd.DataFrame(feature_bias).plot() 
    
    # 新規IDを確認する → ほぼ、初期値の0付近か？
    ## 新規ユーザー
    feature_embeddings[200]
    feature_bias[200]
    ## 新規アイテム
    feature_embeddings[-1]
    feature_bias[-1]
    
    ##############################################    
    # --- IDとは無関係なランダムなバイアスで学習してみる ---
    sample_size = 1000
    n_user = 200
    n_item = 50
    ## id が後半になるほど学習セット中の出現率が低くなる。
    p_user = 1/np.array(range(1, n_user+1)); p_user /= p_user.sum()
    p_item = 1/np.array(range(1, n_item+1)); p_item /= p_item.sum()
    users = np.random.choice(range(n_user), size=sample_size, p=p_user) 
    items = np.random.choice(range(n_item), size=sample_size, p=p_item)  
    user_bias = {u:np.random.rand() for u in range(n_user)} 
    item_bias = {i:np.random.rand() for i in range(n_item)} 
    user_embedding = {u:np.random.rand(5)-0.5 for u in range(n_user)}
    item_embedding = {i:np.random.rand(5)-0.5 for i in range(n_item)}
    def rating(u, i):
        return sum(user_embedding[u] * item_embedding[i]) + user_bias[u] + item_bias[i]    
    ratings = [rating(u, i) for u,i in zip(users, items)]

    ## user=500 と item=20 はそれぞれ新規IDとなる
    test_users = np.random.choice(range(n_user), size=sample_size) 
    test_items = np.random.choice(range(n_item), size=sample_size)  
    test_ratings = [rating(u, i) for u,i in zip(users, items)]    
    # ------------------------------
    ##############################################    

    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 100
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.001
    self.fit(users, items, ratings, test_users, test_items, test_ratings)

    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])

    """ デバック
    self.predict([1]*n_item, range(n_item))
    self.predict([0]*n_item, range(n_item))
    [rating(1, i) for i in range(n_item)]
    """
    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    """
    本テストは想定どおりの結果となり、成功したといえる。
    その成功要因は、以下の変更を加えたことによる。
    [1] 各id の embedding, bias の初期値を0付近のものに変更した。
    [2] l2_reg の対象として　embedding, bias を追加した。（おそらく、マイナーIDのweightが抑制されると思われるが、詳細は不明）
    """

    # --- パラメータごとの影響を確認する。
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['use_deep'] = False
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 


    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.001
    self.dfm_params['learning_rate'] = 0.001
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 

    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.100
    self.dfm_params['learning_rate'] = 0.001
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 

    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)))
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.001
    self.dfm_params['learning_rate'] = 0.010
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 


    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=True)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.010
    self.dfm_params['learning_rate'] = 0.010
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 


    # --- only fm 
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=True)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.010
    self.dfm_params['learning_rate'] = 0.010
    self.dfm_params['use_deep'] = False
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    
    # ---- high l2-reg
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=True)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.100
    self.dfm_params['learning_rate'] = 0.010
    self.dfm_params['use_deep'] = False
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 

    # ---- high learning_rate
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=False)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.0100
    self.dfm_params['l2_reg_embedding'] = 0.0100
    self.dfm_params['l2_reg_bias'] = 0.0100
    self.dfm_params['learning_rate'] = 0.0100
    self.dfm_params['use_deep'] = False
    
    self.fit(users, items, ratings, test_users, test_items, test_ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    
    ## 結論、頻度の違いがバイアスに影響を与えることはない。

    

    # ---- high learning_rate
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=False)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.0100
    #self.dfm_params['l2_reg_embedding'] = 0.0100
    #self.dfm_params['l2_reg_bias'] = 0.0100
    self.dfm_params['learning_rate'] = 0.0020
    self.dfm_params['use_deep'] = False
    self.dfm_params['batch_size'] = 32
    self.dfm_params['loss_type'] = 'mse'
    self.dfm_params['optimizer_type'] = 'sgd'
    #self.dfm_params['optimizer_type'] = 'adam'
    
    self.fit(users, items, ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    
    self.predict([0,0,150,150],[0,10,0,10])



    ##########################
    # MovieLensのCTR問題として定義し直して、性能を比較する
    import numpy as np 
    ctr_users = list(users) + list(np.random.choice(list(set(users)), size=len(users)))
    ctr_items = list(items) + list(np.random.choice(list(set(items)), size=len(items)))
    ctrs      = [1]*len(users) + [0]*len(users)
    
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=True)
    self.dfm_params['epoch'] = 20
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['l2_reg'] = 0.0010
    #self.dfm_params['l2_reg_embedding'] = 0.0020
    #self.dfm_params['l2_reg_bias'] = 0.0020
    self.dfm_params['learning_rate'] = 0.00010
    #self.dfm_params['use_deep'] = False
    self.dfm_params['batch_size'] = 16
    self.dfm_params['loss_type'] = 'logloss'
    self.dfm_params['greater_is_better'] = True
    
    self.fit(ctr_users, ctr_items, ctrs)

    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 

    self.predict([0,0,150,150],[0,10,0,10])
    
    ########################
    # CTR 対応型のテスト
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=False, ctr_prediction=True)
    self.dfm_params['epoch'] = 30
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['batch_size'] = 32
    self.dfm_params['dropout_fm'] = [0.5, 0.5]
    self.dfm_params['l2_reg'] = 0.0
    self.dfm_params['l2_reg_embedding'] = 0.0
    self.dfm_params['l2_reg_bias'] = 0.0
    
    self.fit(users, items, ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    pd.DataFrame(self.predict([200]*50,list(range(50)))).plot() # 新規ユーザーだけ常に一定になる。
    
    self.predict([0,0,150,150],[0,10,0,10])

    self.predict([50]*50,list(range(50)))
    self.predict([100]*50,list(range(50)))
    self.predict([150]*50,list(range(50)))
    self.predict([200]*50,list(range(50))) # 新規ユーザーだけ常に一定になる。
    self.predict([199]*50,list(range(50))) # 新規ユーザーだけ常に一定になる。
    self.predict([198]*50,list(range(50))) # 新規ユーザーだけ常に一定になる。
    self.predict([197]*50,list(range(50))) # 新規ユーザーだけ常に一定になる。
    self.predict(list(range(200)),[50]*200) # 新規ユーザーだけ常に一定になる。

    feature_embeddings[200]
    feature_bias[200]
    
    feature_embeddings[150]
    feature_bias[150]

    feature_embeddings[220]
    feature_embeddings[222]
    
    feature_embeddings[223]
    
    ########################
    # tensorflow の動作テスト
    weight = tf.Variable(initial_value=[[0,1,2,3], [0,10,20,30], [0,100,200,300]], trainable=True, name='test', dtype=tf.float32)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(weight)
    op = weight[1,3].assign(9999.)
    sess.run(op)
    sess.run(weight)
    
    ########################
    # 上手く行かなかったので、テスト
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=False, ctr_prediction=False)
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['deep_layers'] = [16, 16]
    self.dfm_params['l2_reg'] = 0.04 #0.0040
    self.dfm_params['l2_reg_embedding'] = 0.00000001 #0.001
    self.dfm_params['l2_reg_bias'] = 0.001 #0.001
    self.dfm_params['learning_rate'] = 0.0010 #0.001
    self.dfm_params['use_deep'] = True
    self.dfm_params['batch_size'] = 64
    self.dfm_params['loss_type'] = 'mse'
    #self.dfm_params['optimizer_type'] = 'sgd'
    
    self.fit(users, items, ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    #pd.DataFrame(feature_embeddings).plot() 
    #pd.DataFrame(feature_bias).plot() 
    pd.DataFrame(self.predict([200]*50,list(range(50)))).plot() # 新規ユーザーだけ常に一定になる。

    """ Best setting?    
    self = DeepFM(list(range(n_user+1)), list(range(n_item+1)), first_half_fit_only_fm=False, ctr_prediction=False)
    self.dfm_params['epoch'] = 10
    self.dfm_params['embedding_size'] = 4
    self.dfm_params['deep_layers'] = [16, 16]
    self.dfm_params['l2_reg'] = 0.04 #0.0040
    self.dfm_params['l2_reg_embedding'] = 0.00000001 #0.001
    self.dfm_params['l2_reg_bias'] = 0.000000001 #0.001
    self.dfm_params['learning_rate'] = 0.0010 #0.001
    self.dfm_params['use_deep'] = True
    self.dfm_params['batch_size'] = 64
    self.dfm_params['loss_type'] = 'mse'
    #self.dfm_params['optimizer_type'] = 'sgd'
    
    self.fit(users, items, ratings)
    
    feature_embeddings = self.model.sess.run(self.model.weights["feature_embeddings"])
    feature_bias = self.model.sess.run(self.model.weights["feature_bias"])
    concat_bias = self.model.sess.run(self.model.weights["concat_bias"])

    pd.DataFrame(feature_embeddings).plot() 
    pd.DataFrame(feature_bias).plot() 
    pd.DataFrame(self.predict([200]*50,list(range(50)))).plot() # 新規ユーザーだけ常に一定になる。
    """