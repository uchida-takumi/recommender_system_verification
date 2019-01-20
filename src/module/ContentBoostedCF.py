#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is based on :
    https://www.cs.utexas.edu/~ml/papers/cbcf-aaai-02.pdf
    
この手法は、item_attributesに登録された新商品のレコメンドはできるが、新しいユーザーへのレコメンドは原則不可能。
"""

import numpy as np
from itertools import product

from src.module import util

""" プログラミング用
from src.module.ContentBoostedCF import ContentBoostedCF
from src.module.MF import MF

user_ids = user_ids[sep_indexs[0]]
item_ids = item_ids[sep_indexs[0]]
ratings   = values[sep_indexs[0]]

self = ContentBoostedCF(pure_content_predictor=MF(n_latent_factor=50))

user_attributes='NotUse'
"""

class ContentBoostedCF:
    
    def __init__(self, pure_content_predictor, knn=50):
        self.pure_content_predictor = pure_content_predictor
        self.knn = knn

    def fit(self, user_ids, item_ids, ratings, item_attributes, user_attributes='NotUse'):
        """
        Arguments:
            - user_ids [array-like-object]: 
                the array of user id.
            - item_ids [array-like-object]: 
                the array of item id.
            - ratings [array-like-object]: 
                the array of rating.
            - item_attributes [dictinary]:
                dictinary which key is item_id and value is vector of item attributes.
                if None, doesn't train on the attributes.
                ex) {'item00' : [0,1,0], 'item01': [.5,0,.5]]}
        """
        
        # set id transformer
        user_ids_transformer = util.id_transformer()
        item_ids_transformer = util.id_transformer()  
        user_ids_transformer.fit(user_ids)
        item_ids_transformer.fit(list(item_ids))
        item_ids_transformer.fit_update(list(item_attributes))
        
        self.user_ids = user_ids_transformer.transform(user_ids)
        self.item_ids = item_ids_transformer.transform(item_ids)
        self.user_ids_transformer = user_ids_transformer
        self.item_ids_transformer = item_ids_transformer
        self.ratings = ratings
        self.samples = {(u,i):r for u,i,r in zip(self.user_ids, self.item_ids, self.ratings)}
        
        self.item_attributes = {self.item_ids_transformer.transform([i])[0]:vec for i,vec in item_attributes.items()}

        # [eq1]get user-similarity matrx (P) based on user_ids, item_ids, ratings.
        self._fit_P()
        
        # fit pure content based model. after that, v(the pseudo user-rating) can be got.
        self._fit_pure_content_predictor()
        
        # fit the pseudo user-rating matrix (V)
        self._fit_V()

        return self
    
    def predict(self, user_ids, item_ids, item_attributes='NotUse', user_attributes='NotUse'):
        """
        Arguments:
            user_ids [array-like object]:
                pass
            item_ids [array-like object]:
                pass
        """
        results = []
        tf_us = self.user_ids_transformer.transform(user_ids, unknown=None)
        tf_is = self.item_ids_transformer.transform(item_ids, unknown=None)
        mean_values = np.mean(list(self.samples.values()))
        for tf_u,tf_i in zip(tf_us, tf_is):
            if (tf_u is None) or (tf_i is None):
                # user_id と item_attributes が未知の場合は全体平均で返却する。
                predicted = mean_values                
            else:
                predicted = self._predict(tf_u, tf_i) 
            results.append(predicted)

        return np.array(results)
        
        
    
    def _predict(self, a, i):
        """
        get prediction between active user a and item i.
        
        a:
            it means active user.
        i:
            it means a item.
        """

        v_a = self.V[a]
        sw_a = self._get_sw_a(a, _max=2)
        
        sigma_hwP_delta_v = 0
        sigma_hwP = 0
        
        # knn top similar user_ids of a        
        top_similar_user_ids = self.P[a].argsort()[:-self.knn:-1]
        
        
        for u in top_similar_user_ids:
            if u == a:
                continue
            # if similarity score is negative, ignore.
            if self.P[a,u] < 0:
                continue
            
            hw_au = self._get_hw_au(a, u)
            P_au = self.P[a,u]
            
            v_u = self.V[u]
            delta_v_ui = self._get_v(u,i) - v_u.mean()
            
            sigma_hwP_delta_v += hw_au*P_au*delta_v_ui
            sigma_hwP += hw_au*P_au
        
        c_ai = self._get_pure_content_prediction(a, i)
        numerator = sw_a * (c_ai - v_a.mean()) + sigma_hwP_delta_v
        denominator = sw_a + sigma_hwP
        p_ai = v_a.mean() + (numerator / denominator)
            
        return p_ai

    def _fit_P(self):
        """
        [eq1] create pure user-based similarity matrix.
        """
        n_user_ids = len(set(self.user_ids))
        n_item_ids = len(set(self.item_ids))
        
        R = np.zeros(shape=(n_user_ids, n_item_ids))
        
        for u_i,r in self.samples.items():
            R[u_i[0],u_i[1]] = r
        
        self.P = util.pearson_correlation_from_R(R)
    
    def _fit_pure_content_predictor(self):        
        self.pure_content_predictor.fit(self.user_ids, self.item_ids, self.ratings, item_attributes=self.item_attributes)

    def _fit_V(self):
        """
        create pseudo user-rating matrix whose dimmention is (user_id, item_id)
        """
        n_user_ids = len(set(self.user_ids))
        n_item_ids = len(set(self.item_ids))
        
        V = np.zeros(shape=(n_user_ids, n_item_ids))
        
        user_item_ids = list(product(range(n_user_ids), range(n_item_ids)))
        _user_ids = [_[0] for _ in user_item_ids]
        _item_ids = [_[1] for _ in user_item_ids]
        
        
        pure_content_predictions = self.pure_content_predictor.predict_high_speed_but_no_preprocess(_user_ids, _item_ids, item_attributes=self.item_attributes)
        
        """ 高速化の為に上に変更したが、下の記述のほうが一般性は高い。
        pure_content_predictions = self.pure_content_predictor.predict(_user_ids, _item_ids, item_attributes=self.item_attributes)
        """
        
        V = np.zeros(shape=(n_user_ids, n_item_ids))
        for u,i,p in zip(_user_ids, _item_ids, pure_content_predictions):
            V[u,i] = p
        for u,i in self.samples:
            V[u,i] = self.samples[(u,i)]
        self.V = V
        
    
    def _get_v(self, u, i):
        try:
            return self.V[u, i]
        except:
            return self._get_pure_content_prediction(u, i)
        """
        if (u,i) in self.samples:
            return  self.samples[(u,i)]
        else:
            return self._get_pure_content_prediction(u, i)
        """
        
    def _get_sw_a(self, a, _max=2):
        """ eq4 """
        n_a = np.sum(np.array(self.user_ids)==a)
        if n_a < 50:
            return (n_a / 50) * _max
        else:
            return _max
    
    def _get_hw_au(self, a, u):
        """ eq3 """
        m_a, m_u = self._get_m_u(a), self._get_m_u(u) 
        hm_au = (2 * m_a * m_u) / (m_a + m_u) if (m_a + m_u) > 0 else 0
        sg_au = self._get_sg_au(a, u)
        hw_au = hm_au + sg_au
        return hw_au
    
    def _get_m_u(self, u):
        n_u = np.sum(np.array(self.user_ids)==u)
        if n_u < 50:
            return n_u / 50
        else:
            return 1            
    
    def _get_sg_au(self, a, u):
        """ Significance Weighting factor """
        a_rated_item_ids = np.array(self.item_ids)[np.array(self.user_ids)==a]
        u_rated_item_ids = np.array(self.item_ids)[np.array(self.user_ids)==u]
        n = np.in1d(a_rated_item_ids, u_rated_item_ids).sum()
        if n < 50:
            return n/50
        else:
            return 1
                    
    def _get_pure_content_prediction(self, a, i):
        """
        a:
            it means active user.
        i:
            it means a item.
        """
        predicted = self.pure_content_predictor.predict([a], [i], item_attributes=self.item_attributes)
        return predicted[0]
    
            

    


if __name__ == '__main__':
    
    """
    # それなりの大量データでの結果
    from src.module.ContentBoostedCF import ContentBoostedCF
    from src.module.MF import MF

    import numpy as np
    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)
    ratings  = np.random.choice(range(1,6), size=1000)
    item_attributes = {i:np.random.choice([0,1], size=18) for i in range(5000)}

    CBCF = ContentBoostedCF(pure_content_predictor=MF(n_latent_factor=0))
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)

    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)

    CBCF.predict(user_ids, item_ids, item_attributes) 
    self = CBCF
    
    """

    # Usage
    ## 下記のデータは明らかに、item_attribute = [負の影響, 影響なし, 正の影響]になっている。

    user_ids = [1,1,1,1,5,5,7,8]
    item_ids = [1,2,3,4,2,4,1,4]
    ratings  = [5,5,3,1,5,1,5,1]
    item_attributes = {
            1:[0,1,1],
            2:[0,0,1],
            3:[1,1,1],
            4:[1,0,0],
            96:[0,1,1], #影響なしと正
            97:[1,1,0], #負と影響なし
            98:[1,0,1], #負と正
            99:[0,1,0], #影響なしのみ
            }

    from src.module.ContentBoostedCF import ContentBoostedCF
    from src.module.MF import MF

    self = ContentBoostedCF(pure_content_predictor=MF(n_latent_factor=0))
    self.fit(user_ids, item_ids, ratings, item_attributes)

    self.predict(user_ids, item_ids)
    self.predict([1,1,1,1], [96,97,98,99]) # 新しいitem_idの予測
    self.predict([5,5,5,5], [1,2,3,4]) # 学習済みのuser_idの全item_idの予測
    self.predict([7,7,7,7], [1,2,3,4]) # マイナーuser_idの全item_idの予測
    self.predict([8,8,8,8], [1,2,3,4]) # マイナーuser_idの全item_idの予測
    self.predict([99,99], [1,2]) # 新しいuser_idの予測は平均値
    
    self.V
    self._get_pure_content_prediction(1,1)
    

    