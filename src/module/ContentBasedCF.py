#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:11:48 2018

@author: takumi_uchida
"""

import numpy as np
from scipy import spatial
from src.module import util


class ContentBasedCF:
    
    def __init__(self, knn=50):
        """
        Arguments:
            knn [int]:
                the number of similar id which is used in predicting.
        """
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
                ex) {'item00' : [0,1,0], 'item01': [.5,0,.5]]}
        """
        
        # set id transformer
        user_ids_transformer = util.id_transformer()
        item_ids_transformer = util.id_transformer()        
        self.user_ids = user_ids_transformer.fit_transform(user_ids)
        self.item_ids = item_ids_transformer.fit_transform(item_ids)        
        self.user_ids_transformer = user_ids_transformer
        self.item_ids_transformer = item_ids_transformer
        self.mean_ratings = np.mean(ratings)
        
        # set rating
        self.rating = {(u,i):r for u,i,r in zip(self.user_ids, self.item_ids, ratings)}
        
        # set item-attributes matrix
        dim_attribute = len(list(item_attributes.values())[0])
        n_item_ids = len(set(self.item_ids))
        self.item_attributes = np.zeros(shape=(n_item_ids, dim_attribute))
        for i, vec in item_attributes.items():
            _i = item_ids_transformer.transform([i], unknown=None)[0]
            if _i is not None:
                self.item_attributes[_i, :] = vec

        return self
        

    def predict(self, user_ids, item_ids, item_attributes, user_attributes='NotUse'):
        """
        Arguments:
            user_ids [array-like object]:
                pass
            item_ids [array-like object]:
                pass
            item_attributes [dict]:
                pass
        """
        tf_us = self.user_ids_transformer.transform(user_ids, unknown=None)
        item_attrs = [item_attributes[item_id] for item_id in item_ids]
        
        def get_predict(tf_u, item_attr):
            if tf_u is None:
                return self.mean_ratings
            else:
                return self._predict(tf_u, item_attr)
            
        predicted = map(get_predict, tf_us, item_attrs)
        return np.array(list(predicted))


    def _predict(self, u, attributes):
        """
        Arguments:
            u [int]:
                a transformed user_id.
            attributes [array like object]:
                a vector of item attributes.
        """
        knn_item_ids, similarities = self._get_similar_item_ids(attributes)        
        _rating = np.array([self.rating.get((u,i), np.nan) for i in knn_item_ids])
        bo_index = ~np.isnan(_rating)
        if any(bo_index):
            sum_similarities = similarities[bo_index].sum() if similarities[bo_index].sum() > 0 else 1
            predicted = (_rating[bo_index] * similarities[bo_index]).sum() / sum_similarities
        else:
            predicted = self.mean_ratings
        return predicted
            

    def _get_similar_item_ids(self, attributes):
        """
        Arguments:
            attributes [array like object]:
                a vector of item attributes.
                
        Example:
            attributes = [0,1,1]
        """
        # i = 2
        func1d = lambda array: cosine_similarity(array, attributes)
        sims = np.apply_along_axis(func1d, axis=1, arr=self.item_attributes)
        sims[np.isnan(sims)] = 0. # fill np.nan with 0.
        
        # filter top self.knn similarity score
        top_ids = sims.argsort()[:-self.knn:-1]
        
        return top_ids, sims[top_ids]
    
def cosine_similarity(array1, array2):
    sim = 1 - spatial.distance.cosine(array1, array2)
    return sim

    
    


if __name__ == '__main__':
    # Usage
    ## 下記のデータは明らかに、item_attribute = [負の影響, 影響なし, 正の影響]になっている。
    """
    # item_attributesの数は処理速度に無関係
    # 予測のサンプル数は処理速度に比例影響
    # item_idの異なり数が影響しているようだ。
    import numpy as np
    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)
    ratings  = np.random.choice(range(1,6), size=1000)
    item_attributes = {i:np.random.choice([0,1], size=18) for i in range(5000)}
    knn = 50
    CBCF = ContentBasedCF(knn)
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)

    user_ids = np.random.choice(range(100), size=1000)
    item_ids = np.random.choice(range(500), size=1000)

    CBCF.predict(user_ids, item_ids, item_attributes) 
    self = CBCF
    """
    
    user_ids = [1,1,1,1,5,5]
    item_ids = [1,2,3,4,2,4]
    ratings  = [5,5,3,1,5,1]
    item_attributes = {
            1:[0,1,1],
            2:[0,0,1],
            3:[1,1,1],
            4:[1,0,0],
            }
    
    knn = 4
    CBCF = ContentBasedCF(knn)
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)
    CBCF.predict(user_ids, item_ids, item_attributes)
    
    # outsample
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,1,1]})
    CBCF.predict([5], item_ids=[99], item_attributes={99:[0,1,1]})    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,0,1]})    
    CBCF.predict([5], item_ids=[99], item_attributes={99:[0,0,1]})    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,0,0]}) # attributeの合計が0なら予測は0になる。（問題なし）    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[1,1,0]})    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[1,0,0]})    
    CBCF.predict([5], item_ids=[99], item_attributes={99:[1,1,0]})    

    CBCF.predict([55], item_ids=[99], item_attributes={99:[0,1,1]})# 未知のユーザーは予測が平均値になる。
    
    
    