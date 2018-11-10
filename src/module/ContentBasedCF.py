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
        
        # 速度改善のため、予測値のキャッシュを作成する。
        self.predict_cash = {}
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
        # predict
        results = []
        for u,i in zip(user_ids, item_ids):
            tf_u = self.user_ids_transformer.transform([u], unknown=None)[0]
            tf_i = self.item_ids_transformer.transform([i], unknown=None)[0]
            if tf_u is None:
                # user_idが未知の場合は全体平均で返却する。
                predicted = self.mean_ratings
            else:
                if tf_i is not None:
                    item_attr = self.item_attributes[tf_i]
                else:
                    item_attr = item_attributes.get(i, None)
                if item_attr is None:
                    raise 'No item_id in item_attributes.'
                predicted = self.predict_cash.get((tf_u, str(item_attr)), self._predict(tf_u, item_attr))
                self.predict_cash[(tf_u, str(item_attr))] = predicted
            results.append(predicted)

        return np.array(results)

    def _predict(self, u, attributes):
        """
        Arguments:
            u [int]:
                a transformed user_id.
            attributes [array like object]:
                a vector of item attributes.
        """
        knn_item_ids, similarities = self._get_similar_item_ids(attributes)
        C = 0
        predicted = 0
        for i,sim in zip(knn_item_ids, similarities):
            r = self.rating.get((u,i), None)
            if r is not None:                    
                predicted += sim * r
                C += sim
        predicted = predicted / C if C>0 else 0.
        
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
    CBCF = ContentBoostedCF(knn)
    CBCF.fit(user_ids, item_ids, ratings, item_attributes)
    CBCF.predict(user_ids, item_ids, item_attributes)
    
    # outsample
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,1,1]})    
    CBCF.predict([5], item_ids=[99], item_attributes={99:[0,1,1]})    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,0,1]})    
    CBCF.predict([5], item_ids=[99], item_attributes={99:[0,0,1]})    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[0,0,0]}) # attributeの合計が0なら予測は0になる。（問題なし）    
    CBCF.predict([1], item_ids=[99], item_attributes={99:[1,1,0]})    
    CBCF.predict([5], item_ids=[99], item_attributes={99:[1,1,0]})    

    CBCF.predict([55], item_ids=[99], item_attributes={99:[0,1,1]})# 未知のユーザーは予測が平均値になる。
    
    
    