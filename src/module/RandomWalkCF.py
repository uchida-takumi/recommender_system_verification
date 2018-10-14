#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module was on:
    [Hilmi 2008: A Random Walk Method for Alleviating the Sparsity Problem in Collaborative Filtering]
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RandomWalkCF:
    def __init__(self):
        """
        This is Simple Random Walk Collaborative Filtering.
        """
        pass
            
    def fit(self
            ,user_ids, item_ids, values
            ,alpha=0.80, beta=1.00 
            ,DEBUG=False
            ):
        """
        Argument:
            user_ids [np.array; (n_samples,)]:
                the id must be int.
            item_ids [np.array; (n_samples,)]:
                the id must be int.
            values [np.array; (n_samples, )]:
                the values resulted from iteractions between user_id and item_id.
            alpha [float, 0.~1.]:
                the probability to continue random walk at each step.
            beta [float, 0.~1.]:
                the probability to move from item to item as random walk.
                with probability (1-beta), jumpts to arbitrary item.
                                
        """
        # get item-item similar matrix as S from the user-item rating matrix R
        Rating = pd.DataFrame({'user_ids':user_ids, 'item_ids':item_ids, 'values':values})
        R = pd.pivot_table(Rating, 'values', 'user_ids', 'item_ids', aggfunc='mean').fillna(0)

        S = np.nan_to_num(cosine_similarity(R.T))
        m = S.shape[0]

        P = list()
        for i in range(m):
            # i =2
            sum_ = S[i, :].sum()
            if sum_:
                prob = lambda j : beta * S[i,j] / sum_ + (1 - beta) / m
            else:
                prob = lambda j : 0 + (1 - beta) / m
            map_ge = map(prob, range(m))
            P.append(list(map_ge)) 

        P = pd.DataFrame(P)
                            
        hat_P = np.dot(alpha * P, np.linalg.inv(np.eye(m) - alpha * P))
        hat_R = np.dot(R, hat_P)
        hat_R = self._scale(hat_R, R)
        
        self.hat_R = pd.DataFrame(hat_R, index=R.index, columns=R.columns)
                        

    def predict(self, user_ids, item_ids):
        """
        Return predicted valuews of interactions between user_ids and item_ids.

        If, unlearned ids are in user_ids or item_ids, will be returned the appropriate value.
        The latent factors of unlearned id is as average of learned ids.
        The bias of unlearned id is 0.
        The attributes of id which is not in self.*_atributes is as 0 vectors.

        Argument:
            user_ids [np.array; (n_samples,)]:
                the id must be (0 ~ n_user).
            item_ids [np.array; (n_samples,)]:
                the id must be (0 ~ n_item).
        """        
        return_list = [self._predict(user_id, item_id) for user_id, item_id in zip(user_ids, item_ids)]
        return np.array(return_list)                

    def _scale(self, hat_R, R):
        for i in range(hat_R.shape[0]):
            min_, max_ = R.iloc[i, :].min(), R.iloc[i, :].max()
            hat_r = hat_R[i, :]
            hat_r_scaled_0_1     = (hat_r - hat_r.min()) / (hat_r.max() - hat_r.min())
            hat_r_scaled_min_max = hat_r_scaled_0_1 * (max_ - min_) + min_
            hat_R[i, :] = hat_r_scaled_min_max
        return hat_R
    
    def _predict(self, user_id, item_id):
        bo_user = user_id in self.hat_R.index
        bo_item = item_id in self.hat_R.columns
        if bo_user and bo_item:
            return self.hat_R.loc[user_id, item_id]
        elif bo_user:
            return self.hat_R.loc[user_id, :].mean()
        elif bo_item:
            return self.hat_R.loc[:, item_id].mean()
        else:
            return self.hat_R.values.mean()
        

if __name__ == '__main__':
    from time import time
    user_ids = np.random.choice(range(2000), size=100000)
    item_ids = np.random.choice(range(2000), size=100000)
    values   = np.random.choice(range(5), size=100000)
    
    rwcf = RandomWalkCF()
    s = time()
    rwcf.fit(user_ids, item_ids, values)    
    print(time() - s)
    s = time()
    rwcf.predict(user_ids, item_ids)
    print(time() - s)
