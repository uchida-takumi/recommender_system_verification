#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
from .PLSA import plsa
from .util import id_transformer

"""プログラミング用
import copy
import numpy as np
from src.PLSA import plsa
from src.util import id_transformer
"""

class two_way_aspect_model:
    def __init__(self, Z, item_attributes):
        """
        # ARGUMENTs
            Z[int]: 
                the number of latent calss
            item_attributes[dict]:
                key of the dict is item_id, the attribute vector is the values of the dict.
            postive_threshold[int,float]:
                two_way_aspect_model learn only positive touple (user_id, item_id).
                
        # Example
            Z = 2            
        """
        self._two_way_aspect_model = _two_way_aspect_model(Z)
        self.item_attributes = item_attributes

    def fit(self ,user_ids, item_ids, values, postive_threshold=5):
        """
        user_ids [array like object]:
            array of user_id which correspond with item_ids.
            the set of (user_id, item_id) means that the user_id like the item_id.
        item_ids [array like object]:
            array of item_id which correspond with user_ids.
            the set of (user_id, item_id) means that the user_id like the item_id.
        values [array like object]:
            result of the preference. (the score how user like item)
            in this model, only preference which larger than 0 are used. 
        """
        # filter ids in values > 0
        _user_ids, _item_ids = list(), list()
        for user_id, item_id, value in zip(user_ids, item_ids, values):
            if value >= postive_threshold:
                _user_ids.append(user_id)
                _item_ids.append(item_id)
                
        # fit id_transformer
        set_item_ids = set(list(_user_ids) + list(self.item_attributes))
        set_user_ids = set(list(_item_ids))

        item_id_transformer, user_id_transformer = id_transformer(), id_transformer()
        item_id_transformer.fit(set_item_ids)
        user_id_transformer.fit(set_user_ids)

        self.item_id_transformer = item_id_transformer
        self.user_id_transformer = user_id_transformer

        # transform user_ids and item_ids to sequencial id.        
        transformed_item_ids = item_id_transformer.transform(_item_ids)
        transformed_user_ids = user_id_transformer.transform(_user_ids)
        transforemed_item_attributes = \
            {item_id_transformer.transform([_id])[0]:np.array(val) for _id,val in self.item_attributes.items()}
        
        self.transforemed_item_attributes = transforemed_item_attributes
        
        # make matrix Npa, Nma as self._two_way_aspect_model.fit() argument.
        ## Npa(count matrix of person-actor) is equivalent with user-attributes.
        ## Nma(count matrix of movie-actor) is equivalent with item-attributes.
        n_item_ids = len(set_item_ids)
        n_user_ids = len(set_user_ids)
        n_item_attributes = len(list(self.item_attributes.values())[0])
        Nma = np.zeros((n_item_ids, n_item_attributes), dtype=int)
        Npa = np.zeros((n_user_ids, n_item_attributes), dtype=int)
        
        for item_id, val in transforemed_item_attributes.items():
            Nma[item_id] = val
        for user_id, item_id in zip(transformed_user_ids, transformed_item_ids):
            Npa[user_id] += Nma[item_id]
            
        # fit self._two_way_aspect_model
        self._two_way_aspect_model.fit(Npa)
        # predict Ppa (percent of person-movie which is equivalent with user_item)
        self.Ppa = self._two_way_aspect_model.predict(Nma)

    def predict(self, user_ids, item_ids):
        transformed_item_ids = self.item_id_transformer.transform(item_ids, unknown=None)
        transformed_user_ids = self.user_id_transformer.transform(user_ids, unknown=None)

        # return predicted as np.array()
        predicted = []
        for user_id, item_id in zip(transformed_user_ids, transformed_item_ids):
            if (user_id is not None) and (item_id is not None):
                predicted.append(self.Ppa[user_id, item_id])
            elif (user_id is None) and (item_id is not None):
                predicted.append(self.Ppa[:, item_id].mean())
            elif (user_id is not None) and (item_id is None):
                predicted.append(self.Ppa[user_id, :].mean())
            else:
                predicted.append(self.Ppa[:, :].mean())
        
        return np.array(predicted)
            
        

class _two_way_aspect_model:
    def __init__(self, Z):
        """
        # ARGUMENTs
            Z[int]: 
                the number of latent calss
        # Example
            Z = 2            
        """
        self.Z = Z
        
    def fit(self, Npa):    
        """
        # ARGUMENTs
            Npa[np.array]: 
                attributes matrix which dimention is [person, movie-actores]
        # Example
            import numpy as np
            Npa = np.array([
                [2, 0, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 2, 3]
            ])        
        """
        # fit person/actor aspect model with PLSA
        pa_model = plsa(self.Z)
        pa_model.fit(Npa)
        self.pa_model = pa_model
        

    def predict(self, Nma):
        """
        # ARGUMENTs
            Nma[np.array]:
                count matrix which dimention is [movie, movie-actors]
        # Example
            import numpy as np
            Nma = np.array([
                [1, 0, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
            ])
        """
        # the parameters of person/actor aspect model are hold constant.
        Pz = self.pa_model.Pz
        Pa_z = self.pa_model.Py_z
        
        # Fold-In  
        ma_model = plsa(self.Z)
        ma_model.fit(Nma, hold_Pz=Pz, hold_Py_z=Pa_z)
        self.ma_model = ma_model

        Pp_z = self.pa_model.Px_z        
        Pm_z = self.ma_model.Px_z        
        Ppm = (Pz[:,None]*Pp_z).T.dot(Pm_z)
        
        return Ppm
        


if __name__=='__main__':

    user_ids = [5,5,5,6,6,7]
    item_ids = [1,3,6,2,6,3]
    item_attributes = {
            1: [1,0,1,0],
            2: [0,0,1,1],
            3: [0,1,0,0],
            6: [1,0,1,1],
            99: [1,1,1,1],
            }

    twam = two_way_asspect_model(Z=3, item_attributes=item_attributes)
    twam.fit(user_ids, item_ids)
    twam.predict(user_ids, item_ids)
    twam.predict([5], [99]) # 学習中にないitemID
    twam.predict([5], [123]) # attributeにすらないitemID
    twam.predict([123], [3]) 
    twam.predict([123], [123]) 
    # 要、もう少しテストが必要