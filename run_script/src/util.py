#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def read_ml20m_data():
    # set file path
    PATH_rating = './data/ml-20m/ratings.csv'
    PATH_movie  = './data/ml-20m/movies.csv'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_movie  = ['MovieID', 'Title', 'Genres']
    
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep=',', names=COLS_rating, skiprows=1)
    user   = pd.DataFrame()
    movie  = pd.read_csv(PATH_movie, sep=',', names=COLS_movie, skiprows=1)
    
    return rating, user, movie
    

def read_ml100k_data():
    # set file path
    PATH_rating = './data/ml-100k/u.data'
    PATH_user   = './data/ml-100k/u.user'
    PATH_movie  = './data/ml-100k/u.item'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_user   = ['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code']
    GenreCOLS = ['Genre%02d'%d for d in range(19)]
    COLS_movie  = ['MovieID', 'Title', 'release-date', 'unknown0', 'unknown1'] + GenreCOLS
    
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep='\t', names=COLS_rating)
    user   = pd.read_csv(PATH_user, sep='|', names=COLS_user)
    movie  = pd.read_csv(PATH_movie, sep='|', names=COLS_movie, encoding='latin-1')
    
    return rating, user, movie


def read_ml1m_data():
    # set file path
    PATH_rating = './data/ml-1m/ratings.dat'
    PATH_user   = './data/ml-1m/users.dat'
    PATH_movie  = './data/ml-1m/movies.dat'
    
    # set columns
    COLS_rating = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    COLS_user   = 'UserID::Gender::Age::Occupation::Zip-code'.split('::')
    COLS_movie = 'MovieID::Title::Genres'.split('::')
    
    # read DataSet
    rating = pd.read_csv(PATH_rating, sep='::', names=COLS_rating)
    user   = pd.read_csv(PATH_user, sep='::', names=COLS_user)
    movie  = pd.read_csv(PATH_movie, sep='::', names=COLS_movie, encoding='latin-1')
    
    return rating, user, movie



class id_transformer:
    def __init__(self):
        """
        transform ids to the index which start from 0.
        """
        pass
    def fit(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        ids_ = sorted(list(set(ids)))
        self.id_convert_dict = {i:index for index,i in enumerate(ids_)}
    
    def transform(self, ids, unknown=None):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.                
        """
        return [self.id_convert_dict.get(i, unknown) for i in ids]

    def fit_transform(self, ids):
        self.fit(ids)
        return self.transform(ids)
        
    def inverse_transform(self, indexes, unknown=None):
        """
        ARGUMETs:
            indexes [array-like object]: 
                array of index which are transformed                
        """
        return [get_key_from_val(self.id_convert_dict, ind) for ind in indexes]
    
    
        
def get_key_from_val(dict_, val, unknown=None):
    """
    dict_ = {'aa':123}
    val = 123
    get_key_from_val(dict_, val)
    > 'aa'    
    """
    list_vals = list(dict_.values())
    if val in list_vals:
        return list(dict_.keys())[list_vals.index(val)]    
    else:
        return unknown




if __name__=='__main__':
    dict_ = {'aa':123}
    val = 123
    get_key_from_val(dict_, val)
    
    ids = [1,1,1,5,5,9]
    it = id_transformer()
    it.fit(ids)
    it.transform([1,5,9])
    it.transform([1,5,9,99])
    it.inverse_transform([0,1,2])
    it.inverse_transform([0,1,2,88])
    
        
    
