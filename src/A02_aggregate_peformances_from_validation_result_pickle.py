#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import re
import numpy as np
import pandas as pd
from collections import Counter


def main():
    DIR = 'pickle'
    
    pickle_file_list = [os.path.join(DIR, p) for p in os.listdir(DIR) if re.match('^.*.pickle', p)]
    
    frame = []
    for pickle_file in pickle_file_list:
        frame.append(parse(pickle_file))
    
    result = pd.concat(frame, ignore_index=True)
    result.to_csv('output/A02_validation_result.tsv', index=False, sep='\t')


def parse(pickle_file):
    model_name, hold = 'nothing', 'nothing'
    for file_name_part in pickle_file.split('__'):
        if re.match('model_name=', file_name_part):
            model_name = file_name_part.split('=')[1]
        if re.match('hold=', file_name_part):
            hold = file_name_part.split('=')[1].replace('.pickle', '')

    vr = pickle.load(open(pickle_file, 'br'))
    random_seed = vr['VALIDATION_PARAMETERs']['random_seed']
    topN = vr['VALIDATION_PARAMETERs']['topN']
    remove_still_interaction_from_test = vr['VALIDATION_PARAMETERs']['remove_still_interaction_from_test']
        
    # --- categorize items and users for Validation ---    
    cnt_dict = Counter(vr['train_item_ids'])
    head_cnt = np.percentile(list(cnt_dict.values()), q=90)
    head_item_ids = [k for k,cnt in cnt_dict.items() if cnt >= head_cnt]
    tail_item_ids = [k for k,cnt in cnt_dict.items() if cnt <  head_cnt]
    no_trained_item_ids  = np.unique(vr['test_item_ids'][~np.in1d(vr['test_item_ids'], np.unique(vr['train_item_ids']))])

    cnt_dict = Counter(vr['train_user_ids'])
    head_cnt = np.percentile(list(cnt_dict.values()), q=90)
    head_user_ids = [k for k,cnt in cnt_dict.items() if cnt >= head_cnt]
    tail_user_ids = [k for k,cnt in cnt_dict.items() if cnt <  head_cnt]
    no_trained_user_ids  = np.unique(vr['test_user_ids'][~np.in1d(vr['test_user_ids'], np.unique(vr['train_user_ids']))])


    # --- set the numbers of id
    def count_unique_in(array, items):
        """
        arrayの中に含まれるitemsの数とユニーク数を返却する。
        array = np.array([1,3,3,5,5,5,7])
        items = np.array([3,7,13])
        
        return: (3, 2)
        """
        array_in = array[np.in1d(array, items)]
        return array_in.size, len(set(array_in))
    
    n_sample_head_item_ids_in_train, n_unique_head_item_ids_in_train = count_unique_in(vr['train_item_ids'], head_item_ids)
    n_sample_tail_item_ids_in_train, n_unique_tail_item_ids_in_train = count_unique_in(vr['train_item_ids'], tail_item_ids)
    n_sample_no_trained_item_ids_in_train, n_unique_no_trained_item_ids_in_train = count_unique_in(vr['train_item_ids'], no_trained_item_ids)
    
    n_sample_head_item_ids_in_test, n_unique_head_item_ids_in_test = count_unique_in(vr['test_item_ids'], head_item_ids)
    n_sample_tail_item_ids_in_test, n_unique_tail_item_ids_in_test = count_unique_in(vr['test_item_ids'], tail_item_ids)
    n_sample_no_trained_item_ids_in_test, n_unique_no_trained_item_ids_in_test = count_unique_in(vr['test_item_ids'], no_trained_item_ids)

    n_sample_head_user_ids_in_train, n_unique_head_user_ids_in_train = count_unique_in(vr['train_user_ids'], head_user_ids)
    n_sample_tail_user_ids_in_train, n_unique_tail_user_ids_in_train = count_unique_in(vr['train_user_ids'], tail_user_ids)
    n_sample_no_trained_user_ids_in_train, n_unique_no_trained_user_ids_in_train = count_unique_in(vr['train_user_ids'], no_trained_user_ids)
    
    n_sample_head_user_ids_in_test, n_unique_head_user_ids_in_test = count_unique_in(vr['test_user_ids'], head_user_ids)
    n_sample_tail_user_ids_in_test, n_unique_tail_user_ids_in_test = count_unique_in(vr['test_user_ids'], tail_user_ids)
    n_sample_no_trained_user_ids_in_test, n_unique_no_trained_user_ids_in_test = count_unique_in(vr['test_user_ids'], no_trained_user_ids)


    #-------------------------#

    # --- Validation on all items and user ---
    vr['test_values'] = np.array(vr['test_values'])
    vr['predicted_values'] = np.array(vr['predicted_values'])
    vr['test_good_hit_result'] = {k:np.array(v) for k,v in vr['test_good_hit_result'].items()}
    
    test_values = vr['test_values']
    predicted_values = vr['predicted_values']
    test_good_hit_result = vr['test_good_hit_result']
    
    metrics_all_ids = get_metrix(test_values, predicted_values, test_good_hit_result)
        
    def common_process(bo_index, bo_hit_index):
        test_values, predicted_values = vr['test_values'][bo_index], vr['predicted_values'][bo_index]
        test_good_hit_result = {k:v[bo_hit_index] for k,v in vr['test_good_hit_result'].items()}
        return get_metrix(test_values, predicted_values, test_good_hit_result)
    
    # --- Validation on head_item_ids
    bo_index = np.in1d(vr['test_item_ids'], head_item_ids)
    bo_hit_index = np.in1d(vr['test_good_item_ids'], head_item_ids)
    metrics_head_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on tail_item_ids
    bo_index = np.in1d(vr['test_item_ids'], tail_item_ids)
    bo_hit_index = np.in1d(vr['test_good_item_ids'], tail_item_ids)
    metrics_tail_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on no_trained_item_ids
    bo_index = np.in1d(vr['test_item_ids'], no_trained_item_ids)
    bo_hit_index = np.in1d(vr['test_good_item_ids'], no_trained_item_ids)
    metrics_no_trained_item_ids = common_process(bo_index, bo_hit_index)

    #-------------------------#

    # --- Validation on head_user_ids
    bo_index = np.in1d(vr['test_user_ids'], head_user_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], head_user_ids)
    metrics_head_user_ids = common_process(bo_index, bo_hit_index)
    
    # --- Validation on tail_user_ids
    bo_index = np.in1d(vr['test_user_ids'], tail_user_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], tail_user_ids)
    metrics_tail_user_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on no_trained_user_ids
    bo_index = np.in1d(vr['test_user_ids'], no_trained_user_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], no_trained_user_ids)
    metrics_no_trained_user_ids = common_process(bo_index, bo_hit_index)

    #-------------------------#

    # --- Validation on head_user_ids & head_item_ids
    _user_ids, _item_ids = head_user_ids, head_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_head_user_ids__head_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on head_user_ids & tail_item_ids
    _user_ids, _item_ids = head_user_ids, tail_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_head_user_ids__tail_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on head_user_ids & no_trained_item_ids
    _user_ids, _item_ids = head_user_ids, no_trained_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_head_user_ids__no_trained_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on tail_user_ids & head_item_ids
    _user_ids, _item_ids = tail_user_ids, head_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_tail_user_ids__head_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on tail_user_ids & tail_item_ids
    _user_ids, _item_ids = tail_user_ids, tail_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_tail_user_ids__tail_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on tail_user_ids & no_trained_item_ids
    _user_ids, _item_ids = tail_user_ids, no_trained_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_tail_user_ids__no_trained_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on no_trained_user_ids & head_item_ids
    _user_ids, _item_ids = no_trained_user_ids, head_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_no_trained_user_ids__head_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on no_trained_user_ids & tail_item_ids
    _user_ids, _item_ids = no_trained_user_ids, tail_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_no_trained_user_ids__tail_item_ids = common_process(bo_index, bo_hit_index)

    # --- Validation on no_trained_user_ids & no_trained_item_ids
    _user_ids, _item_ids = no_trained_user_ids, no_trained_item_ids
    bo_index = np.in1d(vr['test_user_ids'], _user_ids) & np.in1d(vr['test_item_ids'], _item_ids)
    bo_hit_index = np.in1d(vr['test_good_user_ids'], _user_ids) & np.in1d(vr['test_good_item_ids'], _item_ids)
    metrics_no_trained_user_ids__no_trained_item_ids = common_process(bo_index, bo_hit_index)


    # --- RETURN ---
    def wrapper(dict_metrix):
        return convert_df(
                        dict_metrix
                        ,model_name=model_name
                        ,random_seed=random_seed
                        ,topN=topN
                        ,remove_still_interaction_from_test=remove_still_interaction_from_test
                        ,hold=hold
                        ,n_sample_head_item_ids_in_train=n_sample_head_item_ids_in_train
                        ,n_unique_head_item_ids_in_train=n_unique_head_item_ids_in_train
                        ,n_sample_tail_item_ids_in_train=n_sample_tail_item_ids_in_train
                        ,n_unique_tail_item_ids_in_train=n_unique_tail_item_ids_in_train
                        ,n_sample_no_trained_item_ids_in_train=n_sample_no_trained_item_ids_in_train
                        ,n_unique_no_trained_item_ids_in_train=n_unique_no_trained_item_ids_in_train
                        ,n_sample_head_item_ids_in_test=n_sample_head_item_ids_in_test
                        ,n_unique_head_item_ids_in_test=n_unique_head_item_ids_in_test
                        ,n_sample_tail_item_ids_in_test=n_sample_tail_item_ids_in_test
                        ,n_unique_tail_item_ids_in_test=n_unique_tail_item_ids_in_test
                        ,n_sample_no_trained_item_ids_in_test=n_sample_no_trained_item_ids_in_test
                        ,n_unique_no_trained_item_ids_in_test=n_unique_no_trained_item_ids_in_test
                        ,n_sample_head_user_ids_in_train=n_sample_head_user_ids_in_train
                        ,n_unique_head_user_ids_in_train=n_unique_head_user_ids_in_train
                        ,n_sample_tail_user_ids_in_train=n_sample_tail_user_ids_in_train
                        ,n_unique_tail_user_ids_in_train=n_unique_tail_user_ids_in_train
                        ,n_sample_no_trained_user_ids_in_train=n_sample_no_trained_user_ids_in_train
                        ,n_unique_no_trained_user_ids_in_train=n_unique_no_trained_user_ids_in_train
                        ,n_sample_head_user_ids_in_test=n_sample_head_user_ids_in_test
                        ,n_unique_head_user_ids_in_test=n_unique_head_user_ids_in_test
                        ,n_sample_tail_user_ids_in_test=n_sample_tail_user_ids_in_test
                        ,n_unique_tail_user_ids_in_test=n_unique_tail_user_ids_in_test
                        ,n_sample_no_trained_user_ids_in_test=n_sample_no_trained_user_ids_in_test
                        ,n_unique_no_trained_user_ids_in_test=n_unique_no_trained_user_ids_in_test
                          )
    
    return_metrics = [
            'metrics_all_ids', 
            'metrics_head_item_ids', 'metrics_tail_item_ids', 'metrics_no_trained_item_ids', 
            'metrics_head_user_ids', 'metrics_tail_user_ids', 'metrics_no_trained_user_ids',
            'metrics_head_user_ids__head_item_ids', 'metrics_head_user_ids__tail_item_ids', 'metrics_head_user_ids__no_trained_item_ids',
            'metrics_tail_user_ids__head_item_ids', 'metrics_tail_user_ids__tail_item_ids', 'metrics_tail_user_ids__no_trained_item_ids',
            'metrics_no_trained_user_ids__head_item_ids', 'metrics_no_trained_user_ids__tail_item_ids', 'metrics_no_trained_user_ids__no_trained_item_ids',            
            ]
    frame = []
    for metrics_ in return_metrics:
        df_ = wrapper(eval(metrics_))
        df_['test_ids'] = metrics_
        frame.append(df_)
    
    return pd.concat(frame, axis=0, ignore_index=True)


def get_metrix(test_values, predicted_values, test_good_hit_result):
    errores = np.array(predicted_values) - np.array(test_values)
    MAE = np.mean(np.abs(errores))
    RMSE = np.sqrt(np.mean(np.square(errores)))

    return_dict = dict(MAE=MAE, RMSE=RMSE)
    
    for topN, hit_result in test_good_hit_result.items():
        if np.array(hit_result).size != 0:
            recall    = sum(np.array(hit_result)) / np.array(hit_result).size
            precision = recall / topN
        else:
            recall, precision = np.nan, np.nan
        return_dict['topN={}_recall'.format('%02d'%topN)] = recall
        return_dict['topN={}_precision'.format('%02d'%topN)] = precision
        
    return return_dict
    

def convert_df(dict_metrix, **keyargs):
    """
    Arguments:
        dict_metrix [dictionary]:
            the returned values from get_metrix()
            ex)
                {'MAE': 0.7656306513941742,
                 'RMSE': 0.9728788696895713,
                 'precision': 0.0058041648205582625,
                 'recall': 0.05804164820558263}
        **keyargs:
            the {column: value} you want to put in df
    """
    df_ = pd.DataFrame([dict_metrix])
    for col, val in keyargs.items():
        try:
            df_.loc[0, col] = val
        except:
            df_.loc[0, col] = str(val)            
    return df_


if __name__ == '__main__':
    main()

