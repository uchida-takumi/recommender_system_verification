#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
import copy

def get_CF_varidation_arrays(
                            train_user_ids, train_item_ids, train_values
                           ,test_user_ids, test_item_ids, test_values
                           ,model
                           ,good_score_threshold=5
                           ,n_random_selected_item_ids=1000
                           ,remove_still_interaction_from_test=False
                           ,topN=[5,10], random_seed=None, need_fit=True
                           ,**fit_args):
    """
     This is a inplementation of (*1) which define a metric for 
    Top-N Recommendation Tasks.    
    
    (*1) Paolo 2010: Performance of Recommender Algorithms on Top-N Recommendation Tasks
    
    INPUT:
    -------------------------------
        user_ids, item_ids, values [array like]: 
            dataset for train.
        test_user_ids, test_item_ids, test_values [array like]: 
            dataset for test.
        model : 
            CF model which has self.fit(user_ids, item_ids, values) and self.predict(user_ids, item_ids).
        good_score_threshold [numeric]:
            to aggregate recall and precision for Collaborative Filtering.
        n_random_selected_item_ids [int]:
            to aggregate recall and precision for Collaborative Filtering.
        remove_still_interaction_from_test [True/False]:
            This should set False, if you want to run along with (*1).
            But, I think this should be set True...
        topN [list of int]:
            to aggregate recall and precision for Collaborative Filtering.
        random_seed [int]:
            random seed values.
        

            
    RETURN:
    -------------------------------
        the dictionary involve arrayes which are needed to aggrigate MAE, RMSE, recall, precision. 

    
    EXAMPLE:
    -------------------------------
        import numpy as np
        import re
        
        # set data
        user_ids = np.random.choice(range(100), size=500)
        item_ids = np.random.choice(range(20),  size=500)
        values   = np.random.choice(range(6),   size=500)

        test_user_ids = np.random.choice(range(100), size=500)
        test_item_ids = np.random.choice(range(20),  size=500)
        test_values   = np.random.choice(range(6),   size=500)

        # set simple model    
        class random_model:
            def fit(self, user_ids, item_ids, values):
                pass
            def predict(self, user_ids, item_ids):
                return [np.random.rand() for u in user_ids]
    
        model = random_model()        

        result_dict = get_CF_varidation_arrays(
                                            user_ids, item_ids, values
                                           ,test_user_ids, test_item_ids, test_values
                                           ,model
                                           ,good_score_threshold=5
                                           ,n_random_selected_item_ids=1000
                                           ,remove_still_interaction_from_test=False
                                           ,topN=[5,10,15]
                                           ,random_seed=1
                                           )
        
        errores = result_dict['predicted_values'] - result_dict['test_values']
        MAE  = np.mean(np.abs(errores))
        RMSE = np.sqrt(np.mean(np.square(errores)))
        recall_over_all    = sum(result_dict['test_good_hit_result'][0]) / len(result_dict['test_good_hit_result'][0])
        precision_over_all = recall_over_all / topN[0]                
    """
    # --- initial process
    np.random.seed(seed=random_seed)
    
    # ---- Training ----
    if need_fit:
        model.fit(train_user_ids, train_item_ids, train_values, **fit_args)
        
    # ---- Testing ----
    ## get recall and precision
    good_indexes = [val >= good_score_threshold for val in test_values]
    
    test_good_user_ids = test_user_ids[good_indexes]
    test_good_item_ids = test_item_ids[good_indexes]
    test_good_values   = test_values[good_indexes]
    test_good_result = {tN:[] for tN in topN}
    
    ## 別々に予測すると時間がかかるため、高速化のために一気に予測する    
    bulk_iters, bulk_user_ids, bulk_item_ids = [], [], []
    n_iter = 0
    print('bulk で予測を実行')
    for user_id, item_id, value in zip(test_good_user_ids, test_good_item_ids, test_good_values):
        # user_id, item_id, value = test_good_user_ids[0], test_good_item_ids[0], test_good_values[0]
        if remove_still_interaction_from_test:
            # set user_ids which the user_id still has not interact with.
            the_user_ids_indexes = (train_user_ids == user_id)
            not_still_interact_item_ids = set(train_item_ids[~the_user_ids_indexes])
            item_id_set = not_still_interact_item_ids
        else:
            item_id_set = set(train_item_ids)

        # add test_item_ids in item_id_set
        item_id_set.update(test_item_ids)
    
        # remove the item_id from item_itest_ratingd_set
        item_id_set = item_id_set - {item_id}
        # get random selected item_ids as that the user_id will not be interested in. 
        n_random_selected_item_ids = min(n_random_selected_item_ids, len(item_id_set))
        random_selected_item_ids = np.random.choice(list(item_id_set), size=n_random_selected_item_ids, replace=False)
        _item_ids = np.concatenate([random_selected_item_ids, [item_id]])
        
        bulk_user_ids += copy.deepcopy([user_id]*len(_item_ids))
        bulk_item_ids += copy.deepcopy(list(_item_ids))
        bulk_iters += copy.deepcopy([n_iter]*len(_item_ids))
        
        n_iter += 1
        
    bulk_user_ids = np.array(bulk_user_ids)
    bulk_item_ids = np.array(bulk_item_ids)
    bulk_iters = np.array(bulk_iters)
    bulk_predicted = np.array(model.predict(bulk_user_ids, bulk_item_ids, **fit_args))        

    print('予測結果のhitを計算する')
    n_max_iter = bulk_iters.max() 
    n_iter = 0     
    set_iters = set(bulk_iters)       
    for _iter in range(n_max_iter):
        if (_iter+1) % 100 == 0:
            print(f'get_CF_Varidation_array: {_iter+1}/{n_max_iter}')
        
        indexes = bulk_iters == _iter

        _predicted = bulk_predicted[indexes]
        _item_ids = bulk_item_ids[indexes] # 前の処理で、正解のitem_idは最後に格納されている。
        answer_item_id = _item_ids[-1]

        item_ids__predicted = np.concatenate([_item_ids.reshape(-1,1), _predicted.reshape(-1,1)], axis=1)
        
        # sort by random shuffle
        np.random.shuffle(item_ids__predicted)
        
        # sort by predicted rating.
        item_ids__predicted = item_ids__predicted[np.argsort(item_ids__predicted[:,1])[::-1], :]
        
        # set the result in test_good_result.
        for tN in test_good_result:
            item_ids_in_topN = item_ids__predicted[:tN, 0]                
            if answer_item_id in item_ids_in_topN:
                hit = 1
            else:
                hit = 0    
            test_good_result[tN].append(hit)
            
    # ---- return metric ----
    return_dict = {
            "train_user_ids": train_user_ids,
            "train_item_ids": train_item_ids,
            "train_values": train_values,
            "test_user_ids": test_user_ids,
            "test_item_ids": test_item_ids,
            "test_values": test_values,
            "predicted_values": model.predict(test_user_ids, test_item_ids, **fit_args),
            #"fitted_model": model,
            "test_good_user_ids": test_good_user_ids,
            "test_good_item_ids": test_good_item_ids,
            "test_good_values": test_good_values,
            "test_good_hit_result": test_good_result,   
            "VALIDATION_PARAMETERs": {
                "n_random_selected_item_ids": n_random_selected_item_ids,
                "topN": topN,
                "remove_still_interaction_from_test": remove_still_interaction_from_test,
                "random_seed": random_seed,
                "good_score_threshold": good_score_threshold,
                }
            }
        
    return return_dict



if __name__ == 'How to use':
    import numpy as np
    import re
    # INPUTs
    train_user_ids = np.random.choice(range(100), size=500)
    train_item_ids = np.random.choice(range(20),  size=500)
    train_values   = np.random.choice(range(6),   size=500)

    test_user_ids = np.random.choice(range(100), size=500)
    test_item_ids = np.random.choice(range(20),  size=500)
    test_values   = np.random.choice(range(6),   size=500)

    # set simple model    
    class random_model:
        def fit(self, user_ids, item_ids, values):
            pass
        def predict(self, user_ids, item_ids):
            return [np.random.rand() for u in user_ids]

    model = random_model()        
    
    result_dict = get_CF_varidation_arrays(
                                        train_user_ids, train_item_ids, train_values
                                       ,test_user_ids, test_item_ids, test_values
                                       ,model
                                       ,good_score_threshold=5
                                       ,n_random_selected_item_ids=1000
                                       ,remove_still_interaction_from_test=False
                                       ,topN=[5,10,15]
                                       ,random_seed=1
                                       )
    
    errores = result_dict['predicted_values'] - result_dict['test_values']
    MAE  = np.mean(np.abs(errores))
    RMSE = np.sqrt(np.mean(np.square(errores)))
    recall_over_all = sum(result_dict['test_good_hit_result']) / len(result_dict['test_good_hit_result'])
    precision_over_all = recall_over_all / topN

