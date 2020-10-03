#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf
from time import time

from src.module.knowledge_graph_attention_network.Model.utility.helper import *
from src.module.knowledge_graph_attention_network.Model.utility.batch_test import *

from src.module.knowledge_graph_attention_network.Model.BPRMF import BPRMF as BPRMF_
from src.module.knowledge_graph_attention_network.Model.CKE import CKE as CKE_
from src.module.knowledge_graph_attention_network.Model.CFKG import CFKG as CFKG_
from src.module.knowledge_graph_attention_network.Model.NFM import NFM as NFM_
from src.module.knowledge_graph_attention_network.Model.KGAT import KGAT as KGAT_

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = dict()
config['n_users'] = data_generator.n_users
config['n_items'] = data_generator.n_items
config['n_relations'] = data_generator.n_relations
config['n_entities']  = data_generator.n_entities

if args.model_type in ['kgat', 'cfkg']:
    "Load the laplacian matrix."
    config['A_in'] = sum(data_generator.lap_list)

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['all_v_list'] = data_generator.all_v_list

tf.set_random_seed(2019)
np.random.seed(2019)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args = parse_args()

class KGAT:
    def __init__(self):
        """
        self = KGAT()
        data_config = config
        pretrain_data = None
        
        sess = tf.Session()
        
        self.fit()
        """        
        # setup # これはバグの元、argsは他のモジュールでも読み込まれているため、ここで変更すると不整合が生じる
        """
        args.model_type = 'kgat'
        args.alg_type = 'kgat'
        args.dataset = 'ml' #?
        args.regs = '[1e-5,1e-5]'
        args.layer_size = '[64,32,16]'        
        args.embed_size = 64
        args.lr = 0.0001
        args.batch_size = 1024
        args.node_dropout = '[0.1]'
        args.mess_dropout = '[0.1,0.1,0.1]'
        args.use_att = True
        args.use_kge = True
        """
        self.model = KGAT_(data_config=config, pretrain_data=None, args=args)
        # すべての変数を保存対象とする
        saver = tf.train.Saver()
        
    def fit(self):
        
        t0 = time()

        ##############################
        # Save the model parameters.
        if args.save_flag == 1:
            if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
                weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, self.model.model_type,
                                                                 str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
    
            elif args.model_type in ['ncf', 'nfm', 'kgat']:
                layer = '-'.join([str(l) for l in eval(args.layer_size)])
                weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                    args.weights_path, args.dataset, self.model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
    
            ensureDir(weights_save_path)
            save_saver = tf.train.Saver(max_to_keep=1)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        
        
        ##############################
        # do not pretrain
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')        
        
        ##############################
        # Get the final peformance w.r.t. different sparsity levels.
        pass 
    
        ##############################
        # Train.
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
    
        for epoch in range(args.epoch):
            t1 = time()
            loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
    
            """
            *********************************************************
            Alternative Training for KGAT:
            ... phase 1: to train the recommender.
            """
            for idx in range(n_batch):
                btime = time()
                # positive, negaative のデータセットを取得する（negativeはランダムサンプリング)
                batch_data = data_generator.generate_train_batch()
                # 学習時に入力(feed)するdictionary形式に整える
                feed_dict = data_generator.generate_train_feed_dict(self.model, batch_data)
    
                _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = self.model.train(sess, feed_dict=feed_dict)
    
                loss += batch_loss
                base_loss += batch_base_loss
                kge_loss += batch_kge_loss
                reg_loss += batch_reg_loss
    
            if np.isnan(loss) == True:
                print('ERROR: loss@phase1 is nan.')
                sys.exit()
    
            """
            *********************************************************
            Alternative Training for KGAT:
            ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
            """
            if args.model_type in ['kgat']:
    
                n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1
    
                if args.use_kge is True:
                    # using KGE method (knowledge graph embedding).
                    for idx in range(n_A_batch):
                        btime = time()
    
                        A_batch_data = data_generator.generate_train_A_batch()
                        feed_dict = data_generator.generate_train_A_feed_dict(self.model, A_batch_data)
    
                        _, batch_loss, batch_kge_loss, batch_reg_loss = self.model.train_A(sess, feed_dict=feed_dict)
    
                        loss += batch_loss
                        kge_loss += batch_kge_loss
                        reg_loss += batch_reg_loss
    
                if args.use_att is True:
                    # updating attentive laplacian matrix.
                    self.model.update_attentive_A(sess)
    
            if np.isnan(loss) == True:
                print('ERROR: loss@phase2 is nan.')
                sys.exit()
    
            show_step = 10
            if (epoch + 1) % show_step != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                        epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
                    print(perf_str)
                continue
            
            """
            *********************************************************
            Test.
            """
            t2 = time()
            users_to_test = list(data_generator.test_user_dict.keys())    
            ret = test(sess, self.model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
            
    
            """
            *********************************************************
            Performance logging.
            """
            t3 = time()
    
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
    
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, base_loss, kge_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)
    
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=10)
    
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break
    
            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)
    
        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)
    
        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)
    
        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in pres[idx]]),
                      '\t'.join(['%.5f' % r for r in hit[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)
    
        save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, self.model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'a')
    
        f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s, use_kge=%s, pretrain=%d\n\t%s\n'
                % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, args.use_kge, args.pretrain, final_perf))
        f.close()
            
            
    def get_score_matrix(self, sess, user_ids, item_ids):
        """
        user_ids x tem_ids の表示優先スコアマトリックスを返却する。
        
        user_ids = [0,]
        item_ids = [16,999,0]
        sess = tf.Session()
        """
        sess.run(tf.global_variables_initializer())
        feed_dict = data_generator.generate_test_feed_dict(self.model, user_ids, item_ids)
        return self.model.eval(sess, feed_dict)       
    
    def predict(self, user_ids, item_ids):
        """
        user_ids = [0, 0, 1, 1]
        item_ids = [99,16,99, 4275]
        """
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        uq_user_ids = np.unique(user_ids)
        uq_item_ids = np.unique(item_ids)
        sess = tf.Session()
        score_matrix = self.get_score_matrix(sess, uq_user_ids, uq_item_ids)
        result = []
        for u,i in zip(user_ids, item_ids):
            where_u = np.where(uq_user_ids==u)[0]
            where_i = np.where(uq_item_ids==i)[0]
            result.append(score_matrix[where_u, where_i][0])
        return np.array(result)


if __name__ == 'how to user it':
    pass
    """ テストプログラムの作成を断念中    
    train_txt = ['0 0 1 2',
                 '1 0',
                 '2 1',]
    test_txt = ['3 0',]
    kg_final_txt = ['0 0 0',
                    '0 0 2']
    train_txt = '\n'.join(train_txt)
    test_txt = '\n'.join(test_txt)
    kg_final_txt = '\n'.join(kg_final_txt)
    
    from src.module.knowledge_graph_attention_network.Model.utility.helper import *
    from src.module.knowledge_graph_attention_network.Model.utility.batch_test import *
    import os
    
    args = parse_args()
    dir_data = os.path.join(args.data_path, args.dataset)
    with open(os.path.join(dir_data, 'train.txt'), 'w') as f:
        f.write(train_txt)
    with open(os.path.join(dir_data, 'test.txt'), 'w') as f:
        f.write(test_txt)
    with open(os.path.join(dir_data, 'kg_final.txt'), 'w') as f:
        f.write(kg_final_txt)
    
    from src.module.DeepLearningRec import KGAT
    """

