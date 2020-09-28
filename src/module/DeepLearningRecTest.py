#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:40:23 2020

@author: takumi_uchida
"""

import tensorflow as tf

class tfclass:
    def __init__(self):
        pass
    def train(self, x):
        self.x = tf.constant(x, name='x')
        self.y = tf.placeholder(tf.int32, name='y')
        self.add = self.x + self.y
    def predict(self, sess, feed_dict):
        sess.run(tf.global_variables_initializer())
        return sess.run(self.add, feed_dict)
    

if __name__ == 'how_to_use_this':
    import tensorflow as tf
    from src.module.DeepLearningRecTest import tfclass

    sess = tf.Session()
    self = tfclass()
    self.train(10)
    feed_dict = {self.y:100}
    self.predict(sess, feed_dict)



