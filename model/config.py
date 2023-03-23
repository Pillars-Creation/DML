#!/usr/bin/env python
# -*- coding: utf-8 -*-
# yp @ 2019-03-07 15:46:26

import sys


num_threads = 10

batch_size = 50

feature_size_dnn = 5000
embedding_size_dnn = 128

optimizer = 'sgd'

l2_reg = 0.1
learning_rate = 0.01

batch_norm = False
dropout = 1.0

dnn_layers = [200, 80]

experts =
