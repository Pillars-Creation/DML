#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import tensorflow as tf
import numpy as np

import config
from model_v2 import input_fn, model_fn

model_dir = './model'
data_dir = './testdata'

predict_file_list = os.listdir(data_dir)
predict_file_list = [data_dir + '/' + x for x in predict_file_list]

log_steps = 1000
gpu_info = {'GPU':6}
c = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count=gpu_info), log_step_count_steps=log_steps, save_summary_steps=log_steps)
DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=None, config=c)

for prob in DeepFM.predict(input_fn=lambda: input_fn(predict_file_list, num_epochs=1, batch_size=config.batch_size)):
    #print 'output1:%s output2:%s' % (prob['output1'],prob['output2'])
    print 'begin'
    out1 = prob['output1']
    new1 = []
    for item in out1:
    	new1.append(round(item,2))
    print new1

    out2 = prob['output2']
    new2 = []
    for item in out2:
    	new2.append(round(item,2))
    print new2
     
   