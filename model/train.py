#!/usr/bin/env python
# -*- coding: utf-8 -*-
# yp @ 2019-03-11 11:01:32

import sys
import os

import tensorflow as tf

import config

from model_atml_v1 import input_fn, model_fn


def get_feature_file(data_dir, full_training_file):
    infile_list = os.listdir(data_dir)
    infile_list = [data_dir + '/' + x for x in infile_list]

    if full_training_file:
        return infile_list, infile_list[-1]

    if len(infile_list) < 2:
        return infile_list, infile_list

    train_file_len = len(infile_list) * 9 / 10
    return infile_list[:train_file_len], infile_list[train_file_len:]

def main(_):
    train_file_dir = sys.argv[1]
    out_model_dir = sys.argv[2]
    profiler_dir = sys.argv[3]

    is_online_training, is_eval, is_pred = False, False, False
    if len(sys.argv) > 4:
        if sys.argv[4] == 'training':
            is_online_training = True
        elif sys.argv[4] == 'eval':
            is_eval = True
        elif sys.argv[4] == 'pred':
            is_pred = True

    train_file_list, test_file_list = get_feature_file(train_file_dir, is_online_training or is_eval or is_pred)

    log_steps = 1000

    gpu_info = {}
    c = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count=gpu_info), log_step_count_steps=log_steps, save_summary_steps=log_steps, keep_checkpoint_max=1)

    if is_online_training:
        DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params=None, config=c)
        DeepFM.train(input_fn=lambda: input_fn(train_file_list, batch_size=config.batch_size))
    elif is_eval:
        DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params=None, config=c)
        DeepFM.evaluate(input_fn=lambda: input_fn(test_file_list, batch_size=config.batch_size))
    elif is_pred:
        DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params=None, config=c)
        for prob in DeepFM.predict(input_fn=lambda: input_fn(train_file_list, num_epochs=1, batch_size=config.batch_size)):
            print '%s %s' % (prob['task1'][0], prob['task2'][0])
    else:
        DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=out_model_dir, params=None, config=c)

        # steps = None
        steps = min(3e+7/config.batch_size, 150000)

        hook_list = [tf.train.ProfilerHook(save_steps=log_steps, output_dir=sys.argv[3], show_memory=True, show_dataflow=True)]

        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file_list, num_epochs=1, batch_size=config.batch_size), max_steps=steps, hooks=hook_list)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_file_list, num_epochs=1, batch_size=config.batch_size), steps=None, start_delay_secs=120, throttle_secs=180)

        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
