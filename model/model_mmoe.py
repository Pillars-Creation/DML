#!/usr/bin/env python
# -*- coding: utf-8 -*-
# yp @ 2019-03-07 14:28:06

import sys
import os

import tensorflow as tf

import config


def input_fn(filenames, batch_size, num_epochs=1, perform_shuffle=False):
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        label1, label2, feat = tf.split(columns.values, num_or_size_splits=3)

        label1 = tf.string_to_number(label1, tf.float32)
        label2 = tf.string_to_number(label2, tf.float32)
        
        feat = tf.string_to_number(tf.string_split(feat, ',').values, tf.int64)
        size = tf.reshape(tf.size(feat), [1])

        feat_dnn = feat % config.feature_size_dnn

        return {"size": size, "feat": feat, 'feat_dnn': feat_dnn, 'label2': label2}, label1

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=config.num_threads)  # .prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.padded_batch(batch_size, ({'size':[1], 'feat':[-1],  'feat_dnn':[-1],  'label2':[1]}, [1]))

    dataset = dataset.prefetch(1)  # one batch

    # return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels1 = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels1


def mask_pedding(output, non_padding_length, num_dim=2):
    key_masks = tf.sequence_mask(tf.reshape(non_padding_length, [-1]), tf.shape(output)[1])
    if num_dim == 3:
        key_masks = tf.expand_dims(key_masks, -1)
    return output * tf.cast(key_masks, tf.float32)

def model_fn(features, labels, mode=None, params=None):
 
    l2_reg = config.l2_reg
    learning_rate = config.learning_rate
 
    feat, feat_dnn, size, labels2 = features['feat'], features['feat_dnn'], features['size'], features['label2']

    with tf.variable_scope("deep"):
        DEEP_V = tf.get_variable(name='deep_v', shape=[config.feature_size_dnn, config.embedding_size_dnn], initializer=tf.glorot_normal_initializer())

        dnn_inputs = mask_pedding(tf.nn.embedding_lookup(DEEP_V, feat_dnn), size, 3) # [None, F, embedding_size]
        dnn_inputs = tf.reduce_sum(dnn_inputs, 1) # / tf.cast(size, tf.float32) # mean


        dnn_inputs_1 = tf.contrib.layers.fully_connected(inputs=dnn_inputs, num_outputs=config.embedding_size_dnn, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='feature1')
        dnn_inputs_2 = tf.contrib.layers.fully_connected(inputs=dnn_inputs, num_outputs=config.embedding_size_dnn, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='feature2')
        dnn_inputs_3 = tf.contrib.layers.fully_connected(inputs=dnn_inputs, num_outputs=config.embedding_size_dnn, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='feature3')

        dnn_inputs_1 += dnn_inputs
        dnn_inputs_2 += dnn_inputs
        dnn_inputs_3 += dnn_inputs

        gate1 = tf.contrib.layers.fully_connected(inputs=dnn_inputs, num_outputs=3,activation_fn=tf.nn.softmax, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='gate1')

        gate2 = tf.contrib.layers.fully_connected(inputs=dnn_inputs, num_outputs=3,activation_fn=tf.nn.softmax, \
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='gate2')

        p1,p2,p3 = tf.split(gate1, 3, axis=1)
        p4,p5,p6 = tf.split(gate2, 3, axis=1)

        task1_dnn_inputs_1 = tf.multiply(dnn_inputs_1, p1)
        task1_dnn_inputs_2 = tf.multiply(dnn_inputs_2, p2)
        task1_dnn_inputs_3 = tf.multiply(dnn_inputs_3, p3)

        task2_dnn_inputs_1 = tf.multiply(dnn_inputs_1, p4)
        task2_dnn_inputs_2 = tf.multiply(dnn_inputs_2, p5)
        task2_dnn_inputs_3 = tf.multiply(dnn_inputs_3, p6)

        task1_dnn_inputs = tf.concat([task1_dnn_inputs_1,task1_dnn_inputs_2,task1_dnn_inputs_3],1)
        task2_dnn_inputs = tf.concat([task2_dnn_inputs_1,task2_dnn_inputs_2,task2_dnn_inputs_2],1)

        for i, num_node in enumerate(config.dnn_layers):
            task1_dnn_inputs = tf.contrib.layers.fully_connected(inputs=task1_dnn_inputs, num_outputs=num_node, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp_news%d' % i)
            if config.batch_norm:
                task1_dnn_inputs = batch_norm_layer(task1_dnn_inputs, train_phase=train_phase, scope_bn='bn_news_%d' %i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task1_dnn_inputs = tf.nn.dropout(task1_dnn_inputs, keep_prob=config.dropout)

        y_d_task1 = tf.contrib.layers.fully_connected(inputs=task1_dnn_inputs, num_outputs=1, activation_fn=tf.identity, weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out_news')

        for i, num_node in enumerate(config.dnn_layers):
            task2_dnn_inputs = tf.contrib.layers.fully_connected(inputs=task2_dnn_inputs, num_outputs=num_node, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp_video%d' % i)
            if config.batch_norm:
                task2_dnn_inputs = batch_norm_layer(task2_dnn_inputs, train_phase=train_phase, scope_bn='bn_video_%d' %i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                task2_dnn_inputs = tf.nn.dropout(task2_dnn_inputs, keep_prob=config.dropout)

        y_d_task2 = tf.contrib.layers.fully_connected(inputs=task2_dnn_inputs, num_outputs=1, activation_fn=tf.identity, weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out_video')

    predictions={'task1': tf.sigmoid(y_d_task1),'task2': tf.sigmoid(y_d_task2)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d_task1, labels=labels)) \
            +  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_d_task2, labels=labels2))
   
    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
                       "auc1": tf.metrics.auc(labels, tf.sigmoid(y_d_task1)),
                       "auc2": tf.metrics.auc(labels2, tf.sigmoid(y_d_task2))
                       }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops)


    #------bulid optimizer------
    if config.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif config.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif config.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif config.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:  # 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    return train_op, loss, labels

if __name__ == '__main__':
    train_file_list = sys.argv[1]


