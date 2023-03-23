#!/bin/sh -x
set -e
set -u


echo "Start @ `date +'%F %T'`"

gpu_device=4

models_dir='./model'
profiler_dir='./profiling'
train_data_dir='/data1/annehywang/DataSetProcess/data/Census-Income/Train/Task1'
test_data_dir='/data1/annehywang/DataSetProcess/data/Census-Income/Test/Task1'

rm -r $models_dir

## train
time CUDA_VISIBLE_DEVICES=$gpu_device ./train.py $train_data_dir $models_dir $profiler_dir training 
## eval
time CUDA_VISIBLE_DEVICES=$gpu_device ./train.py $test_data_dir $models_dir $profiler_dir eval

echo "Done @ `date +'%F %T'`"
