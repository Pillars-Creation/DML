#!/usr/bin/env python
# coding=utf-8
import sys
import time
import shutil

import tensorflow as tf

import utils.data_loader as data_load
import utils.model_op as op
import utils.my_utils as my_utils
from models import mmoe



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run(main=main)
    
    