# -*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf

from utils import *
from inputs import *
from layers import *

with open('config.json', 'r') as f:
    conf = json.load(f)

conf['CHECKPOINTS_DIR'] += 'vnet_clip_boundary/'

def deploy():

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[1, None, None, None, 1], name='x_input')

    conv_1 = conv3d_x3(x, kernel_size=3, in_channels=1, out_channels=16, layer_name='conv_1')
    pool = conv3d_as_pool(conv_1, kernel_size=3, in_channels=16, out_channels=32, layer_name='pool1')

    conv_2 = conv3d_x3(pool, kernel_size=3, in_channels=32, out_channels=32, layer_name='conv_2')
    pool = conv3d_as_pool(conv_2, kernel_size=3, in_channels=32, out_channels=64, layer_name='pool2')

    conv_3 = conv3d_x3(pool, kernel_size=3, in_channels=64, out_channels=64, layer_name='conv_3')
    pool = conv3d_as_pool(conv_3, kernel_size=3, in_channels=64, out_channels=128, layer_name='pool3')

    conv_4 = conv3d_x3(pool, kernel_size=3, in_channels=128, out_channels=128, layer_name='conv_4')
    pool = conv3d_as_pool(conv_4, kernel_size=3, in_channels=128, out_channels=256, layer_name='pool4')

    bottom = conv3d_x3(pool, kernel_size=3, in_channels=256, out_channels=256, layer_name='bottom')

    deconv_4 = deconv3d_x3(conv_4, bottom,   kernel_size=3, in_channels=256, out_channels=256, layer_name='deconv_4')
    deconv_3 = deconv3d_x3(conv_3, deconv_4, kernel_size=3, in_channels=256, out_channels=128, layer_name='deconv_3')
    deconv_2 = deconv3d_x3(conv_2, deconv_3, kernel_size=3, in_channels=128, out_channels=64, layer_name='deconv_2')
    deconv_1 = deconv3d_x3(conv_1, deconv_2, kernel_size=3, in_channels=64,  out_channels=32, layer_name='deconv_1')

    y_conv = conv3d(deconv_1, kernel_size=1, in_channels=32, out_channels=2, layer_name='output', activation_func=tf.identity)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            y_pred_img = tf.to_float(tf.argmax(y_conv, 4))

    with tf.Session() as sess:
        saver = tf.train.Saver()  # Add ops to save and restore all the

        print('Inference step, do not need initiazing...')
        ckpt_path = tf.train.latest_checkpoint(conf['CHECKPOINTS_DIR'])
        start_i = int(ckpt_path.split('-')[-1])
        print('Restore %d checkpoints' % start_i)
        saver.restore(sess, ckpt_path)

        for i in range(10):
            pred_img = sess.run(y_pred_img, {x : load_inference(nii_index=i)})
            print('Processing %4dth images...' % i)
            np.save('./pred/vnet_' + str(i), pred_img)

        print('Live long and prosper.')

def main(_):
    deploy()

if __name__ == '__main__':
    tf.app.run(main=main)
