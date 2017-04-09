# -*- coding: utf-8 -*-
"""
Created on 2017-04-09 17:26:35

@author: kimmy
"""
import json
import numpy as np
import tensorflow as tf

from utils import *
from inputs import *

def load_check(base_path='./data/Check/', nii_index=0):
    """Load nii data to numpy ndarray with **arbitrary** size."""

    image_path = [base_path + p for p in os.listdir(base_path) if not p.endswith('_Label.nii')]
    xs = nib.load(image_path[nii_index]).get_data()

    xs = rescale_intensity(xs, out_range=np.uint8)
    xs = xs / np.max(xs)

    # Normalize images.
    xs = (xs - np.mean(xs)) / np.std(xs)

    return xs[None, ..., None]

def combined_deconv_vnet(dc, cc, kernel_size, in_channels, out_channels, layer_name):
    with tf.name_scope(layer_name):
        up = deconv_as_up(dc, kernel_size, in_channels, out_channels, layer_name='up')
        up = crop(up, cc) + cc
        dc = combined_conv(up, kernel_size, out_channels, out_channels, layer_name='combined_conv')
        return crop(dc, cc) + up

with open('config.json', 'r') as f:
    conf = json.load(f)

conf['CHECKPOINTS_DIR'] += 'vnet/'

def deploy():

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[1, None, None, None, 1], name='x_input')

    cc1 = x + combined_conv(x, kernel_size=3, in_channels=1, out_channels=16, layer_name='combined_conv_1')
    with tf.name_scope('conv_pool_1'):
        pool = conv3d_as_pool(cc1, in_channels=16, out_channels=64, layer_name='pool1')

    cc2 = pool + combined_conv(pool, kernel_size=3, in_channels=64, out_channels=64, layer_name='combined_conv_2')
    with tf.name_scope('conv_pool_2'):
        pool = conv3d_as_pool(cc2, in_channels=64, out_channels=128, layer_name='pool2')

    cc3 = pool + combined_conv(pool, kernel_size=3, in_channels=128, out_channels=128, layer_name='combined_conv_3')
    with tf.name_scope('conv_pool_3'):
        pool = conv3d_as_pool(cc3, in_channels=128, out_channels=256, layer_name='pool3')

    cc4 = pool + combined_conv(pool, kernel_size=3, in_channels=256, out_channels=256, layer_name='combined_conv_4')
    with tf.name_scope('conv_pool_4'):
        pool = conv3d_as_pool(cc4, in_channels=256, out_channels=512, layer_name='pool4')

    bottom = pool + combined_conv(pool, kernel_size=3, in_channels=512, out_channels=512, layer_name='combined_conv_bottom')

    dc4 = combined_deconv_vnet(bottom, cc4, kernel_size=3, in_channels=512, out_channels=256, layer_name='combined_deconv_4')
    dc3 = combined_deconv_vnet(dc4, cc3, kernel_size=3, in_channels=256, out_channels=128, layer_name='combined_deconv_3')
    dc2 = combined_deconv_vnet(dc3, cc2, kernel_size=3, in_channels=128, out_channels=64, layer_name='combined_deconv_2')
    dc1 = combined_deconv_vnet(dc2, cc1, kernel_size=3, in_channels=64, out_channels=16, layer_name='combined_deconv_1')

    y_conv = dense3d(dc1, kernel_size=1, in_channels=16, out_channels=2, layer_name='output')

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
            # pred_img = sess.run(y_pred_img, {x : load_check(base_path='./data/Check/', nii_index=i)})
            pred_img = sess.run(y_pred_img, {x : load_inference(nii_index=i)})
            print('Processing %4dth images...' % i)
            np.save('./pred/check_' + str(i), pred_img)


def main(_):
    deploy()

if __name__ == '__main__':
    tf.app.run(main=main)
