# -*- coding: utf-8 -*-
"""
Created on 2017-04-06 17:56:42

@author: kimmy
"""
import json
import numpy as np
import tensorflow as tf
from inputs import *
from utils import *

with open('config.json', 'r') as f:
    conf = json.load(f)

conf['LOG_DIR'] += 'unet_3d/'

def train():
    # 0 -- train, 1 -- test, 2 -- val
    MODE = tf.placeholder(tf.uint8, shape=[], name='mode')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[conf['BATCH_SIZE'], conf['DEPTH'], conf['HEIGHT'], conf['WIDTH'], conf['CHANNEL']], name='x_input')
        tf.summary.image('images', x[:, conf['DEPTH']//2])  # requires 4-d tensor, here takes the middle slice across x-axis

    cc1 = combined_conv(x, kernel_size=3, out_channels=16, layer_name='combined_conv_1')
    with tf.name_scope('max_pool_1'):
        pool = max_pool_optional_norm(cc1, 2)

    cc2 = combined_conv(pool, kernel_size=3, out_channels=64, layer_name='combined_conv_2')
    with tf.name_scope('max_pool_2'):
        pool = max_pool_optional_norm(cc2, 2)

    cc3 = combined_conv(pool, kernel_size=3, out_channels=128, layer_name='combined_conv_3')
    with tf.name_scope('max_pool_3'):
        pool = max_pool_optional_norm(cc3, 2)

    cc4 = combined_conv(pool, kernel_size=3, out_channels=256, layer_name='combined_conv_4')
    with tf.name_scope('max_pool_3'):
        pool = max_pool_optional_norm(cc4, 2)

    bottom = combined_conv(pool, kernel_size=3, out_channels=512, layer_name='combined_conv_bottom')

    dc4 = combined_deconv(bottom, cc4, kernel_size=3, out_channels=256, layer_name='combined_deconv_4')
    dc3 = combined_deconv(dc4, cc3, kernel_size=3, out_channels=128, layer_name='combined_deconv_3')
    dc2 = combined_deconv(dc3, cc2, kernel_size=3, out_channels=64, layer_name='combined_deconv_2')
    dc1 = combined_deconv(dc2, cc1, kernel_size=3, out_channels=16, layer_name='combined_deconv_1')

    with tf.name_scope('output'):
        kernel_size = 1
        with tf.name_scope('weights'):
            W_shape = [kernel_size, kernel_size, kernel_size, 16, 2]
            stddev = np.sqrt(2 / (kernel_size**3 * 16))
            W = weight_variable(W_shape, stddev)
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = bias_variable([2])
            variable_summaries(b)
        with tf.name_scope('y_conv'):
            y_conv = tf.nn.elu(conv3d(dc1, W) + b)

        # Acquire output shape to crop the imputs for elementwise computation
        _, y_conv_depth, y_conv_height, y_conv_width, _ = y_conv.get_shape().as_list()
        tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 0, None])
        tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 1, None])

    y_ = tf.placeholder(tf.float32, shape=[conf['BATCH_SIZE'], y_conv_depth, y_conv_height, y_conv_width, 1], name='y_input')
    tf.summary.image('labels', y_[:, y_conv_depth // 2, ..., 0, None])  # None to keep dims

    with tf.name_scope('loss'):
        y_softmax = tf.nn.softmax(y_conv)
        dice_loss = dice_coef(y_, y_softmax)
        dice_pct = evaluation_metrics(y_[..., 0], tf.argmax(y_conv, 4))
        tf.summary.scalar('dice', dice_pct)
        tf.summary.scalar('total_loss', dice_loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(conf['LEARNING_RATE']).minimize(dice_loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            y_pred_img = tf.to_float(tf.argmax(y_conv, 4))
            correct_predictions = tf.equal(y_pred_img, y_[..., 0])
        tf.summary.image('predicted_images', y_pred_img[:, y_conv_depth//2, ..., None])
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries
    merged = tf.summary.merge_all()

    def feed_dict(mode=0):
        if mode == 0: data_range = 10
        if mode == 1: data_range = (10, 11, 12)
        if mode == 2: data_range = (13, 14)

        batch_index = np.random.choice(data_range, size=conf['BATCH_SIZE'])
        bulk = [load_data(nii_index=i) for i in batch_index]
        # Convert tuples to np.array
        batch_images, batch_labels = [np.array(item) for item in list(zip(*bulk))]

        # Crop the labels
        # offset_depth = (DEPTH - y_conv_depth) // 2 = (36 -32) // 2 = 2
        # offset_rows = (HEIGHT - y_conv_height) // 2 = (300 - 288) // 2 = 6
        # offset_cols = (WIDTH - y_conv_width) // 2 = (300 - 288) // 2 = 6
        batch_labels = batch_labels[:, 2: (-2) , 6: (-6), 6: (-6), :]
        return {x: batch_images, y_: batch_labels, MODE: mode}

    with tf.Session() as sess:
        saver = tf.train.Saver()  # Add ops to save and restore all the variables.
        start_i = 0
        end_i = int(conf['NUM_EPOCHS'] * conf['TRAIN_SIZE'] * conf['AUGMENT_SIZE'] // conf['BATCH_SIZE'])

        if eval(conf['IS_TRAIN_FROM_SCRATCH']):
            print('Start initializing...')
            tf.global_variables_initializer().run()
        else:
            ckpt_path = tf.train.latest_checkpoint('./checkpoints/unet_3d/')
            saver.restore(sess, ckpt_path)
            start_i = int(ckpt_path.split('-')[-1])
            print('Resume training from %s, do not need initiazing...' % (start_i))

        train_writer = tf.summary.FileWriter(conf['LOG_DIR'] + 'train', sess.graph)
        test_writer = tf.summary.FileWriter(conf['LOG_DIR'] + 'test')

        for i in range(start_i, end_i):
            if i % 10 == 0:
                summary, acc, dice_overlap = sess.run([merged, accuracy, dice_pct], feed_dict=feed_dict(mode=1))
                test_writer.add_summary(summary, i)
                print('Testing accuracy at step %s: %s\tdice overlap percentage: %s' % (i, acc, dice_overlap))
                if i % 200 == 0:  # Save the variables to disk
                    saver.save(sess, './checkpoints/unet_3d/unet_3d', global_step=i)
            else:                   # Record execution stats
                if (i + 1) % 100 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                  feed_dict=feed_dict(),
                                  options=run_options,
                                  run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % (i + 1))
                    train_writer.add_summary(summary, i + 1)
                else:       # Record a summary
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict())

        total_acc = []
        for i in range(int(conf['VAL_SIZE'] * conf['AUGMENT_SIZE'] // conf['BATCH_SIZE'])):
            acc = sess.run([accuracy], feed_dict(mode=2))
            total_acc.append(acc)

        print('Final accuracy is %7.3f' % np.mean(total_acc))

        train_writer.close()
        test_writer.close()

def main(_):
    if eval(conf['IS_TRAIN_FROM_SCRATCH']):
        if tf.gfile.Exists(conf['LOG_DIR']):
            tf.gfile.DeleteRecursively(conf['LOG_DIR'])
        tf.gfile.MakeDirs(conf['LOG_DIR'])
    train()

if __name__ == '__main__':
    tf.app.run(main=main)
