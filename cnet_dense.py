# -*- coding: utf-8 -*-
"""
Created on 2017-04-07 13:41:05

@author: kimmy
"""
import json
import numpy as np
import tensorflow as tf

from utils import *
from inputs import *

with open('config.json', 'r') as f:
    conf = json.load(f)

conf['IS_TRAIN_FROM_SCRATCH'] = 'True'
conf['LOG_DIR'] += 'cnet_dense/'
conf['LEARNING_RATE'] = 1e-6


def train():
    # 0 -- train, 1 -- test, 2 -- val
    MODE = tf.placeholder(tf.uint8, shape=[], name='mode')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[conf['BATCH_SIZE'], conf['DEPTH'], conf['HEIGHT'], conf['WIDTH'], conf['CHANNEL']], name='x_input')
        tf.summary.image('images', x[:, conf['DEPTH']//2])  # requires 4-d tensor, here takes the middle slice across x-axis

    with tf.name_scope('top1'):
        conv1_1 = dense3d(x, 3, 16, layer_name='conv1')
        conv1_2 = dense3d(conv1_1, 3, 16, layer_name='conv2')
        concat1_1 = tf.concat([conv1_1, conv1_2], axis=4, name='concat1')
        conv1_3 = dense3d(concat1_1, 3, 16, layer_name='conv3')
        concat1_2 = tf.concat([concat1_1, conv1_3], axis=4, name='concat2')

    with tf.name_scope('top2'):
        pool2_1 = conv3d_as_pool(conv1_1, 64, layer_name='pool1')
        pool2_2 = conv3d_as_pool(concat1_1, 64, layer_name='pool2')
        pool2_3 = conv3d_as_pool(concat1_2, 64, layer_name='pool3')

        conv2_1 = dense3d(pool2_1, 3, 64, layer_name='conv1')
        add2_1 = conv2_1 + pool2_2
        concat2_1 = tf.concat([pool2_1, add2_1], axis=4, name='concat1')

        conv2_2 = dense3d(concat2_1, 3, 64, layer_name='conv2')
        add2_2 = conv2_2 + pool2_3
        concat2_2 = tf.concat([concat2_1, add2_2], axis=4, name='concat2')

    with tf.name_scope('top3'):
        pool3_1 = conv3d_as_pool(pool2_1, 128, layer_name='pool1')
        pool3_2 = conv3d_as_pool(concat2_1, 128, layer_name='pool2')
        pool3_3 = conv3d_as_pool(concat2_2, 128, layer_name='pool3')

        conv3_1 = dense3d(pool3_1, 3, 128, layer_name='conv1')
        add3_1 = conv3_1 + pool3_2
        concat3_1 = tf.concat([pool3_1, add3_1], axis=4, name='concat1')

        conv3_2 = dense3d(concat3_1, 3, 128, layer_name='conv2')
        add3_2 = conv3_2 + pool3_3
        concat3_2 = tf.concat([concat3_1, add3_2], axis=4, name='concat2')

    with tf.name_scope('top4'):
        pool4_1 = conv3d_as_pool(pool3_1, 256, layer_name='pool1')
        pool4_2 = conv3d_as_pool(concat3_1, 256, layer_name='pool2')
        pool4_3 = conv3d_as_pool(concat3_2, 256, layer_name='pool3')

        conv4_1 = dense3d(pool4_1, 3, 256, layer_name='conv1')
        add4_1 = conv4_1 + pool4_2
        concat4_1 = tf.concat([pool4_1, add4_1], axis=4, name='concat1')

        conv4_2 = dense3d(concat4_1, 3, 256, layer_name='conv2')
        add4_2 = conv4_2 + pool4_3
        concat4_2 = tf.concat([concat4_1, add4_2], axis=4, name='concat2')

    with tf.name_scope('vertex'):
        pool_v_1 = conv3d_as_pool(pool4_1, 512, layer_name='pool1')
        pool_v_2 = conv3d_as_pool(concat4_1, 512, layer_name='pool2')
        pool_v_3 = conv3d_as_pool(concat4_2, 512, layer_name='pool3')

        conv_v_1 = dense3d(pool_v_1, 3, 512, layer_name='conv1')
        add_v_1 = conv_v_1 + pool_v_2
        concat_v_1 = tf.concat([pool_v_1, add_v_1], axis=4, name='concat1')

        conv_v_2 = dense3d(concat_v_1, 3, 512, layer_name='conv2')
        add_v_2 = conv_v_2 + pool_v_3

    with tf.name_scope('bottom4'):
        up_pool4_1 = crop(deconv_as_up(add_v_2, 3, 256, layer_name='up_pool_1'), pool4_3)
        up_pool4_2 = crop(deconv_as_up(add_v_1, 3, 256, layer_name='up_pool_2'), pool4_2)
        up_pool4_3 = crop(deconv_as_up(pool_v_1, 3, 256, layer_name='up_pool_3'), pool4_1)

        up_conv4_1 = dense3d(up_pool4_1, 3, 256, layer_name='conv1')
        up_add4_1 = up_conv4_1 + up_pool4_2
        up_concat4_1 = tf.concat([up_pool4_1, up_add4_1], axis=4, name='concat1')

        up_conv4_2 = dense3d(up_concat4_1, 3, 256, layer_name='conv2')
        up_add4_2 = up_conv4_2 + up_pool4_3
        up_concat4_2 = tf.concat([up_concat4_1, up_add4_2], axis=4, name='concat2')

    with tf.name_scope('bottom3'):
        up_pool3_1 = crop(deconv_as_up(up_conv4_1, 3, 128, layer_name='up_pool_1'), pool3_3)
        up_pool3_2 = crop(deconv_as_up(up_add4_1, 3, 128, layer_name='up_pool_2'), pool3_2)
        up_pool3_3 = crop(deconv_as_up(up_add4_2, 3, 128, layer_name='up_pool_3'), pool3_1)

        up_conv3_1 = dense3d(up_pool3_1, 3, 128, layer_name='conv1')
        up_add3_1 = up_conv3_1 + up_pool3_2
        up_concat3_1 = tf.concat([up_pool3_1, up_add3_1], axis=4, name='concat1')

        up_conv3_2 = dense3d(up_concat3_1, 3, 128, layer_name='conv2')
        up_add3_2 = up_conv3_2 + up_pool3_3
        up_concat3_2 = tf.concat([up_concat3_1, up_add3_2], axis=4, name='concat2')

    with tf.name_scope('bottom2'):
        up_pool2_1 = crop(deconv_as_up(up_conv3_1, 3, 64, layer_name='up_pool_1'), pool2_3)
        up_pool2_2 = crop(deconv_as_up(up_add3_1, 3, 64, layer_name='up_pool_2'), pool2_2)
        up_pool2_3 = crop(deconv_as_up(up_add3_2, 3, 64, layer_name='up_pool_3'), pool2_1)

        up_conv2_1 = dense3d(up_pool2_1, 3, 64, layer_name='conv1')
        up_add2_1 = up_conv2_1 + up_pool2_2
        up_concat2_1 = tf.concat([up_pool2_1, up_add2_1], axis=4, name='concat1')

        up_conv2_2 = dense3d(up_concat2_1, 3, 64, layer_name='conv2')
        up_add2_2 = up_conv2_2 + up_pool2_3
        up_concat2_2 = tf.concat([up_concat2_1, up_add2_2], axis=4, name='concat2')

    with tf.name_scope('bottom1'):
        up_pool1_1 = crop(deconv_as_up(up_conv2_1, 3, 16, layer_name='up_pool_1'), conv1_3)
        up_pool1_2 = crop(deconv_as_up(up_add2_1, 3, 16, layer_name='up_pool_2'), conv1_2)
        up_pool1_3 = crop(deconv_as_up(up_add2_2, 3, 16, layer_name='up_pool_3'), conv1_1)

        up_conv1_1 = dense3d(up_pool1_1, 3, 16, layer_name='conv1')
        up_add1_1 = up_conv1_1 + up_pool1_2
        up_concat1_1 = tf.concat([up_pool1_1, up_add1_1], axis=4, name='concat1')

        up_conv1_2 = dense3d(up_concat1_1, 3, 16, layer_name='conv2')
        up_add1_2 = up_conv1_2 + up_pool1_3
        up_concat1_2 = tf.concat([up_concat1_1, up_add1_2], axis=4, name='concat2')

    y_conv = dense3d(up_concat1_2, 1, 2, 'output')

    _, y_conv_depth, y_conv_height, y_conv_width, _ = y_conv.get_shape().as_list()
    tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 0, None])
    tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 1, None])

    y_ = tf.placeholder(tf.float32, shape=[conf['BATCH_SIZE'], y_conv_depth, y_conv_height, y_conv_width, 1], name='y_input')
    tf.summary.image('labels', y_[:, y_conv_depth // 2, ..., 0, None])  # None to keep dims

    print('*'*60, 'top1')
    print('-'*30, 'conv1_1/2', conv1_1.get_shape().as_list())
    print('-'*30, 'concat1_1', concat1_1.get_shape().as_list())
    print('-'*30, 'conv1_3', conv1_3.get_shape().as_list())
    print('-'*30, 'concat1_2', concat1_2.get_shape().as_list())

    print('*'*60, 'top2')
    print('-'*30, 'pool2_1/2/3', pool2_1.get_shape().as_list())
    print('-'*30, 'conv2_1, add2_1', conv2_1.get_shape().as_list())
    print('-'*30, 'concat2_1', concat2_1.get_shape().as_list())
    print('-'*30, 'conv2_2, add2_2', conv2_2.get_shape().as_list())
    print('-'*30, 'concat2_2', concat2_2.get_shape().as_list())

    print('*'*60, 'top3')
    print('-'*30, 'pool3_1/2/3', pool3_1.get_shape().as_list())
    print('-'*30, 'conv3_1, add3_1', conv3_1.get_shape().as_list())
    print('-'*30, 'concat3_1', concat3_1.get_shape().as_list())
    print('-'*30, 'conv3_2, add3_2', conv3_2.get_shape().as_list())
    print('-'*30, 'concat3_2', concat3_2.get_shape().as_list())

    print('*'*60, 'top4')
    print('-'*30, 'pool4_1/2/3', pool4_1.get_shape().as_list())
    print('-'*30, 'conv4_1, add4_1', conv4_1.get_shape().as_list())
    print('-'*30, 'concat4_1', concat4_1.get_shape().as_list())
    print('-'*30, 'conv4_2, add4_2', conv4_2.get_shape().as_list())
    print('-'*30, 'concat4_2', concat4_2.get_shape().as_list())

    print('*'*60, 'vertex')
    print('-'*30, 'pool_v_1/2/3', pool_v_1.get_shape().as_list())
    print('-'*30, 'conv_v_1, add_v_1', conv_v_1.get_shape().as_list())
    print('-'*30, 'concat_v_1', concat_v_1.get_shape().as_list())
    print('-'*30, 'conv_v_2, add_v_2', conv_v_2.get_shape().as_list())

    print('*'*60, 'bottom4')
    print('-'*30, 'up_pool4_1/2/3', up_pool4_1.get_shape().as_list())
    print('-'*30, 'up_conv4_1, up_add4_1', up_conv4_1.get_shape().as_list())
    print('-'*30, 'up_concat4_1', up_concat4_1.get_shape().as_list())
    print('-'*30, 'up_conv4_2, up_add4_2', up_conv4_2.get_shape().as_list())
    print('-'*30, 'up_concat4_2', up_concat4_2.get_shape().as_list())

    print('*'*60, 'bottom3')
    print('-'*30, 'up_pool3_1/2/3', up_pool3_1.get_shape().as_list())
    print('-'*30, 'up_conv3_1, up_add3_1', up_conv3_1.get_shape().as_list())
    print('-'*30, 'up_concat3_1', up_concat3_1.get_shape().as_list())
    print('-'*30, 'up_conv3_2, up_add3_2', up_conv3_2.get_shape().as_list())
    print('-'*30, 'up_concat3_2', up_concat3_2.get_shape().as_list())

    print('*'*60, 'bottom2')
    print('-'*30, 'up_pool2_1/2/3', up_pool2_1.get_shape().as_list())
    print('-'*30, 'up_conv2_1, up_add2_1', up_conv3_1.get_shape().as_list())
    print('-'*30, 'up_concat2_1', up_concat2_1.get_shape().as_list())
    print('-'*30, 'up_conv2_2, up_add2_2', up_conv3_2.get_shape().as_list())
    print('-'*30, 'up_concat2_2', up_concat2_2.get_shape().as_list())

    print('*'*60, 'bottom1')
    print('-'*30, 'up_pool1_1/2/3', up_pool1_1.get_shape().as_list())
    print('-'*30, 'up_conv1_1, up_add1_1', up_conv1_1.get_shape().as_list())
    print('-'*30, 'up_concat1_1', up_concat1_1.get_shape().as_list())
    print('-'*30, 'up_conv1_2, up_add1_2', up_conv1_2.get_shape().as_list())
    print('-'*30, 'up_concat1_2', up_concat1_2.get_shape().as_list())


    print('*'*60, 'output')
    print('-'*30, 'y_conv', y_conv.get_shape().as_list())


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

        return {x: batch_images, y_: batch_labels, MODE: mode}

    with tf.Session() as sess:
        saver = tf.train.Saver()  # Add ops to save and restore all the variables.
        start_i = 0
        end_i = int(conf['NUM_EPOCHS'] * conf['TRAIN_SIZE'] * conf['AUGMENT_SIZE'] // conf['BATCH_SIZE'])

        if eval(conf['IS_TRAIN_FROM_SCRATCH']):
            print('Start initializing...')
            tf.global_variables_initializer().run()
        else:
            ckpt_path = tf.train.latest_checkpoint('./checkpoints/cnet_dense/')
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
                    saver.save(sess, './checkpoints/vnet/vnet', global_step=i)
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
