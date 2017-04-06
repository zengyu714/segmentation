# -*- coding: utf-8 -*-
"""
Created on 2017-04-06 17:56:42

@author: kimmy
"""
import numpy as np
import tensorflow as tf
from inputs import *

TRAIN_SIZE = 10
TEST_SIZE = 3
VAL_SIZE = 2

DEPTH = 36
HEIGHT = 300  # rows
WIDTH = 300   # cols
CHANNEL = 1

BATCH_SIZE = 2
AUGMENT_SIZE = 10000
NUM_EPOCHS = 100

# FLAGS
EPISILON = 1e-7
LOG_DIR = './logs/'
USE_BATCH_NORM = False

IS_TRAIN_FROM_SCRATCH = False

def train():
    # 0 -- train, 1 -- test, 2 -- val
    MODE = tf.placeholder(tf.uint8, shape=[], name='mode')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DEPTH, HEIGHT, WIDTH, CHANNEL], name='x_input')
        tf.summary.image('images', x[:, DEPTH//2])  # requires 4-d tensor, here takes the middle slice across x-axis

    def weight_variable(shape, stddev=0.1):
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor, for TensorBoard visualization. """

        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def max_pool_optional_norm(x, n, to_norm=USE_BATCH_NORM):
        pool = tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='VALID')
        if MODE == 0 and to_norm:
            pool = tf.map_fn(lambda p: tf.nn.local_response_normalization(p, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75), pool)
        return pool

    def conv3d(x, W):
        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def combined_conv(inputs, kernel_size, out_channels, layer_name, activation_func=tf.nn.relu):
        _, depth, height, width, in_channels = inputs.get_shape().as_list()
        with tf.name_scope(layer_name):
            with tf.name_scope('sub_conv1'):
                with tf.name_scope('weights'):
                    W_shape = [kernel_size, kernel_size, kernel_size, in_channels, out_channels]
                    stddev = np.sqrt(2 / (kernel_size**3 * in_channels))
                    W = weight_variable(W_shape, stddev)
                    variable_summaries(W)
                with tf.name_scope('biases'):
                    b = bias_variable([out_channels])
                    variable_summaries(b)
                with tf.name_scope('activation'):
                    z_1 = activation_func(conv3d(inputs, W) + b)
                tf.summary.image('activation', z_1[:, depth//2, ..., 1, None])
            with tf.name_scope('sub_conv2'):
                with tf.name_scope('weights'):
                    W_shape = [kernel_size, kernel_size, kernel_size, out_channels, out_channels]
                    stddev = np.sqrt(2 / (kernel_size**3 * out_channels))
                    W = weight_variable(W_shape, stddev)
                    variable_summaries(W)
                with tf.name_scope('biases'):
                    b = bias_variable([out_channels])
                    variable_summaries(b)
                with tf.name_scope('activation'):
                    z_2 = activation_func(conv3d(z_1, W) + b)
                tf.summary.image('activation', z_2[:, depth//2, ..., 1, None])
            return z_2

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

    def deconv3d(x, W, deconv_outshape, upsample_factor):
        return tf.nn.conv3d_transpose(x, W, deconv_outshape,
                    strides=[1, upsample_factor, upsample_factor, upsample_factor, 1], padding='SAME')

    def crop_and_concat(lhs, rhs):
        # Convert 5-d tensor to 4-d tensor.
        lhs_shape = tf.shape(lhs)
        rhs_shape = tf.shape(rhs)
        offsets = [0, (lhs_shape[1] - rhs_shape[1]) // 2, (lhs_shape[2] - rhs_shape[2]) // 2, (lhs_shape[3] - rhs_shape[3]) // 2, 0]
        size = [-1, rhs_shape[1], rhs_shape[2], rhs_shape[3], -1]
        cropped_lhs = tf.slice(lhs, offsets, size)
        cropped_lhs.set_shape(rhs.get_shape().as_list())
        return tf.concat([cropped_lhs, rhs], axis=4)

    def combined_deconv(inputs, concat_inputs, kernel_size, out_channels, layer_name, activation_func=tf.nn.relu):
        batch_size, depth, height, width, input_channels = inputs.get_shape().as_list()
        with tf.name_scope(layer_name):
            with tf.name_scope('upsample'):
                with tf.name_scope('weights'):
                    # Notice the order of inputs and outputs, which is required by `conv3d_transpose`
                    W_shape = [kernel_size, kernel_size, kernel_size, out_channels, input_channels]
                    stddev = np.sqrt(2 / (kernel_size**3 * input_channels))
                    W = weight_variable(W_shape, stddev)
                    variable_summaries(W)
                with tf.name_scope('biases'):
                    b = bias_variable([out_channels])
                    variable_summaries(b)
                with tf.name_scope('deconv'):
                    deconv_outshape = [batch_size, depth*2, height*2, width*2, out_channels]
                    up = activation_func(deconv3d(inputs, W, deconv_outshape, upsample_factor=2) + b)
                tf.summary.image('activation', up[:, depth // 2, ..., 1, None])
                with tf.name_scope('crop_and_concat'):
                    glue = crop_and_concat(concat_inputs, up)
            return combined_conv(glue, kernel_size, out_channels, layer_name, activation_func)

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
            y_conv = tf.nn.relu(conv3d(dc1, W) + b)

        # Acquire output shape to crop the imputs for elementwise computation
        _, y_conv_depth, y_conv_height, y_conv_width, _ = y_conv.get_shape().as_list()
        tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 0, None])
        tf.summary.image('y_conv', y_conv[:, y_conv_depth // 2, ..., 1, None])

    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, y_conv_depth, y_conv_height, y_conv_width, 1], name='y_input')
    tf.summary.image('labels', y_[:, y_conv_depth // 2, ..., 0, None])  # None to keep dims

    def dice_coef(y_true, y_conv):
        """Compute dice among **positive** labels to avoid unbalance.

        Argument:
            y_true: [batch_size, depth, height, width, 1]
            y_conv: [batch_size, depth, height, width, 2]
        """
        y_true = tf.to_float(tf.reshape(y_true[..., 0], [-1]))
        y_conv = tf.to_float(tf.reshape(y_conv[..., 1], [-1]))
        intersection = tf.reduce_sum(y_conv * y_true)
        union = tf.reduce_sum(y_conv * y_conv) + tf.reduce_sum(y_true * y_true)
        dice = 2.0 * intersection / union
        return 1 - tf.clip_by_value(dice, 0, 1.0 - EPISILON)

    def evaluation_metrics(y_true, y_conv):
        y_true = tf.to_float(y_true)
        y_conv = tf.to_float(y_conv)
        intersection = tf.reduce_sum(y_conv * y_true)
        union = tf.reduce_sum(y_conv) + tf.reduce_sum(y_true)
        dice = 2.0 * intersection / union * 100
        return dice

    with tf.name_scope('loss'):
        y_softmax = tf.nn.softmax(y_conv)
        dice_loss = dice_coef(y_, y_softmax)
        dice_pct = evaluation_metrics(y_[..., 0], tf.argmax(y_conv, 4))
        tf.summary.scalar('dice', dice_pct)
        tf.summary.scalar('total_loss', dice_loss)


    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(dice_loss)

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

        batch_index = np.random.choice(data_range, size=BATCH_SIZE)
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
        end_i = int(NUM_EPOCHS * TRAIN_SIZE * AUGMENT_SIZE // BATCH_SIZE)

        if IS_TRAIN_FROM_SCRATCH:
            print('Start initializing...')
            tf.global_variables_initializer().run()
        else:
            print('Resume training, do not need initiazing...')
            ckpt_path = tf.train.latest_checkpoint('./checkpoints/')
            start_i = int(ckpt_path.split('-')[-1])
            saver.restore(sess, ckpt_path)


        train_writer = tf.summary.FileWriter(LOG_DIR + 'train', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + 'test')

        for i in range(start_i, end_i):
            if i % 10 == 0:
                summary, acc, dice_overlap = sess.run([merged, accuracy, dice_pct], feed_dict=feed_dict(mode=1))
                test_writer.add_summary(summary, i)
                print('Testing accuracy at step %s: %s\tdice overlap percentage: %s' % (i, acc, dice_overlap))
                if i % 200 == 0:  # Save the variables to disk
                    saver.save(sess, './checkpoints/v-net', global_step=i)
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
        for i in range(int(VAL_SIZE * AUGMENT_SIZE // BATCH_SIZE)):
            acc = sess.run([accuracy], feed_dict(mode=2))
            total_acc.append(acc)

        print('Final accuracy is %7.3f' % np.mean(total_acc))
        train_writer.close()
        test_writer.close()

def main(_):
    if IS_TRAIN_FROM_SCRATCH:
        if tf.gfile.Exists(LOG_DIR):
            tf.gfile.DeleteRecursively(LOG_DIR)
        tf.gfile.MakeDirs(LOG_DIR)
    train()

if __name__ == '__main__':
    tf.app.run(main=main)
