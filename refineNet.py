import numpy as np
import tensorflow as tf

from utils import *
from inputs import *
from layers import *

with open('config.json', 'r') as f:
    conf = json.load(f)

conf['IS_TRAIN_FROM_SCRATCH'] = 'False'
conf['LEARNING_RATE'] = 3e-6
conf['LOG_DIR'] += 'refineNet/'
conf['CHECKPOINTS_DIR'] += 'refineNet/'

def _conv3d_x2(inputs, kernel_size, in_channels, out_channels, layer_name, activation_func=tf.nn.relu):
    """Implement the Residual Conv Uint, note here put the `relu` after `conv` cause IVDs segmentation does not need pretraining."""
    with tf.name_scope(layer_name):
        z = conv3d(inputs, kernel_size, in_channels, out_channels, 'dense_1')
        z = conv3d(z, kernel_size, out_channels, out_channels, 'dense_2')
        return z + inputs

def _residual_pool(inputs, kernel_size, strides, in_channels, out_channels, layer_name):
    with tf.name_scope(layer_name):
        _ksize = [1, kernel_size, kernel_size, kernel_size, 1]
        _strides = [1, strides, strides, strides, 1]
        pool = tf.nn.max_pool3d(inputs, _ksize, _strides, padding='SAME', name='pool')
        conv = conv3d(pool, 3, in_channels, out_channels, 'conv3x3', activation_func=tf.identity)
        return conv

def _chained_res_pool(inputs, kernel_size, strides, in_channels, out_channels, layer_name):
    with tf.name_scope(layer_name):
        init = tf.nn.relu(inputs, name='relu')
        pool_conv_1 = _residual_pool(init, kernel_size, strides, in_channels, out_channels, 'pool_conv_1')
        pool_conv_2 = _residual_pool(pool_conv_1, kernel_size, strides, out_channels, out_channels, 'pool_conv_2')
        return init + pool_conv_1 + pool_conv_2

def train():
    # 0 -- train, 1 -- test, 2 -- val
    MODE = tf.placeholder(tf.uint8, shape=[], name='mode')

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[1, None, None, None, 1], name='x_input')
        tf.summary.image('images', x[:, tf.shape(x)[1] // 2])  # requires 4-d tensor, here takes the middle slice across x-axis

    conv_1 = conv3d_x3(x, kernel_size=3, in_channels=1, out_channels=16, layer_name='conv_1')
    mini_1 = conv3d_as_pool(conv_1, 3, 16, 64, 'scaled_x2')  # 1 / 2 size of original images

    conv_2 = conv3d_x3(mini_1, kernel_size=3, in_channels=64, out_channels=64, layer_name='conv_2')
    mini_2 = conv3d_as_pool(conv_2, 3, 64, 128, 'scaled_x4')

    conv_3 = conv3d_x3(mini_2, kernel_size=3, in_channels=128, out_channels=128, layer_name='conv_3')
    mini_3 = conv3d_as_pool(conv_3, 3, 128, 256, 'scaled_x8')

    conv_4 = conv3d_x3(mini_3, kernel_size=3, in_channels=256, out_channels=256, layer_name='conv_4')
    mini_4 = conv3d_as_pool(conv_4, 3, 256, 512, 'scaled_x16')

    conv_5 = conv3d_x3(mini_4, kernel_size=3, in_channels=512, out_channels=512, layer_name='conv_5')

    with tf.name_scope('refineNet_5'):
        with tf.name_scope('rcu'):
            # Residual Conv Unit
            rcu = _conv3d_x2(conv_5, 3, 512, 512, 'rcu_1')
            rcu = _conv3d_x2(rcu, 3, 512, 512, 'rcu_2')
        with tf.name_scope('fusion'):
            # The bottom only takes one inputs so no real fusion.
            fusion = tf.identity(rcu, name='fusion')
        with tf.name_scope('pool'):
            # Chained Residual Pooling.
            res_pool = _chained_res_pool(fusion, kernel_size=5, strides=1, in_channels=512, out_channels=512, layer_name='pool')
        with tf.name_scope('output'):
            out = _conv3d_x2(res_pool, 3, 512, 512, 'output')

    with tf.name_scope('refineNet_4'):
        with tf.name_scope('rcu'):
            # Generates feature maps of the same feature dimension, the smallest one.
            with tf.name_scope('from_4'):
                rcu_4 = _conv3d_x2(conv_4, 3, 256, 256, 'rcu4_1')
                rcu_4 = _conv3d_x2(rcu_4, 3, 256, 256, 'rcu4_2')
            with tf.name_scope('from_5'):
                rcu_5 = _conv3d_x2(out, 3, 512, 512, 'rcu5_1')
                rcu_5 = _conv3d_x2(rcu_5, 3, 512, 512, 'rcu5_2')
        with tf.name_scope('fusion'):
            fusion_4 = tf.identity(rcu_4, name='fusion4')
            fusion_5 = deconv3d_as_up(rcu_5, 3, 512, 256, 'fusion5')
            fusion = fusion_4 + crop(fusion_5, fusion_4)
        with tf.name_scope('pool'):
            res_pool = _chained_res_pool(fusion, kernel_size=5, strides=1, in_channels=256, out_channels=256, layer_name='pool')
        with tf.name_scope('output'):
            out = conv_4 + _conv3d_x2(res_pool, 3, 256, 256, 'output')

    with tf.name_scope('refineNet_3'):
        with tf.name_scope('rcu'):
            with tf.name_scope('from_3'):
                rcu_3 = _conv3d_x2(conv_3, 3, 128, 128, 'rcu3_1')
                rcu_3 = _conv3d_x2(rcu_3, 3, 128, 128, 'rcu3_2')
            with tf.name_scope('from_4'):
                rcu_4 = _conv3d_x2(out, 3, 256, 256, 'rcu4_1')
                rcu_4 = _conv3d_x2(rcu_4, 3, 256, 256, 'rcu4_2')
        with tf.name_scope('fusion'):
            fusion_3 = tf.identity(rcu_3, name='fusion3')
            fusion_4 = deconv3d_as_up(rcu_4, 3, 256, 128, 'fusion4')
            fusion = fusion_3 + crop(fusion_4, fusion_3)
        with tf.name_scope('pool'):
            res_pool = _chained_res_pool(fusion, kernel_size=5, strides=1, in_channels=128, out_channels=128, layer_name='pool')
        with tf.name_scope('output'):
            out = conv_3 + _conv3d_x2(res_pool, 3, 128, 128, 'output')

    with tf.name_scope('refineNet_2'):
        with tf.name_scope('rcu'):
            with tf.name_scope('from_2'):
                rcu_2 = _conv3d_x2(conv_2, 3, 64, 64, 'rcu2_1')
                rcu_2 = _conv3d_x2(rcu_2, 3, 64, 64, 'rcu2_2')
            with tf.name_scope('from_3'):
                rcu_3 = _conv3d_x2(out, 3, 128, 128, 'rcu3_1')
                rcu_3 = _conv3d_x2(rcu_3, 3, 128, 128, 'rc13_2')
        with tf.name_scope('fusion'):
            fusion_2 = tf.identity(rcu_2, name='fusion2')
            fusion_3 = deconv3d_as_up(rcu_3, 2, 128, 64, 'fusion3')
            fusion = fusion_2 + crop(fusion_3, fusion_2)
        with tf.name_scope('pool'):
            res_pool = _chained_res_pool(fusion, kernel_size=5, strides=1, in_channels=64, out_channels=64, layer_name='pool')
        with tf.name_scope('output'):
            out = conv_2 + _conv3d_x2(res_pool, 3, 64, 64, 'output')

    with tf.name_scope('refineNet_1'):
        with tf.name_scope('rcu'):
            with tf.name_scope('from_1'):
                rcu_1 = _conv3d_x2(conv_1, 3, 16, 16, 'rcu1_1')
                rcu_1 = _conv3d_x2(rcu_1, 3, 16, 16, 'rcu1_2')
            with tf.name_scope('from_2'):
                rcu_2 = _conv3d_x2(out, 3, 64, 64, 'rcu2_1')
                rcu_2 = _conv3d_x2(rcu_2, 3, 64, 64, 'rcu2_2')
        with tf.name_scope('fusion'):
            fusion_1 = tf.identity(rcu_1, name='fusion1')
            fusion_2 = deconv3d_as_up(rcu_2, 2, 64, 16, 'fusion2')
            fusion = fusion_1 + crop(fusion_2, fusion_1)
        with tf.name_scope('pool'):
            res_pool = _chained_res_pool(fusion, kernel_size=5, strides=1, in_channels=16, out_channels=16, layer_name='pool')
        with tf.name_scope('output'):
            out = conv_1 + _conv3d_x2(res_pool, 3, 16, 16, 'output')

    with tf.name_scope('output'):
        y_conv = conv3d(out, 1, 16, 2, 'y_conv', activation_func=tf.identity)
        tf.summary.image('y_conv_0', y_conv[:, tf.shape(y_conv)[1] // 2, ..., 0, None])
        tf.summary.image('y_conv_1', y_conv[:, tf.shape(y_conv)[1] // 2, ..., 1, None])

    y_ = tf.placeholder(tf.float32, shape=[1, None, None, None, 1], name='y_input')
    tf.summary.image('labels', y_[:, tf.shape(y_conv)[1] // 2, ..., 0, None])  # None to keep dims

    with tf.name_scope('loss'):
        y_softmax = tf.nn.softmax(y_conv)
        loss = dice_loss(y_, y_softmax)
        dice_pct = evaluation_metrics(y_[..., 0], tf.argmax(y_conv, 4))
        tf.summary.scalar('dice', dice_pct)
        tf.summary.scalar('total_loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(conf['LEARNING_RATE']).minimize(loss)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            y_pred_img = tf.to_float(tf.argmax(y_conv, 4))
            correct_predictions = tf.equal(y_pred_img, y_[..., 0])
            tf.summary.image('predicted_images', y_pred_img[:, tf.shape(y_conv)[1]//2, ..., None])
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries.
    merged = tf.summary.merge_all()

    def feed_dict(mode=0):
        if mode == 0: data_range = 10
        if mode == 1: data_range = (10, 11, 12)
        if mode == 2: data_range = (13, 14)

        data = load_data(nii_index=np.random.choice(data_range))
        # Directly assign will make image and label to 4-d tensor.
        # Dice loss do not need weights map
        image, labele, _ = np.split(np.array(data), 3)

        return {x: image, y_: labele, MODE: mode}

    with tf.Session() as sess:
        saver = tf.train.Saver()  # Add ops to save and restore all the variables.
        start_i = 0
        end_i = int(conf['NUM_EPOCHS'] * conf['TRAIN_SIZE'] * conf['AUGMENT_SIZE'])

        if eval(conf['IS_TRAIN_FROM_SCRATCH']):
            print('Start initializing...')
            tf.global_variables_initializer().run()
        else:
            ckpt_path = tf.train.latest_checkpoint(conf['CHECKPOINTS_DIR'])
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
                    saver.save(sess, conf['CHECKPOINTS_DIR'] + 'refineNet', global_step=i)
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
        for i in range(int(conf['VAL_SIZE'] * conf['AUGMENT_SIZE'])):
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
