import json
import tensorflow as tf

with open('config.json', 'r') as f:
    conf = json.load(f)

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
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

def conv3d(inputs, kernel_size, in_channels, out_channels, layer_name, activation_func=tf.nn.relu, strides=[1, 1, 1, 1, 1]):
    """Compute the z = f(W * x + b)"""
    depth = tf.shape(inputs)[1]
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W_shape = [kernel_size, kernel_size, kernel_size, in_channels, out_channels]
            stddev = tf.to_float(tf.sqrt(2 / (kernel_size**3 * in_channels)))   # float32 is required by `truncated_normal`
            W = weight_variable(W_shape, stddev)
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = bias_variable([out_channels])
            variable_summaries(b)
        with tf.name_scope('activation'):
            z = activation_func(tf.nn.conv3d(inputs, W, strides, padding='SAME') + b)
        tf.summary.image('activation', z[:, depth//(strides[1] * 2), ..., 0, None])
        return z

def conv3d_as_pool(inputs, kernel_size, in_channels, out_channels, layer_name, activation_func=tf.nn.relu, strides=[1, 2, 2, 2, 1]):
    return conv3d(inputs, kernel_size, in_channels, out_channels, layer_name, activation_func, strides)

def conv3d_x3(inputs, kernel_size, in_channels, out_channels, layer_name):
    """Three serial convs with a residual connection."""
    with tf.name_scope(layer_name):
        # Adjust channels for final add.
        z = conv3d(inputs, kernel_size, in_channels, out_channels, 'dense_1')
        z_out = conv3d(z, kernel_size, out_channels, out_channels, 'dense_2')
        z_out = conv3d(z_out, kernel_size, out_channels, out_channels, 'dense_3')
        return z + z_out

def crop(lhs, rhs):
    """Assume lhs is bigger."""
    lhs_shape = tf.shape(lhs)
    rhs_shape = tf.shape(rhs)
    offsets = [0, (lhs_shape[1] - rhs_shape[1]) // 2, (lhs_shape[2] - rhs_shape[2]) // 2, (lhs_shape[3] - rhs_shape[3]) // 2, 0]
    size = [-1, rhs_shape[1], rhs_shape[2], rhs_shape[3], -1]
    cropped_lhs = tf.slice(lhs, offsets, size)
    # cropped_lhs.set_shape(rhs.get_shape().as_list())
    return cropped_lhs

def deconv3d_as_up(inputs, kernel_size, in_channels, out_channels, layer_name, activation_func=tf.nn.relu, strides=[1, 2, 2, 2, 1]):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # Notice the order of inputs and outputs, which is required by `conv3d_transpose`
            W_shape = [kernel_size, kernel_size, kernel_size, out_channels, in_channels]
            stddev = tf.to_float(tf.sqrt(2 / (kernel_size**3 * in_channels)))
            W = weight_variable(W_shape, stddev)
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = bias_variable([out_channels])
            variable_summaries(b)
        with tf.name_scope('deconv'):
            basic_shape = tf.stack([1, tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3], out_channels])
            deconv_outshape = basic_shape * strides
            up = activation_func(tf.nn.conv3d_transpose(inputs, W, output_shape=deconv_outshape, strides=strides) + b)
        tf.summary.image('activation', up[:, tf.shape(inputs)[1] // 2, ..., 0, None])
    return up

def deconv3d_x3(lhs, rhs, kernel_size, in_channels, out_channels, layer_name):
     with tf.name_scope(layer_name):
        rhs_up = deconv3d_as_up(rhs, kernel_size, in_channels, out_channels, layer_name='up')
        rhs_add = crop(rhs_up, lhs) + lhs
        conv = conv3d_x3(rhs_add, kernel_size, out_channels, out_channels, layer_name='conv')
        return crop(conv, rhs_add)

def deconv3d_concat(lhs, rhs, kernel_size, in_channels, out_channels, layer_name):
    with tf.name_scope(layer_name):
        rhs_up = deconv3d_as_up(rhs, kernel_size, in_channels, out_channels, layer_name='up')
        rhs_concat = tf.concat([crop(rhs_up, lhs), lhs], axis=4)
        conv = conv3d_x3(rhs_concat, kernel_size, out_channels * 2, out_channels, layer_name='conv')
        return crop(conv, rhs_concat)
