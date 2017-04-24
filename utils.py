import os

import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

import nibabel as nib
import tensorflow as tf


def dice_loss(y_true, y_softmax_conv):
    """Compute dice among **positive** labels to avoid unbalance.

    Argument:
        y_true: [batch_size, depth, height, width, 1]
        y_softmax_conv: [batch_size, depth, height, width, 2]
    """
    y_true = tf.to_float(tf.reshape(y_true[..., 0], [-1]))
    y_conv = tf.to_float(tf.reshape(y_softmax_conv[..., 1], [-1]))
    intersection = tf.reduce_sum(y_conv * y_true)
    union = tf.reduce_sum(y_conv * y_conv) + tf.reduce_sum(y_true * y_true)  # y_true is binary
    dice = 2.0 * intersection / union
    return 1 - tf.clip_by_value(dice, 0, 1.0 - 1e-7)

# Cross Entropy
# penalizes at each position the deviation of porbs(logits) from 1

# 1. weighted_softmax_cross_entropy_loss
# ---------------------------------------------------------------------------------------------------------
# Formula : Loss = targets * -log(softmax(logits)) * pos_weight + (1 - targets) * -log(1 - softmax(logits))
# Warning: not STABLE!
# ---------------------------------------------------------------------------------------------------------

# 2. weighted_sigmoid_cross_entropy_loss
# ---------------------------------------------------------------------------------------------------------
# Formula : Loss = targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
# Warning:
#    1) slow and not sure whether to converge
#    2) ```
#       loss = tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_conv, pos_weight=weight)
#       return tf.reduce_mean(loss)
#       ```
#       Here logits is the output directly from the final layer without softmax.
# ---------------------------------------------------------------------------------------------------------

def weighted_loss(y_true, y_softmax_conv, weight):
    """Compute weighted loss function per pixel.
    Loss = (1 - softmax(logits)) * targets * weight + softmax(logits) * (1 - targets) * weight

    Argument:
        y_true: [batch_size, depth, height, width, 1]
        weight_map: [batch_size, depth, height, width, 1]
        y_softmax_conv: [batch_size, depth, height, width, 2]
    """
    y_true = tf.to_float(tf.reshape(y_true[..., 0], [-1]))
    weight = tf.to_float(tf.reshape(weight[..., 0], [-1]))
    y_conv = tf.to_float(tf.reshape(y_softmax_conv[..., 1], [-1]))

    loss_pos = 1 / 2 * tf.pow((1 - y_conv), 2) * y_true * weight
    loss_neg = 1 / 2 * tf.pow(y_conv, 2) * (1 - y_true) * weight

    return tf.reduce_mean(loss_pos + loss_neg)

def evaluation_metrics(y_true, y_conv):
    y_true = tf.to_float(y_true)
    y_conv = tf.to_float(y_conv)
    intersection = tf.reduce_sum(y_conv * y_true)
    union = tf.reduce_sum(y_conv) + tf.reduce_sum(y_true)
    dice = 2.0 * intersection / union * 100
    return dice

def show_slices(im_3d, indices=None):
    """ Function to display slices of 3-d image """

    plt.rcParams['image.cmap'] = 'gray'

    if indices is None:
        indices = np.array(im_3d.shape) // 2
    assert len(indices) == 3, """Except 3-d array, but receive %d-d array
    indexing.""" % len(indices)

    x_th, y_th, z_th = indices
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(im_3d[x_th, :, :])
    axes[1].imshow(im_3d[:, y_th, :])
    axes[2].imshow(im_3d[:, :, z_th])
    plt.suptitle("Center slices for spine image")
