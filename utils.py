import os

import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

import nibabel as nib
import tensorflow as tf


def dice_loss(y_true, y_conv):
    """Compute dice among **positive** labels to avoid unbalance.

    Argument:
        y_true: [batch_size, depth, height, width, 1]
        y_conv: [batch_size, depth, height, width, 2]
    """
    y_true = tf.to_float(tf.reshape(y_true[..., 0], [-1]))
    y_conv = tf.to_float(tf.reshape(y_conv[..., 1], [-1]))
    intersection = tf.reduce_sum(y_conv * y_true)
    union = tf.reduce_sum(y_conv * y_conv) + tf.reduce_sum(y_true * y_true)  # y_true is binary
    dice = 2.0 * intersection / union
    return 1 - tf.clip_by_value(dice, 0, 1.0 - 1e-7)

def cross_entropy_loss(y_true, y_conv, weight):
    """Compute weighted cross entropy multiplied by weight_map.

    Argument:
        y_true: [batch_size, depth, height, width, 1]
        y_conv: [batch_size, depth, height, width, 2]
        weight_map: [batch_size, depth, height, width, 1]
    """
    y_true = tf.to_float(tf.reshape(y_true[..., 0], [-1]))
    y_conv = tf.to_float(tf.reshape(y_conv[..., 1], [-1]))
    weight = tf.to_float(tf.reshape(weight[..., 0], [-1]))
    
    loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true, logits=y_conv, pos_weight=weight)
    return tf.reduce_mean(loss)
    
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

def plot_image_and_prediction(i=0, j_slice=None):
    data_path = os.listdir('./data/Test/')
    data = nib.load('./data/Test/' + data_path[i]).get_data()
    pred_label = np.load('./pred/check_' + str(i) + '.npy')[0]

    if j_slice is None:
        j_slice = pred_label.shape[0] // 2

    plt.imshow(data[j_slice, :, :], 'gray')
    plt.imshow(pred_label[j_slice, :, :], 'jet', alpha=0.5)
    plt.show()
