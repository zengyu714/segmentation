import os

import numpy as np
import nibabel as nib

from skimage.transform import resize
from skimage.exposure import rescale_intensity, adjust_gamma, adjust_sigmoid

def _augment(xs):
    """Image adjustment doesn't change image shape, but for intensity.

    Return:
        images: 4-d tensor with shape [depth, height, width, channels]
    """

    # `xs` has shape [depth, height, width] with value in [0, 1]
    brt_gamma, brt_gain = np.random.uniform(low=0.75, high=1.25, size=2)
    aj_bright = adjust_gamma(xs, brt_gamma, brt_gain)
    cta_gain = np.random.uniform(low=9, high=11)
    aj_contrast = adjust_sigmoid(aj_bright, gain=cta_gain)

    return aj_contrast

def _translate(xs, ys):
    """Perform translate and zoom function."""
    r1, c1, r2, c2 = np.random.uniform(30, size=4).astype(np.uint8)
    trans_xs, trans_ys = [item[:, r1: r2, c1: c2] for item in [xs, ys]]
    xs, ys = [resize(item, (36, 300, 300), preserve_range=True) for item in [xs, ys]]
    return xs, ys

def load_data(base_path='./data/Train/', mode='train', nii_index=0):
    """Load nii data to numpy ndarray.

    Return:
        tuples of (images, labels) with arbitrary shape.
        + images: float32 dtype [36, 300, 300, 1] with normalization
        + labels: uint8 dtype [36, 300, 300, 1]
    """

    # Better use os.path.join() probably
    label_path = [base_path + p for p in os.listdir(base_path) if p.endswith('Label.nii')]
    image_path = [p.replace('_Label', '') for p in label_path]

    xs, ys = [nib.load(p[nii_index]).get_data() for p in [image_path, label_path]]
    resized_x, resized_y = [resize(item, (36, 300, 300), preserve_range=True) for item in [xs, ys]]

    # Rescale the image to just the positive (0, 255) range
    xs = rescale_intensity(resized_x, out_range=np.uint8)
    xs = xs / np.max(xs)            # Convert to float64.

    if mode == 'train':
        xs = _augment(xs)
        xs, ys = _translate(xs, ys)
        flipud, fliplr = np.random.choice([1, -1], size=2)
        xs = xs[:, ::flipud, ::fliplr]       # flip up-down or left-right
        ys = ys[:, ::flipud, ::fliplr]

    xs, ys = [item[..., np.newaxis] for item in [xs, ys]]

    return xs, ys
