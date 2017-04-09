import os

import numpy as np
import nibabel as nib

from skimage.transform import rotate
from skimage.exposure import rescale_intensity, adjust_gamma, adjust_sigmoid

def _augment(xs):
    """Image adjustment doesn't change image shape, but for intensity.

    Return:
        images: 4-d tensor with shape [depth, height, width, channels]
    """

    # `xs` has shape [depth, height, width] with value in [0, 1].
    brt_gamma, brt_gain = np.random.uniform(low=0.75, high=1.25, size=2)
    aj_bright = adjust_gamma(xs, brt_gamma, brt_gain)
    contrast_gain = np.random.uniform(low=5, high=12)
    aj_contrast = adjust_sigmoid(aj_bright, gain=contrast_gain)

    return aj_contrast

def _rotate(xs, ys):
    """Rotate images and labels."""

    degree = np.int(np.random.uniform(low=-15, high=15))
    # Original order is [channels, height, width].
    HWC_xs, HWC_ys = [np.transpose(item, [1, 2, 0]) for item in [xs, ys]]
    xs, ys = [rotate(item, degree, mode='symmetric', preserve_range=True) for item in [HWC_xs, HWC_ys]]
    # Back to [height, width, channels].
    xs, ys = [np.transpose(item, [2, 0, 1]) for item in [xs, ys]]
    return xs, ys

def _translate(xs, ys):
    """Perform translate, and the displacement is skewed to 0. Specifically,
        sampling from the modified power function distribution.
    """

    samples = np.random.power(5, size=4)               # samples now in range [0, 1]
    skewed_samples = np.uint8((- samples + 2) * 30)    # skewed_samples in range [1, 31]
    r1, c1, r2, c2 = skewed_samples                    # discard 0 for indexing `-0`
    trans_xs, trans_ys = [item[:, r1: -r2, c1: -c2] for item in [xs, ys]]
    return trans_xs, trans_ys

def load_data(base_path='./data/Train/', nii_index=0):
    """Load nii data to numpy ndarray with **arbitrary** size.

    Return:
        tuples of (images, labels).
        + images: float32 dtype with normalization
        + labels: uint8 dtype
    """

    # Better use os.path.join() probably.
    label_path = [base_path + p for p in os.listdir(base_path) if p.endswith('Label.nii')]
    image_path = [p.replace('_Label', '') for p in label_path]

    xs, ys = [nib.load(p[nii_index]).get_data() for p in [image_path, label_path]]

    # Rescale the image to just the positive (0, 255) range.
    xs = rescale_intensity(xs, out_range=np.uint8)
    xs = xs / np.max(xs)                  # convert to float64

    # Image augmentation.
    xs = _augment(xs)
    xs, ys = _rotate(xs, ys)
    xs, ys = _translate(xs, ys)

    flipud, fliplr = np.random.choice([1, -1], size=2)
    xs = xs[:, ::flipud, ::fliplr]       # flip up-down or left-right
    ys = ys[:, ::flipud, ::fliplr]

    # Normalize images.
    xs = (xs - np.mean(xs)) / np.std(xs)
    # Regenerate the binary label.
    ys = (ys > 0).astype(np.uint8)

    xs, ys = [item[..., np.newaxis] for item in [xs, ys]]
    return xs, ys

def load_inference(base_path='./data/Test/Test_Subject', nii_index=0):
    """Load nii data, whose name is, for example, 'Test_Subject01.nii'.

    Arguments:
        nii_index: counts from 0.
    """
    filename = base_path + str(nii_index + 1).zfill(2) + '.nii'
    xs = nib.load(filename).get_data()

    xs = rescale_intensity(xs, out_range=np.uint8)
    xs = xs / np.max(xs)

    # Normalize images.
    xs = (xs - np.mean(xs)) / np.std(xs)
    return xs[None, ..., None]
