import os

import numpy as np
import nibabel as nib

from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
from skimage.transform import rotate, rescale
from skimage.exposure import rescale_intensity, adjust_gamma, adjust_sigmoid

def _augment(xs):
    """Image adjustment doesn't change image shape, but for intensity.

    Return:
        images: 4-d tensor with shape [depth, height, width, channels]
    """

    # `xs` has shape [depth, height, width] with value in [0, 1].
    brt_gamma, brt_gain = np.random.uniform(low=0.9, high=1.1, size=2)
    aj_bright = adjust_gamma(xs, brt_gamma, brt_gain)
    contrast_gain = np.random.uniform(low=5, high=10)
    aj_contrast = adjust_sigmoid(aj_bright, gain=contrast_gain)
    return aj_contrast

def _rotate_and_rescale(xs, ys):
    """Rotate images and labels and scale image and labels by a certain factor.
    Both need to swap axis from [depth, height, width] to [height, width, depth]
    required by skimage.transform library.
    """

    degree = np.int(np.random.uniform(low=-7, high=7))
    factor = np.random.uniform(low=0.9, high=1.2)
    # swap axis
    HWC_xs, HWC_ys = [np.transpose(item, [1, 2, 0]) for item in [xs, ys]]
    # rotate and rescale
    HWC_xs, HWC_ys = [rotate(item, degree, mode='symmetric', preserve_range=True) for item in [HWC_xs, HWC_ys]]
    HWC_xs, HWC_ys = [rescale(item, factor, mode='symmetric', preserve_range=True) for item in [HWC_xs, HWC_ys]]
    # swap back
    xs, ys = [np.transpose(item, [2, 0, 1]) for item in [HWC_xs, HWC_ys]]
    return xs, ys

def _translate(xs, ys):
    """Perform translate, and the displacement is skewed to 0.
        In detail, take samples from the modified power function distribution.
    """

    samples = np.random.power(5, size=4)               # samples now in range [0, 1]
    skewed_samples = np.int8((- samples + 2) * 15)     # skewed_samples in range [1, 16]
    r1, c1, r2, c2 = skewed_samples                    # discard 0 for indexing `-0`
    trans_xs, trans_ys = [item[:, r1: -r2, c1: -c2] for item in [xs, ys]]
    return trans_xs, trans_ys

def weights_map(ys):
    """Compute corresponding weight map when use cross entropy loss.

    Argument:
        ys: [depth, height, width]

    Return:
        weights_map: [depth, height, width]
    """
    weights = ys.astype(np.float64)

    # Balance class frequencies.
    cls_ratio = np.sum(1 - ys) / np.sum(ys)
    weights *= cls_ratio

    # Generate boundaries using morphological operation.
    se = generate_binary_structure(3, 1)
    bigger = binary_dilation(ys, structure=se).astype(np.float64)
    small = binary_erosion(ys, structure=se).astype(np.float64)
    edge = bigger - small

    # Balance edge frequencies.
    edge_ratio = np.sum(bigger) / np.sum(edge)
    edge *= np.exp(edge_ratio) * 10

    # `weights` should > 0
    # `targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))`
    return weights + edge + 1

def load_data(base_path='./data/Train/', nii_index=0):
    """Load nii data to numpy ndarray with **arbitrary** size.

    Return:
        tuples of (images, labels).
        + images: float32 dtype with normalization
        + labels: uint8 dtype
    """

    # use os.path.join() better probably
    label_path = [base_path + p for p in os.listdir(base_path) if p.endswith('Label.nii')]
    image_path = [p.replace('_Label', '') for p in label_path]

    xs, ys = [nib.load(p[nii_index]).get_data() for p in [image_path, label_path]]

    # Normalize image to range [0, 1] for image processing.
    local_max = np.max(xs, axis=(1, 2), keepdims=True)
    local_min = np.min(xs, axis=(1, 2), keepdims=True)
    # `xs` with dtype float64
    xs = (xs - local_min) / (local_max - local_min)

    # Image augmentation.
    xs, ys = _rotate_and_rescale(xs, ys)
    xs, ys = _translate(xs, ys)
    xs = _augment(xs)

    # Flip up-down or left-right.
    # --------------------------------------------------------------------------
    # flipud, fliplr = np.random.choice([1, -1], size=2)
    # xs = xs[:, ::flipud, ::fliplr]
    # ys = ys[:, ::flipud, ::fliplr]
    # --------------------------------------------------------------------------

    # Regenerate the binary label, just in case.
    ys = (ys > 0).astype(np.uint8)
    weights = weights_map(ys)

    xs, ys, weights = [item[..., np.newaxis] for item in [xs, ys, weights]]
    return xs, ys, weights

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
    local_max = np.max(xs, axis=(1, 2), keepdims=True)
    local_min = np.min(xs, axis=(1, 2), keepdims=True)
    xs = (xs - local_min) / (local_max - local_min)
    return xs[None, ..., None]
