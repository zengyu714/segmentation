import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# % matplotlib inline

from scipy import ndimage
from skimage import morphology
from skimage.measure import regionprops, label

from inputs import _banish_darkness


def localization(x, y):
    """Simple post-processing and get IVDs positons.

    Return:
        positons: calculated by `ndimage.measurements.center_of_mass`
        y:        after fill holes and remove small objects.
    """
    labels, nums = label(y, return_num=True)
    areas = np.array([prop.filled_area for prop in regionprops(labels)])
    assert nums >= 7,  'Fail in this test, should detect at least seven regions.'

    # Segment a joint region which should be separate (if any).
    while np.max(areas) > 10000:
        y = ndimage.binary_opening(y, structure=np.ones((3, 3, 3)))
        areas = np.array([prop.filled_area for prop in regionprops(label(y))])

    # Remove small objects.
    threshold = sorted(areas, reverse=True)[7]
    y = morphology.remove_small_objects(y, threshold + 1)

    # Fill holes.
    y = ndimage.binary_closing(y, structure=np.ones((3, 3, 3)))
    y = morphology.remove_small_holes(y, min_size=512, connectivity=3)

    positions = ndimage.measurements.center_of_mass(x, label(y), range(1, 8))
    return np.array(positions), y

def save_as_img(x, y, positions, savename):
    """Convert predicted `.npy` result to black-and-white image per slice.
    Overlapped with Inputs gray image and positons for better visual effects.

    Notice: positons **annotations** are drawn approximately at depth/2 in x-axis.
    """
    for i in range(x.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(x[i], 'gray')
        ax.imshow(y[i], 'jet', alpha=0.5)
        if i == x.shape[0] // 2:
            ax.plot(positions[:, 2], positions[:, 1], 'c+', ms=7)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(savename + str(i + 1) + '.png', bbox_inches='tight', dpi=x.shape[1])
        # plt.show()
        plt.close(fig)

def save_as_nii(y, savename):
    y_nii = nib.Nifti1Image(y.astype(np.uint8), np.eye(4))
    nib.save(y_nii, savename + '.nii')

def write_csv(positions, savename):
    """The 7 IVD centers in mm unit are stored from T11-T12 (the first one) to
    L5-S1 (the last one) in CSV format."""

    # Sort localizations.
    positions = np.array(sorted(positions, key=lambda i: i[2], reverse=True))

    # Convert to mm unit.
    # The resolution of all images were resampled to 2 mm × 1.25 mm × 1.25 mm.
    positions *= np.array([2, 1.25, 1.25])

    csv_file = open(savename + '.csv', 'w+', newline='')
    writer = csv.writer(csv_file)
    writer.writerows(positions)
    csv_file.close()

def submit():
    """Assume:
        predicted file is stored at `./pred/`
        Inputs file is stored at `./Test/`, say, 'Test_Subject01.nii'
    """
    test_file_base = './data/Test/Test_Subject'
    pred_file_base = './pred/vnet_'

    for idx in range(10):
        test_filename = test_file_base + str(idx + 1).zfill(2) + '.nii'
        pred_filename = pred_file_base + str(idx) + '.npy'

        x = nib.load(test_filename).get_data()
        y_clipped = np.load(pred_filename)[0]

        # Restore clipped results back to inputs size.
        y = np.zeros_like(x, dtype=np.bool)
        *_, top, bottom = _banish_darkness(x, y)
        y[:, top: bottom] = y_clipped

        # Localize predicted results and includs post-processing this step.
        positions, y = localization(x, y)

        # Save results.
        savename = str(idx + 1).zfill(2)
        write_csv(positions, './final_results/' + savename)
        save_as_nii(y, './final_results/' + savename)
        save_as_img(x, y, positions, './visualize/' + savename + '_')
        print('Test Subject {} has done.'.format(idx))

if __name__ == '__main__':
    submit()
