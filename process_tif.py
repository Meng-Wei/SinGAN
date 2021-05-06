# Combine three tiff into one tiff.
from skimage import io as img
from skimage import color, morphology, filters
from skimage.external import tifffile
from skimage.util import img_as_ubyte
import numpy as np

from os import listdir
from os.path import isfile, join



def read_image(image):
    x = img.imread(image)
    print(x.shape, np.max(x), np.min(x))
    return x


group = range(4, 5)
band = [2, 3, 4]

for g in group:
    cur_group = '../thesis/dataset/group%d/' % g
    files = [f for f in listdir(cur_group) if isfile(join(cur_group, f))]
    files.sort()

    images = []
    for f in files:
        if 'DS_Store' in f:
            continue
        print(join(cur_group, f))
        cur_image = tifffile.imread(join(cur_group, f))
        images.append(cur_image)
    images = np.array(images)
    # save_name = join(cur_group, 'image%d.tif' % g)
    save_name = join('../thesis/dataset/dense/image%d.tif' % g)
    tifffile.imsave(save_name, images)