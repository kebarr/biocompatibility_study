import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import exposure
from skimage.filters import rank_order
from scipy.ndimage.morphology import binary_dilation
from skimage import morphology
from skimage import measure



def segment_neun(data):
    lowpass = ndimage.gaussian_filter(data, 4)
    # high pass filtered image
    labels = data - lowpass
    # mask regions with intensity greater than 0
    mask = labels >= 1
    # rank order non zero image pixels
    labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
    # rescale to normal range for greyscale images
    rescaled = exposure.rescale_intensity(labels, out_range=(0, 255))
    markers = np.zeros_like(rescaled)
    markers[rescaled>200] = 1
    return markers


def create_centroid_bitmap(x_dim, y_dim, props):
     bitmap = np.zeros((x_dim, y_dim))
     number_added = 0
     for p in props:
         centroid = p.centroid
         x_coord = int(centroid[0])
         y_coord = int(centroid[1])
         bitmap[x_coord, y_coord] = 1
         number_added += 1
     print("added %d points to bitmap" % number_added)
     return bitmap


def run_analysis(filename):
    # open image as greyscale
    im = Image.open(filename).convert('L')
    data = np.array(im, dtype=float)
    # localise injection site
    i = get_injection_site_props(data, 115)
    segmented = segment_neun(data)
    # label and calculate properties of cells
    props = measure.regionprops(measure.label(segmented))
    x_dim = data.shape[0]
    y_dim = data.shape[1]
    # find centroids of cells
    bm = create_centroid_bitmap(x_dim, y_dim, props)
    # count cells in regions 
    final = compare_intensities(bm, i, 5, iterations_needed=100)
    return final





