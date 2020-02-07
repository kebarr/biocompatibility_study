import numpy as np
from scipy import ndimage
from PIL import Image
from skimage import exposure
from skimage.filters import rank_order
from scipy.ndimage.morphology import binary_dilation
from skimage import measure
import os




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
     for p in props:
         centroid = p.centroid
         x_coord = int(centroid[0])
         y_coord = int(centroid[1])
         bitmap[x_coord, y_coord] = 1
     return bitmap


# helper class to store results
class CountResults(object):
    def __init__(self):
        self.sums = []
        self.region_counts = []
        self.areas = []


def count_cells(image_arr, injection_site, pixels_per_iteration, iterations_needed=200):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    for x, y in injection_site_coords:
        mask[x,y] = 1
    intensity = np.sum(image_arr)
    res = CountResults()
    area = np.sum(mask)
    for i in range(iterations_needed):
        masked = mask*image_arr
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_counts.append(intensity-np.sum(masked))
        intensity = np.sum(masked)
        res.sums.append(intensity)
        mask = binary_dilation(mask, iterations=pixels_per_iteration)
        res.areas.append(np.sum(mask)-area)
        area = np.sum(mask)
    return res


def run_analysis(filename, injection_site):
    # open image as greyscale
    im = Image.open(filename).convert('L')
    data = np.array(im, dtype=float)
    # localise injection site
    segmented = segment_neun(data)
    # label and calculate properties of cells
    props = measure.regionprops(measure.label(segmented))
    x_dim = data.shape[0]
    y_dim = data.shape[1]
    # find centroids of cells
    bm = create_centroid_bitmap(x_dim, y_dim, props)
    # count cells in regions 
    final = count_cells(bm, injection_site, 1, iterations_needed=100)
    return final





