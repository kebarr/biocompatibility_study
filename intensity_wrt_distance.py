import numpy as np
from scipy.ndimage.morphology import binary_dilation

class IntensityResults(object):
    def __init__(self):
        self.sums = []
        self.region_intensities = []
        self.intensities_in_masks = []
        self.areas_full = []
        self.areas_with_previous_subtracted = []


    def average_intensity_per_region(self):
        return [int(i)/int(a) for i,a in zip(self.region_intensities, self.areas_with_previous_subtracted)]


def compare_intensities(image_arr, injection_site, iterations_needed):
    injection_site_coords = injection_site.coords
    mask = np.zeros_like(image_arr)
    for x, y in injection_site_coords:
        mask[x,y] = 1
    initial_area = np.sum(mask)
    # problem is that first intensity is entire image- initial mask, need 
    masked = np.ma.masked_array(image_arr,mask=~np.array(mask, dtype=bool))
    intensity_of_first_mask = np.sum(masked.compressed())
    mask = binary_dilation(mask)
    masked = np.ma.masked_array(image_arr,mask=~mask)
    intensity = np.sum(masked.compressed()) - intensity_of_first_mask # so first intensity will actually be intensity of everything outside mask
    res = IntensityResults()
    for i in range(iterations_needed):
        # intensity in region is total intensity including region - total instensity excluding region
        res.region_intensities.append(intensity)
        intensity = np.sum(masked.compressed())
        res.intensities_in_masks.append(intensity)
        res.areas_full.append(np.sum(mask))
        area = np.sum(mask)
        mask = binary_dilation(mask)
        masked = np.ma.masked_array(image_arr,mask=~mask)
        intensity = np.sum(masked.compressed()) - intensity   
        area = np.sum(mask) - area
        res.areas_with_previous_subtracted.append(area)
        if i%20 == 0: 
            print("finished iteration %d" % (i))
    return res


def run_analysis(img, base_filename, injection_site):
    intensities = compare_intensities(img*255, injection_site,300)
    final = intensities.average_intensity_per_region()
    res_filename = base_filename.split(".tif")[0] + "_results.txt"
    with open(res_filename, 'w') as f:
        for i in final:
            f.write(str(i) +", ")
        f.write("\n")
        for i in intensities.region_intensities:
            f.write(str(i) +", ")