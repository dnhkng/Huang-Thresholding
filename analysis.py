import numpy
import math
import scipy.ndimage
import skimage.measure

def update_progress(progress):
    print '\r[{0}] {1}%'.format('#'*(progress/10), progress)
    
def mask_image_initial(data):
    """ This function generates a mask, that sets the area outside the plate to black.
    Its basically an array the size of the picture, with a filled circle in the middle """
    #  these are the parameters for the mask, such as height and width, and diameter of the circle.
    h, w = 1040, 1392
    a, b = 523, 695
    r = 490
    y, x = numpy.ogrid[-a:h - a, -b:w - b]
    #Thanks, Mr. Pythagoras!
    mask = x * x + y * y <= r * r
    array = numpy.zeros((h, w))
    array[mask] = 1
    masked = data * array
    array = numpy.ones((h, w)) * numpy.mean(masked[mask])
    array[mask] = 0
    masked = masked + array
    return masked


def remove_background(data):
    """Remove background and stretchs data to 8-bit range"""
    # convert to floating point (integers lose 99% of data!)
    data = data.astype(float)
    data = numpy.fabs(4096 - data)
    # remove strongly blurred 'background' from slightly blurred foreground
    no_background = scipy.ndimage.median_filter(data, 2) / scipy.ndimage.median_filter(data, 20) * numpy.mean(data)
    #normalizes images to 256 level depth
    minimum_value = numpy.min(no_background)
    no_background -= minimum_value
    maximum_value = numpy.max(no_background)
    no_background *= (256 / maximum_value)
    return no_background


def mask_image_final(data):
    """ This function generates a mask, that sets the area outside the plate to black.
    Its basically an array the size of the picture, with a filled circle in the middle """
    # these are the parameters for the mask, such as height and width, and diameter of the circle.
    h, w = 1040, 1392
    a, b = 523, 695
    r = 484
    y, x = numpy.ogrid[-a:h - a, -b:w - b]
    #Thanks, Mr. Pythagorus!
    mask = x * x + y * y <= r * r
    array = numpy.zeros((h, w) )
    array[mask] = 1
    masked = data * array
    return masked


def huang(data):
    """Implements Huang's fuzzy thresholding method 
        Uses Shannon's entropy function (one can also use Yager's entropy function) 
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""
    threshold = -1
    first_bin = 0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = 254
    for ih in range(254, -1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / (last_bin - first_bin)
    mu_0 = numpy.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin, 254):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_0[ih] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    mu_1 = numpy.zeros(shape=(254, 1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(last_bin, 1, -1):
        sum_pix += (ih * data[ih])
        num_pix += data[ih]
        mu_1[ih - 1] = sum_pix / num_pix  # NUM_PIX cannot be zero !
    min_ent = float("inf")
    for it in range(254):
        ent = 0.0
        for ih in range(it):
            # Equation (4) in Reference
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_0[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        for ih in range(it + 1, 254):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * math.fabs(ih - mu_1[it]))
            if not ((mu_x < 1e-06) or (mu_x > 0.999999)):
                # Equation (6) & (8) in Reference
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
        update_progress(256)    
    return threshold


def process_image(data):
    """Returns a object map derived from an image"""
    image_masked = mask_image_initial(data)
    image_clean = remove_background(image_masked)
    histogram, bin_edges = numpy.histogram(image_clean, bins=range(257))
    huang_threshold = huang(histogram)
    threshold = numpy.where(image_clean > huang_threshold, 1, 0)
    # cleans up image
    threshold = scipy.ndimage.binary_erosion(threshold, structure=numpy.ones((4, 4))).astype(threshold.dtype)
    threshold = scipy.ndimage.binary_dilation(threshold, structure=numpy.ones((2, 2))).astype(threshold.dtype)
    # finds objects
    object_map, count = scipy.ndimage.label(threshold)
    properties = skimage.measure.regionprops(object_map, properties=['Area', 'Perimeter', 'Centroid'])
    # list of objects that should be removed (too small, non circular etc.)
    remove_object = numpy.zeros(count + 1, dtype=bool)
    for i in range(1, count):
        if properties[i - 1]['Area'] < 10:
            remove_object[i] = True
        if (4 * math.pi * properties[i - 1]['Area'] /
                (properties[i - 1]['Perimeter'] * properties[i - 1]['Perimeter']) <= 0.9):
            remove_object[i] = True
    # remove the pixels of the unwanted objects from the map by setting them to zero
    remove_pixel = remove_object[object_map]
    object_map[remove_pixel] = 0
    # remove false object generated end the plate edge
    object_map = mask_image_final(object_map)
    # find the remaining objects, and print their properties
    object_map, count = scipy.ndimage.label(object_map)

    return object_map


def get_positions(object_map):
    """docstring for positions"""
    locations = skimage.measure.regionprops(object_map, properties=['Centroid'])
    locations = [i['Centroid'] for i in locations]
    return locations

if __name__ == "__main__":
    # Load and process the image
    image = numpy.load('colonies.npy')
    objects = process_image(image)
    positions = get_positions(objects)

    # Plot the found colonies!
    import matplotlib.pyplot as plt

    plt.imshow(objects, cmap='jet')
    plt.show()