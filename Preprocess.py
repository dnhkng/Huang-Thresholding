import Threshold
import numpy
import math
import scipy.ndimage
import skimage.measure


image = numpy.load('colonies.npy')


def maskImageInitial(image):
    """ This function generates a mask, that sets the area outside the plate to black. Its basically an array the size of the picture, with a filled circle in the middle """	
    # these are the parameters for the mask, such as height and width, and diameter of the circle.
    h,w = 1040, 1392
    a,b = 523, 695
    r = 490
    y,x = numpy.ogrid[-a:h-a, -b:w-b]
    #Thanks, Mr. Pythagorus!
    mask = x*x + y*y <= r*r
    array = numpy.zeros((h, w))
    print x
    array[mask] = 1
    masked = image*array
    array = numpy.ones((h, w))*numpy.mean(masked[mask])
    array[mask] = 0
    masked = masked + array
    return masked


def removeBackgroup(image):
    """Remove background and stretchs data to 8-bit range"""
    image = image.astype(float) #convert to floating point (integers lose 99% of data!)
    image = numpy.fabs(4096 - image)
    noBackground = scipy.ndimage.median_filter(image,2)/scipy.ndimage.median_filter(image,20)*numpy.mean(image) # remove stongly blurred 'background' from slightly blurred foregound
    minval = numpy.min(noBackground) #normalizes images to 256 level depth
    noBackground = noBackground - minval
    maxval = numpy.max(noBackground)
    noBackground = noBackground * (256/maxval)
    return noBackground


def maskImageFinal(image):
    """ This function generates a mask, that sets the area outside the plate to black. Its basically an array the size of the picture, with a filled circle in the middle """	
    # these are the parameters for the mask, such as height and width, and diameter of the circle.
    h,w = 1040, 1392
    a,b = 523, 695
    r = 484
    y,x = numpy.ogrid[-a:h-a, -b:w-b]
    #Thanks, Mr. Pythagorus!
    mask = x*x + y*y <= r*r
    array = numpy.zeros((h, w))
    array[mask] = 1
    masked = image*array
    return masked

image_masked = maskImageInitial(image)
imageClean = removeBackgroup(image_masked)

histogram, bin_edges = numpy.histogram(imageClean, bins = range(257))
huangThreshold = Threshold.Huang(histogram)

thresholded = numpy.where(imageClean > huangThreshold, 1, 0)

# cleans up image
thresholded = scipy.ndimage.binary_erosion(thresholded, structure=numpy.ones((4,4))).astype(thresholded.dtype)
thresholded = scipy.ndimage.binary_dilation(thresholded, structure=numpy.ones((2,2))).astype(thresholded.dtype)

# finds objects
objects, count = scipy.ndimage.label(thresholded)

# size and brightness of objects
sizes = scipy.ndimage.sum(thresholded, objects, range(count + 1))
mean_vals = scipy.ndimage.sum(imageClean, objects, range(1, count + 1))


# find the properties of each object, list of dictionaries
properties = skimage.measure.regionprops(objects, properties=['Area', 'Perimeter','Centroid'])

# list of objects that should be removed (too small, non circular etc.)
removeObject = numpy.zeros(count+1,dtype=bool)    #[False] * num_features
for i in range(1,count): 
    if  properties[i-1]['Area'] < 10:
        removeObject[i] = True
    if (4 * math.pi * properties[i-1]['Area'] / (properties[i-1]['Perimeter']*properties[i-1]['Perimeter']) <= 0.9):
        removeObject[i] = True

# remove the pixels of the unwanted objects from the map by setting them to zero
remove_pixel = removeObject[objects]
objects[remove_pixel] = 0

# remove false object generated end the plate edge
objects = maskImageFinal(objects)


# find the remaining objects, and print their properties
objects, count = scipy.ndimage.label(objects)
properties = skimage.measure.regionprops(objects, properties=['Area', 'Perimeter','Centroid'])
for i in range(len(properties)):
    print 'Colony: ', i, properties[i-1]['Centroid'], properties[i-1]['Area']


# Plot the found colonies!
import matplotlib.pyplot as plt
plt.imshow(objects, cmap='jet')
plt.show()