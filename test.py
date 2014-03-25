import numpy
import math

import Threshold


colonies = numpy.load('processed.npy')

hist, bin_edges = numpy.histogram(colonies, bins = range(257))


x = Threshold.Huang(hist)

y = numpy.where(colonies > x, 1, 0)



import scipy.ndimage as spim
from skimage.feature import peak_local_max
from skimage.morphology import watershed,label, closing, square
import skimage.measure

y = spim.binary_erosion(y, structure=numpy.ones((4,4))).astype(y.dtype)
y = spim.binary_dilation(y, structure=numpy.ones((2,2))).astype(y.dtype)


labeled_array, num_features = spim.label(y)#[0]


sizes = spim.sum(y, labeled_array, range(num_features + 1))
mean_vals = spim.sum(colonies, labeled_array, range(1, num_features + 1))


p = skimage.measure.regionprops(labeled_array, properties=['Area', 'Perimeter','Centroid'])

goodColonies = numpy.zeros(num_features+1,dtype=bool)    #[False] * num_features




for i in range(1,num_features):  
    if (4 * math.pi * p[i-1]['Area'] / (p[i-1]['Perimeter']*p[i-1]['Perimeter']) <= 0.99):
        goodColonies[i] = True


remove_pixel = goodColonies[labeled_array]
labeled_array[remove_pixel] = 0  
labeled_array, num_features = spim.label(labeled_array)
p = skimage.measure.regionprops(labeled_array, properties=['Area', 'Perimeter','Centroid'])
for i in range(len(p)):
    print 'Colony: ', i, p[i-1]['Centroid'], p[i-1]['Area']



import matplotlib.pyplot as plt
plt.imshow(labeled_array, cmap='gray')
plt.show()