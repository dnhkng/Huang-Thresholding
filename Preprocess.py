import numpy
import scipy.ndimage as spim
colonies = numpy.load('colonies.npy')

def pepareImage(image):
    """Remove background and fit data to 8-bit range"""
    image = image.astype(float) #convert to floating point (integers lose 99% of data!)
    colonies_blurred = spim.median_filter(colonies,2)/spim.median_filter(colonies,20)*numpy.mean(colonies) # remove stongly blurred 'background' from slightly blurred foregound
    minval = numpy.min(colonies_blurred)
    colonies_blurred = colonies_blurred - minval
    maxval = numpy.max(colonies_blurred)
    colonies_blurred = colonies_blurred * (256/maxval)

colonies = numpy.load('colonies.npy')




numpy.save("processed", colonies_blurred)

import matplotlib.pyplot as plt
plt.imshow(colonies_blurred)
plt.show()
