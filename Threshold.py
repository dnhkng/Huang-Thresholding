import numpy
import math

def Huang(data):
    """Implements Huang's fuzzy thresholding method 
        Uses Shannon's entropy function (one can also use Yager's entropy function) 
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing  
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51"""
        
    threshold=-1;

    first_bin=  0
    for ih in range(254):
        if data[ih] != 0:
            first_bin = ih
            break
     
    last_bin=254;
    for ih in range(254,-1,-1):
        if data[ih] != 0:
            last_bin = ih
            break

    term = 1.0 / (last_bin - first_bin)
    
    print first_bin, last_bin, term
    mu_0 = numpy.zeros(shape=(254,1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(first_bin,254):
        sum_pix = sum_pix + (ih * data[ih])
        num_pix = num_pix + data[ih]
        mu_0[ih] = sum_pix / num_pix # NUM_PIX cannot be zero !

    mu_1 = numpy.zeros(shape=(254,1))
    num_pix = 0.0
    sum_pix = 0.0
    for ih in range(last_bin, 1, -1 ):
        sum_pix = sum_pix + (ih * data[ih])
        num_pix = num_pix + data[ih]

        mu_1[ih-1] = sum_pix / num_pix # NUM_PIX cannot be zero !

    min_ent = float("inf")
    for it in range(254): 
        ent = 0.0
        for ih in range(it):
            # Equation (4) in Reference
            mu_x = 1.0 / ( 1.0 + term * math.fabs( ih - mu_0[it]))
            if ( not ((mu_x  < 1e-06 ) or (mu_x > 0.999999))):

                # Equation (6) & (8) in Reference
                ent = ent + data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x) )
        
        
        for ih in range(it + 1, 254):
            # Equation (4) in Ref. 1 */
            mu_x = 1.0 / (1.0 + term * math.fabs( ih - mu_1[it]))
            if ( not((mu_x  < 1e-06 ) or ( mu_x > 0.999999))):
                # Equation (6) & (8) in Reference
                ent = ent + data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x) )  
        if (ent < min_ent):
            min_ent = ent
            threshold = it
        print "min_ent, threshold ", min_ent, threshold
    return threshold