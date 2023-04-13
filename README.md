# Huang-Thresholding
A nice thresholding algorithm for bacterial colony isolation
Implements Huang's fuzzy thresholding method, recoded from the Java version from ImageJ.

Uses Shannon's entropy function, from *Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51*


### Installation

Copy the python files to your working directory.

### Testing
```Python
python -m unittest test_huang_thresholding.py
```
### Usage 
There are bacterial colonies images in 'colonies.npy'

To calculate the Huang Thresholding, you would use the following code to first get the data into numpy and get the histogram values, and then calculate the threshold.

```Python
import numpy as np
from huang_thresholding import HuangThresholding

colony_data = np.load("colonies.npy"), bins=range(257)
histogram_data, _ = np.histogram(colony_data, bins=range(257))
huang_thresholding = HuangThresholding(histogram_data)
threshold = huang_thresholding.find_threshold()
```