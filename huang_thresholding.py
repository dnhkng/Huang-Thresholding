import math

import numpy as np


class HuangThresholding:
    def __init__(self, data):
        self.data = data
        self.first_bin, self.last_bin = self.find_bin_limits()
        self.term = 1.0 / max(1, self.last_bin - self.first_bin)
        self.mu_0, self.mu_1 = self.calculate_mu()

    def find_bin_limits(self):
        """
        Find the first and last non-zero bins.
        """
        non_zero_indices = np.nonzero(self.data)[0]
        first_bin = non_zero_indices[0]
        last_bin = non_zero_indices[-1]
        return first_bin, last_bin

    def calculate_mu(self):
        """
        Calculate mu_0 and mu_1 using vectorized operations for efficiency.
        """
        indices = np.arange(len(self.data))
        num_pix_cumsum = np.cumsum(self.data)
        sum_pix_cumsum = np.cumsum(indices * self.data)
        mu_0 = sum_pix_cumsum / np.where(num_pix_cumsum == 0, 1, num_pix_cumsum)
        
        num_pix_cumsum_rev = np.cumsum(self.data[::-1])[::-1]
        sum_pix_cumsum_rev = np.cumsum((indices[::-1]) * self.data[::-1])[::-1]  # Use indices dynamically
        mu_1 = sum_pix_cumsum_rev / np.where(num_pix_cumsum_rev == 0, 1, num_pix_cumsum_rev)

        return mu_0, mu_1

    def calculate_entropy(self, it):
        """
        Calculate entropy for a given threshold.
        """
        ent = 0.0
        for ih in range(it):
            mu_x = 1.0 / (1.0 + self.term * abs(ih - self.mu_0[it]))
            if not (mu_x < 1e-6 or mu_x > 1 - 1e-6):
                ent -= self.data[ih] * (
                    mu_x * math.log(mu_x) + (1.0 - mu_x) * math.log(1.0 - mu_x)
                )

        for ih in range(it + 1, len(self.data)):
            mu_x = 1.0 / (1.0 + self.term * abs(ih - self.mu_1[it]))
            if not (mu_x < 1e-6 or mu_x > 1 - 1e-6):
                ent -= self.data[ih] * (
                    mu_x * math.log(mu_x) + (1.0 - mu_x) * math.log(1.0 - mu_x)
                )

        return ent

    def find_threshold(self):
        """
        Implements Huang's fuzzy thresholding method.
        Uses Shannon's entropy function (one can also use Yager's entropy function).
        Huang L.-K. and Wang M.-J.J. (1995) "Image Thresholding by Minimizing
        the Measures of Fuzziness" Pattern Recognition, 28(1): 41-51
        """
        threshold = -1
        min_ent = float("inf")
        for it in range(self.first_bin, self.last_bin + 1):
            ent = self.calculate_entropy(it)
            if ent < min_ent:
                min_ent = ent
                threshold = it

        return threshold
