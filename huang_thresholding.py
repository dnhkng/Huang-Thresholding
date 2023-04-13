import math

import numpy as np


class HuangThresholding:
    def __init__(self, data):
        self.data = data
        self.first_bin, self.last_bin = self.find_bin_limits()
        self.term = 1.0 / (self.last_bin - self.first_bin)
        self.mu_0, self.mu_1 = self.calculate_mu()

    def find_bin_limits(self):
        """
        Find the first and last non-zero bins.
        """
        first_bin = next(i for i, x in enumerate(self.data[:254]) if x != 0)
        last_bin = next(i for i, x in enumerate(reversed(self.data[:255])) if x != 0)

        return first_bin, 254 - last_bin

    def calculate_mu(self):
        """
        Calculate mu_0 and mu_1.
        """
        mu_0 = np.zeros(254)
        num_pix = sum_pix = 0.0
        for ih in range(self.first_bin, 254):
            sum_pix += ih * self.data[ih]
            num_pix += self.data[ih]
            mu_0[ih] = sum_pix / num_pix

        mu_1 = np.zeros(254)
        num_pix = sum_pix = 0.0
        for ih in range(self.last_bin, 1, -1):
            sum_pix += ih * self.data[ih]
            num_pix += self.data[ih]
            mu_1[ih - 1] = sum_pix / num_pix

        return mu_0, mu_1

    def calculate_entropy(self, it):
        """
        Calculate entropy for a given threshold.
        """
        ent = 0.0
        for ih in range(it):
            mu_x = 1.0 / (1.0 + self.term * abs(ih - self.mu_0[it]))
            if not (mu_x < 1e-6 or mu_x > 0.999999):
                ent += self.data[ih] * (
                    -mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x)
                )

        for ih in range(it + 1, 254):
            mu_x = 1.0 / (1.0 + self.term * abs(ih - self.mu_1[it]))
            if not (mu_x < 1e-6 or mu_x > 0.999999):
                ent += self.data[ih] * (
                    -mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x)
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
        for it in range(254):
            ent = self.calculate_entropy(it)
            if ent < min_ent:
                min_ent = ent
                threshold = it

        return threshold
