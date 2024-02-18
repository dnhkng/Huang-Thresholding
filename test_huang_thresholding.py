import unittest

import numpy as np

from huang_thresholding import HuangThresholding


class TestHuangThresholding(unittest.TestCase):
    def setUp(self):
        self.data, _ = np.histogram(np.load("colonies.npy"), bins=range(257))
        self.huang_thresholding = HuangThresholding(self.data)

    def test_find_bin_limits(self):
        first_bin, last_bin = self.huang_thresholding.find_bin_limits()
        self.assertEqual(first_bin, 105)
        self.assertEqual(last_bin, 255)

    def test_calculate_mu(self):
        mu_0, mu_1 = self.huang_thresholding.calculate_mu()
        self.assertTrue(np.allclose(mu_0[105:108], [105., 105.66666667, 106.33333333]))
        self.assertTrue(np.allclose(mu_1[251:254], [253.35519838, 253.80931545, 254.26037546]))

    def test_calculate_entropy(self):
        ent = self.huang_thresholding.calculate_entropy(125)
        self.assertAlmostEqual(ent, 208572.86939064699)

    def test_find_threshold(self):
        threshold = self.huang_thresholding.find_threshold()
        self.assertEqual(threshold, 163)


if __name__ == "__main__":
    unittest.main()
