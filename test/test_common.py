import unittest
import animo
import numpy as np


class TestXYData(unittest.TestCase):

    def test_init(self):
        xy = animo.XYData(np.array([0.0, 1.0, 1.5, 2.0]), np.array([1.0, 2.0, 4.0, 5.0]))
        self.assertEqual(len(xy.x), 4)
        self.assertEqual(len(xy.y), 4)
        self.assertEqual(xy.x[0], 0.0)
        self.assertEqual(xy.x[1], 1.0)
        self.assertEqual(xy.x[2], 1.5)
        self.assertEqual(xy.x[3], 2.0)
        self.assertEqual(xy.y[0], 1.0)
        self.assertEqual(xy.y[1], 2.0)
        self.assertEqual(xy.y[2], 4.0)
        self.assertEqual(xy.y[3], 5.0)

    def test_unequal_length(self):
        self.assertRaises(ValueError, animo.XYData,
                          np.array([0.0, 1.0]), np.array([1.0, 2.0, 4.0, 5.0]))
