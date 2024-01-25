import os.path
import unittest
import pytest
import animo
import numpy as np


@pytest.fixture(autouse=True)
def cleanup():
    to_delete = [os.path.join('test', 'data.txt')]
    yield to_delete
    for item in to_delete:
        if os.path.exists(item):
            os.remove(item)


class TestWriteData(unittest.TestCase):

    def test_write_one_col(self):
        x = np.array([1.0, 2.0, 2.5])
        animo.write_data((x, ), ('X [m]', ), os.path.join('test', 'data.txt'))
        with open(os.path.join('test', 'data.txt')) as f:
            lines = f.readlines()
            self.assertEqual(4, len(lines))
            self.assertEqual("# X [m]", lines[0].strip())
            self.assertEqual("1.000000000000000000e+00", lines[1].strip())
            self.assertEqual("2.000000000000000000e+00", lines[2].strip())
            self.assertEqual("2.500000000000000000e+00", lines[3].strip())

    def test_write_three_col(self):
        x = np.array([1.0, 2.0, 2.5, 5.6])
        y = np.array([-1.00056, 2000.000001, 15678654543.1, -1])
        z = np.array([0.00011111, 0, 13, -14.007])
        animo.write_data((x, y, z), ('X [m]', 'Y', 'Z = m^2 [meteres]'),
                         os.path.join('test', 'data.txt'))
        with open(os.path.join('test', 'data.txt')) as f:
            lines = f.readlines()
            self.assertEqual(5, len(lines))
            self.assertEqual("# X [m]\tY\tZ = m^2 [meteres]", lines[0].strip())
        t = np.loadtxt(os.path.join('test', 'data.txt'), delimiter=',')
        self.assertEqual(t.shape, (4, 3))
        self.assertAlmostEqual(float(t[0, 0]), 1.0, places=18)
        self.assertAlmostEqual(float(t[1, 0]), 2.0, places=18)
        self.assertAlmostEqual(float(t[2, 0]), 2.5, places=18)
        self.assertAlmostEqual(float(t[3, 0]), 5.6, places=18)
        self.assertAlmostEqual(float(t[0, 1]), -1.00056, places=18)
        self.assertAlmostEqual(float(t[1, 1]), 2000.000001, places=18)
        self.assertAlmostEqual(float(t[2, 1]), 15678654543.1, places=18)
        self.assertAlmostEqual(float(t[3, 1]), -1.0, places=18)
        self.assertAlmostEqual(float(t[0, 2]), 0.00011111, places=18)
        self.assertAlmostEqual(float(t[1, 2]), 0.0, places=18)
        self.assertAlmostEqual(float(t[2, 2]), 13.0, places=18)
        self.assertAlmostEqual(float(t[3, 2]), -14.007, places=18)

    def test_unequal_length(self):
        x = np.array([1.0, 2.0])
        y = np.array([-1.00056, 2000.000001, 15678654543.1])
        self.assertRaises(ValueError, animo.write_data, (x, y, ), ('X [m]', 'Y', ),
                          os.path.join('test', 'data.txt'))

    def test_unequal_columns_and_headers(self):
        x = np.array([1.0, 2.0])
        y = np.array([-1.00056, 2000.000001])
        self.assertRaises(ValueError, animo.write_data, (x, y, ), ('X [m]', ),
                          os.path.join('test', 'data.txt'))
