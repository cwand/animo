import unittest
import animo
import os


class TestExtractTACFrom01Labelmap(unittest.TestCase):

    def test_extract(self):
        image = animo.load_image_series_from_file(
            os.path.join('test', 'data', '8_3V'), ['0008|0022', '0008|0032'])
        roi = animo.load_image_from_file(os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))
        x, tac = animo.extract_tac_from_01labelmap(image, roi)
        self.assertEqual(x.shape, (9,))
        self.assertEqual(tac.shape, (9,))
        self.assertEqual(x[0], 0.0)
        self.assertEqual(x[1], 3.0)
        self.assertEqual(x[2], 6.3)
        self.assertEqual(x[3], 9.5)
        self.assertEqual(x[4], 12.8)
        self.assertEqual(x[5], 16.0)
        self.assertEqual(x[6], 19.3)
        self.assertEqual(x[7], 22.5)
        self.assertEqual(x[8], 25.8)
        self.assertAlmostEqual(float(tac[0]), 2062.73466, 1)
        self.assertAlmostEqual(float(tac[1]), 1553329.386, 0)
        self.assertAlmostEqual(float(tac[2]) / 1000.0, 18607.9194, 0)
        self.assertAlmostEqual(float(tac[3]) / 1000.0, 48918.2085, 1)
        self.assertAlmostEqual(float(tac[4]) / 1000.0, 41655.6189, 1)
        self.assertAlmostEqual(float(tac[5]) / 10000.0, 1051.76287, 0)
        self.assertAlmostEqual(float(tac[6]), 731450.412, 0)
        self.assertAlmostEqual(float(tac[7]), 1653.982659, 3)
        self.assertAlmostEqual(float(tac[8]), 66.6683136, 4)
