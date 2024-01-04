import unittest
import animo
import numpy as np
import os


class TestExtractTACFrom01Labelmap(unittest.TestCase):

    def test_extract(self):
        image = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                               ['0008|0022', '0008|0032'])
        roi = animo.load_image_from_file(os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))
        tac = animo.extract_tac_from_01labelmap(image, roi)
        self.assertEqual(tac.x.shape, (9, 1))
        self.assertEqual(tac.y.shape, (9, 1))
        self.assertEqual(tac.x[0], 0.0)
        self.assertEqual(tac.x[1], 3.0)
        self.assertEqual(tac.x[2], 6.0)
        self.assertEqual(tac.x[3], 9.0)
        self.assertEqual(tac.x[4], 12.0)
        self.assertEqual(tac.x[5], 15.0)
        self.assertEqual(tac.x[6], 18.0)
        self.assertEqual(tac.x[7], 21.0)
        self.assertEqual(tac.x[8], 24.0)
