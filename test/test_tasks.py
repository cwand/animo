import unittest
import os
import animo
import xmltodict
from typing import Any
import numpy as np


class TestImageSeriesLoader(unittest.TestCase):

    def test_image_series_load(self):
        f = open(os.path.join('test', 'xml_input', 'image_series_load.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {}
        animo.image_series_loader(task, no)
        self.assertIsInstance(no['8_3V'], animo.ImageData)
        self.assertEqual(no['8_3V'].voxel_data.shape, (9, 64, 128, 128))
        self.assertEqual(no['8_3V'].meta_data,
                         {'0008|0022': ['20231201', '20231201', '20231201', '20231201', '20231201',
                                        '20231201', '20231201', '20231201', '20231201'],
                          '0008|0032': ['133028.0', '133031.0', '133034.3', '133037.5', '133040.8',
                                        '133044.0', '133047.3', '133050.5', '133053.8']})

    def test_image_series_load_no_meta(self):
        f = open(os.path.join('test', 'xml_input', 'image_series_load_no_meta.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {}
        animo.image_series_loader(task, no)
        self.assertIsInstance(no['8_3V'], animo.ImageData)
        self.assertEqual(no['8_3V'].voxel_data.shape, (9, 64, 128, 128))
        self.assertEqual(no['8_3V'].meta_data, {})


class TestImageLoader(unittest.TestCase):

    def test_image_load(self):
        f = open(os.path.join('test', 'xml_input', 'image_load.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {}
        animo.image_loader(task, no)
        self.assertIsInstance(no['Cyl101'], animo.ImageData)
        self.assertEqual(np.max(no['Cyl101'].voxel_data), 1)
        self.assertEqual(no['Cyl101'].voxel_data.shape, (64, 128, 128))
        self.assertEqual(np.sum(no['Cyl101'].voxel_data), 1701)
        self.assertEqual(no['Cyl101'].meta_data, {})


class TestTACFromLabelmap(unittest.TestCase):

    def test_image_load(self):
        f = open(os.path.join('test', 'xml_input', 'tac.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {
            '8_3V': animo.load_image_series_from_file(
                os.path.join('test', 'data', '8_3V'), tags=['0008|0022', '0008|0032']),
            'Cyl101': animo.load_image_from_file(
                os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))
        }
        animo.tac_from_labelmap(task, no)
        self.assertIsInstance(no['8_3V_TAC'], animo.XYData)
        tac = no['8_3V_TAC']
        self.assertEqual(tac.x.shape, (9,))
        self.assertEqual(tac.y.shape, (9,))
        self.assertEqual(tac.x[0], 0.0)
        self.assertEqual(tac.x[1], 3.0)
        self.assertEqual(tac.x[2], 6.3)
        self.assertEqual(tac.x[3], 9.5)
        self.assertEqual(tac.x[4], 12.8)
        self.assertEqual(tac.x[5], 16.0)
        self.assertEqual(tac.x[6], 19.3)
        self.assertEqual(tac.x[7], 22.5)
        self.assertEqual(tac.x[8], 25.8)
        self.assertAlmostEqual(float(tac.y[0]), 2062.73466, 1)
        self.assertAlmostEqual(float(tac.y[1]), 1553329.386, 0)
        self.assertAlmostEqual(float(tac.y[2]) / 1000.0, 18607.9194, 0)
        self.assertAlmostEqual(float(tac.y[3]) / 1000.0, 48918.2085, 1)
        self.assertAlmostEqual(float(tac.y[4]) / 1000.0, 41655.6189, 1)
        self.assertAlmostEqual(float(tac.y[5]) / 10000.0, 1051.76287, 0)
        self.assertAlmostEqual(float(tac.y[6]), 731450.412, 0)
        self.assertAlmostEqual(float(tac.y[7]), 1653.982659, 3)
        self.assertAlmostEqual(float(tac.y[8]), 66.6683136, 4)