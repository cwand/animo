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


class TestImageDecayCorrection(unittest.TestCase):

    def test_decay_correct(self):
        f = open(os.path.join('test', 'xml_input', 'decay_corr.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']

        oi = animo.load_image_series_from_file(
            os.path.join('test', 'data', '8_3V'), tags=['0008|0022', '0008|0032'])
        # Acquisition time: 133028

        ref = animo.load_image_series_from_file(
            os.path.join('test', 'data', 'static'), tags=['0008|0022', '0008|0032'])
        # Acquisition time: 124241
        # Seconds between: 2867

        no: dict[str, Any] = {'A': oi, 'B': ref}
        self.assertEqual(oi.get_no_frames(), 9)
        self.assertEqual(oi.get_matrix_size(), (64, 128, 128, ))
        self.assertAlmostEqual(float(oi.voxel_data[0, 7, 64, 56]), 125.77615, places=5)
        self.assertAlmostEqual(float(oi.voxel_data[3, 46, 59, 70]), 471.94589, places=5)
        self.assertAlmostEqual(float(oi.voxel_data[8, 54, 68, 67]), 187.133247, places=5)
        animo.image_decay_correction(task, no)
        self.assertAlmostEqual(float(oi.voxel_data[0, 7, 64, 56]), 137.8827, places=4)
        self.assertAlmostEqual(float(oi.voxel_data[3, 46, 59, 70]), 517.3730, places=4)
        self.assertAlmostEqual(float(oi.voxel_data[8, 54, 68, 67]), 205.1458, places=4)


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
        self.assertIsInstance(no['8_3V_T'], np.ndarray)
        self.assertIsInstance(no['8_3V_TAC'], np.ndarray)
        t = no['8_3V_T']
        tac = no['8_3V_TAC']
        self.assertEqual(t.shape, (9,))
        self.assertEqual(tac.shape, (9,))
        self.assertEqual(t[0], 0.0)
        self.assertEqual(t[1], 3.0)
        self.assertEqual(t[2], 6.3)
        self.assertEqual(t[3], 9.5)
        self.assertEqual(t[4], 12.8)
        self.assertEqual(t[5], 16.0)
        self.assertEqual(t[6], 19.3)
        self.assertEqual(t[7], 22.5)
        self.assertEqual(t[8], 25.8)
        self.assertAlmostEqual(float(tac[0]), 2062.73466, 1)
        self.assertAlmostEqual(float(tac[1]), 1553329.386, 0)
        self.assertAlmostEqual(float(tac[2]) / 1000.0, 18607.9194, 0)
        self.assertAlmostEqual(float(tac[3]) / 1000.0, 48918.2085, 1)
        self.assertAlmostEqual(float(tac[4]) / 1000.0, 41655.6189, 1)
        self.assertAlmostEqual(float(tac[5]) / 10000.0, 1051.76287, 0)
        self.assertAlmostEqual(float(tac[6]), 731450.412, 0)
        self.assertAlmostEqual(float(tac[7]), 1653.982659, 3)
        self.assertAlmostEqual(float(tac[8]), 66.6683136, 4)


class TestMean(unittest.TestCase):

    def test_mean_0(self):
        f = open(os.path.join('test', 'xml_input', 'average.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'x': np.array([0.0, 0.0, 0.0]), }
        animo.average(task, no)
        self.assertIsInstance(no['mu'], float)
        self.assertEqual(no['mu'], 0.0)
        self.assertIsInstance(no['s2'], float)
        self.assertEqual(no['s2'], 0.0)

    def test_mean_2d(self):
        f = open(os.path.join('test', 'xml_input', 'average.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'x': np.array([[0.0, 1.0, 1.5], [2.0, -1.0, 0.1]]), }
        animo.average(task, no)
        self.assertIsInstance(no['mu'], float)
        self.assertEqual(no['mu'], 0.6)
        self.assertIsInstance(no['s2'], float)
        self.assertAlmostEqual(no['s2'], 1.01666667, places=8)


class TestIntXY(unittest.TestCase):

    def test_int_0(self):
        f = open(os.path.join('test', 'xml_input', 'intxy.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'X': np.array([0.0, 1.0]), 'Y': np.array([0.0, 0.0])}
        animo.integrate(task, no)
        self.assertIsInstance(no['integral'], float)
        self.assertEqual(no['integral'], 0.0)

    def test_int_triangle(self):
        f = open(os.path.join('test', 'xml_input', 'intxy.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'X': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                              'Y': np.array([0.0, 0.0, 1.0, 0.0, 0.0])}
        animo.integrate(task, no)
        self.assertIsInstance(no['integral'], float)
        self.assertEqual(no['integral'], 1.0)


class TestEval(unittest.TestCase):

    def test_eval_simple(self):
        f = open(os.path.join('test', 'xml_input', 'eval.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'A': 3.0, 'B': 1.5, 'C': -7.5, }
        animo.eval_expr(task, no)
        self.assertIsInstance(no['D'], float)
        self.assertEqual(no['D'], 0.5)

    def test_eval_exp(self):
        f = open(os.path.join('test', 'xml_input', 'eval2.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {'A': 300, 'B': 4000, }
        animo.eval_expr(task, no)
        self.assertIsInstance(no['C'], float)
        self.assertAlmostEqual(no['C'], 341.0396, places=4)
