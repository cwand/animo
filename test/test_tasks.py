import unittest
import os
import pytest
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

    def test_image_series_load_decay(self):
        f = open(os.path.join('test', 'xml_input', 'image_series_load_decay.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']

        ref = animo.load_image_series_from_file(
            os.path.join('test', 'data', 'static'), tags=['0008|0022', '0008|0032'])
        no: dict[str, Any] = {'static': ref}

        animo.image_series_loader(task, no)
        self.assertIsInstance(no['8_3V'], animo.ImageData)

        self.assertEqual(no['8_3V'].get_no_frames(), 9)
        self.assertEqual(no['8_3V'].get_matrix_size(), (64, 128, 128,))
        # self.assertAlmostEqual(float(no['8_3V'].voxel_data[0, 7, 64, 56]), 125.77615, places=5)
        # self.assertAlmostEqual(float(no['8_3V'].voxel_data[3, 46, 59, 70]), 471.94589, places=5)
        # self.assertAlmostEqual(float(no['8_3V'].voxel_data[8, 54, 68, 67]), 187.133247, places=5)
        # Seconds between: 2867
        self.assertAlmostEqual(float(no['8_3V'].voxel_data[0, 7, 64, 56]), 137.8827, places=4)
        self.assertAlmostEqual(float(no['8_3V'].voxel_data[3, 46, 59, 70]), 517.3730, places=4)
        self.assertAlmostEqual(float(no['8_3V'].voxel_data[8, 54, 68, 67]), 205.1458, places=4)


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
                os.path.join('test', 'data', '8_3V'), tags=['0008|0022', '0008|0032', '0018|1242']),
            'Cyl101': animo.load_image_from_file(
                os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))
        }
        animo.tac_from_labelmap(task, no)
        self.assertIsInstance(no['8_3V_T'], np.ndarray)
        self.assertIsInstance(no['8_3V_D'], np.ndarray)
        self.assertIsInstance(no['8_3V_TAC'], np.ndarray)
        t = no['8_3V_T']
        d = no['8_3V_D']
        tac = no['8_3V_TAC']
        self.assertEqual(t.shape, (9,))
        self.assertEqual(d.shape, (9,))
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
        self.assertEqual(d[0], 3.04)
        self.assertEqual(d[1], 3.26)
        self.assertEqual(d[2], 3.26)
        self.assertEqual(d[3], 3.26)
        self.assertEqual(d[4], 3.25)
        self.assertEqual(d[5], 3.26)
        self.assertEqual(d[6], 3.25)
        self.assertEqual(d[7], 3.26)
        self.assertEqual(d[8], 3.26)
        self.assertAlmostEqual(float(tac[0]), 2062.73466, 1)
        self.assertAlmostEqual(float(tac[1]), 1553329.386, 0)
        self.assertAlmostEqual(float(tac[2]) / 1000.0, 18607.9194, 0)
        self.assertAlmostEqual(float(tac[3]) / 1000.0, 48918.2085, 1)
        self.assertAlmostEqual(float(tac[4]) / 1000.0, 41655.6189, 1)
        self.assertAlmostEqual(float(tac[5]) / 10000.0, 1051.76287, 0)
        self.assertAlmostEqual(float(tac[6]), 731450.412, 0)
        self.assertAlmostEqual(float(tac[7]), 1653.982659, 3)
        self.assertAlmostEqual(float(tac[8]), 66.6683136, 4)


class TestTACIntegrate(unittest.TestCase):

    def test_integrate(self):
        f = open(os.path.join('test', 'xml_input', 'tac_integral.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {
            'A': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'B': np.array([0.0, 1.0, 1.0, 2.0, 0.0]),
        }
        animo.tac_integrate(task, no)
        self.assertIsInstance(no['C'], float)
        self.assertEqual(no['C'], 4.0)


class TestMeanVarCalc(unittest.TestCase):

    def test_mean_var(self):
        f = open(os.path.join('test', 'xml_input', 'mean_var.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True)
        task = tree['animo']['task']
        no: dict[str, Any] = {
            'A': np.array([1.2, 2.1, 3.0, 2.4]),
        }
        animo.mean_var_calc(task, no)
        self.assertIsInstance(no['mu'], float)
        self.assertEqual(no['mu'], 2.175)
        self.assertIsInstance(no['var'], float)
        self.assertEqual(no['var'], 0.5625)


@pytest.fixture(autouse=True)
def cleanup():
    to_delete = [os.path.join('test', 'test_write_task.txt')]
    yield to_delete
    for item in to_delete:
        if os.path.exists(item):
            os.remove(item)


class TestWriter(unittest.TestCase):

    def test_write_one_col(self):
        f = open(os.path.join('test', 'xml_input', 'writer1.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True, force_list=('col', ))
        task = tree['animo']['task']
        no: dict[str, Any] = {
            'x': np.array([1.0, 2.0, 2.5]),
        }
        animo.writer(task, no)
        with open(os.path.join('test', 'test_write_task.txt')) as f:
            lines = f.readlines()
            self.assertEqual(4, len(lines))
            self.assertEqual("# A", lines[0].strip())
            self.assertEqual("1.000000000000000000e+00", lines[1].strip())
            self.assertEqual("2.000000000000000000e+00", lines[2].strip())
            self.assertEqual("2.500000000000000000e+00", lines[3].strip())

    def test_write_two_col(self):
        f = open(os.path.join('test', 'xml_input', 'writer2.xml'))
        tree = xmltodict.parse(f.read(), xml_attribs=True, force_list=('col', ))
        task = tree['animo']['task']
        no: dict[str, Any] = {
            'x': np.array([1.0, 2.0, 2.5, 5.6]),
            'y': np.array([-1.00056, 2000.000001, 15678654543.1, -1]),
        }
        animo.writer(task, no)
        with open(os.path.join('test', 'test_write_task.txt')) as f:
            lines = f.readlines()
            self.assertEqual(5, len(lines))
            self.assertEqual("# A\tB [m/s]", lines[0].strip())
        t = np.loadtxt(os.path.join('test', 'test_write_task.txt'), delimiter=',')
        self.assertEqual(t.shape, (4, 2))
        self.assertAlmostEqual(float(t[0, 0]), 1.0, places=18)
        self.assertAlmostEqual(float(t[1, 0]), 2.0, places=18)
        self.assertAlmostEqual(float(t[2, 0]), 2.5, places=18)
        self.assertAlmostEqual(float(t[3, 0]), 5.6, places=18)
        self.assertAlmostEqual(float(t[0, 1]), -1.00056, places=18)
        self.assertAlmostEqual(float(t[1, 1]), 2000.000001, places=18)
        self.assertAlmostEqual(float(t[2, 1]), 15678654543.1, places=18)
        self.assertAlmostEqual(float(t[3, 1]), -1.0, places=18)
