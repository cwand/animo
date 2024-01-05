import unittest
import animo
import numpy as np
import os
from datetime import datetime


class TestImageData(unittest.TestCase):

    def test_init(self):
        id = animo.ImageData(np.array([[0.0, 1.0], [1.5, 2.0]]), {'key1': ['X'], 'key2': ['2']})
        self.assertEqual(id.voxel_data.shape, (2, 2))
        self.assertEqual(id.voxel_data[0, 0], 0.0)
        self.assertEqual(id.voxel_data[0, 1], 1.0)
        self.assertEqual(id.voxel_data[1, 0], 1.5)
        self.assertEqual(id.voxel_data[1, 1], 2.0)
        self.assertEqual(len(id.meta_data.keys()), 2)
        self.assertEqual(id.meta_data['key1'], ['X'])
        self.assertEqual(id.meta_data['key2'], ['2'])


class TestLoadImageSeriesFromFile(unittest.TestCase):

    def test_load_single_image_no_meta(self):
        id = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'))
        self.assertEqual(np.max(id.voxel_data), 1864120.97838)
        self.assertEqual(id.voxel_data.shape, (1, 256, 256, 256))
        self.assertEqual(id.voxel_data[0, 140, 101, 140], 1864120.97838)
        self.assertEqual(id.meta_data, {})

    def test_load_single_image_1_slice_1_meta(self):
        id = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'), ['0008|0022'])
        self.assertEqual(id.voxel_data.shape, (1, 256, 256, 256))
        self.assertEqual(id.meta_data, {'0008|0022': ['20230621']})

    def test_load_single_image_9_slice_2_meta(self):
        id = animo.load_image_series_from_file(
            os.path.join('test', 'data', '8_3V'), ['0008|0022', '0008|0032'])
        self.assertEqual(id.voxel_data.shape, (9, 64, 128, 128))
        self.assertEqual(id.meta_data, {'0008|0022': ['20231201', '20231201', '20231201',
                                                      '20231201', '20231201', '20231201',
                                                      '20231201', '20231201', '20231201'],
                                        '0008|0032': ['133028.0', '133031.0', '133034.3',
                                                      '133037.5', '133040.8', '133044.0',
                                                      '133047.3', '133050.5', '133053.8']})

    def test_wrong_meta(self):
        self.assertRaises(ValueError, animo.load_image_series_from_file,
                          os.path.join('test', 'data', 'nema'), tags=['0004|0022'])


class TestLoadImageFromFile(unittest.TestCase):

    def test_load_image(self):
        id = animo.load_image_from_file(os.path.join('test', 'data', 'segs', 'Cyl101.nrrd'))
        self.assertEqual(np.max(id.voxel_data), 1)
        self.assertEqual(id.voxel_data.shape, (64, 128, 128))
        self.assertEqual(np.sum(id.voxel_data), 1701)
        self.assertEqual(id.meta_data, {})


class TestGetAcqDatetime(unittest.TestCase):

    def test_acq_date_time_1_image(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'),
                                              tags=['0008|0022', '0008|0032'])
        ds = animo.get_acq_datetime(x)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0], datetime(2023, 6, 21, 10, 51, 44, 0))

    def test_acq_date_time_9_images(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                              tags=['0008|0022', '0008|0032'])
        ds = animo.get_acq_datetime(x)
        self.assertEqual(len(ds), 9)
        self.assertEqual(ds[0], datetime(2023, 12, 1, 13, 30, 28, 0))
        self.assertEqual(ds[1], datetime(2023, 12, 1, 13, 30, 31, 0))
        self.assertEqual(ds[2], datetime(2023, 12, 1, 13, 30, 34, 300000))
        self.assertEqual(ds[3], datetime(2023, 12, 1, 13, 30, 37, 500000))
        self.assertEqual(ds[4], datetime(2023, 12, 1, 13, 30, 40, 800000))
        self.assertEqual(ds[5], datetime(2023, 12, 1, 13, 30, 44, 0))
        self.assertEqual(ds[6], datetime(2023, 12, 1, 13, 30, 47, 300000))
        self.assertEqual(ds[7], datetime(2023, 12, 1, 13, 30, 50, 500000))
        self.assertEqual(ds[8], datetime(2023, 12, 1, 13, 30, 53, 800000))

    def test_acq_date_time_missing_meta_22(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'),
                                              tags=['0008|0021', '0008|0032'])
        self.assertRaises(KeyError, animo.get_acq_datetime, x)

    def test_acq_date_time_missing_meta_32(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'),
                                              tags=['0008|0022', '0008|0031'])
        self.assertRaises(KeyError, animo.get_acq_datetime, x)

    def test_acq_date_time_missing_meta(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'))
        self.assertRaises(KeyError, animo.get_acq_datetime, x)
