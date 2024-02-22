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

    def test_get_frame_matrix_size_simple(self):
        id = animo.ImageData(np.array([[0.0, 1.0], [1.5, 2.0]]), {'key1': ['X'], 'key2': ['2']})
        self.assertEqual(id.get_no_frames(), 1)
        self.assertEqual(id.get_matrix_size(), (2, 2))

    def test_decay_correction(self):
        id1 = animo.ImageData(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            {'0008|0022': ['20240119'], '0008|0032': ['130000.0']})
        id2 = animo.ImageData(
            np.array([[2.0, 4.0], [6.0, 8.0]]),
            {'0008|0022': ['20240119'], '0008|0032': ['140000.0']})
        id1.decay_correction(id2, 3600.0)
        self.assertEqual(id1.voxel_data[0, 0], 5.0)
        self.assertEqual(id1.voxel_data[0, 1], 10.0)
        self.assertEqual(id1.voxel_data[1, 0], 15.0)
        self.assertEqual(id1.voxel_data[1, 1], 20.0)
        self.assertEqual(id2.voxel_data[0, 0], 2.0)
        self.assertEqual(id2.voxel_data[0, 1], 4.0)
        self.assertEqual(id2.voxel_data[1, 0], 6.0)
        self.assertEqual(id2.voxel_data[1, 1], 8.0)
        id2.decay_correction(id1, 3600.0)
        self.assertEqual(id1.voxel_data[0, 0], 5.0)
        self.assertEqual(id1.voxel_data[0, 1], 10.0)
        self.assertEqual(id1.voxel_data[1, 0], 15.0)
        self.assertEqual(id1.voxel_data[1, 1], 20.0)
        self.assertEqual(id2.voxel_data[0, 0], 4.0)
        self.assertEqual(id2.voxel_data[0, 1], 8.0)
        self.assertEqual(id2.voxel_data[1, 0], 12.0)
        self.assertEqual(id2.voxel_data[1, 1], 16.0)

    def test_decay_correction_frames(self):
        id1 = animo.ImageData(
            np.array([[[[10.0, 20.0], [30.0, 40.0]]], [[[5.0, 10.0], [20.0, 30.0]]]]),
            {'0008|0022': ['20240119', '20240119', ], '0008|0032': ['130000.0', '133000.0', ]})
        id2 = animo.ImageData(
            np.array([[[[2.0, 4.0], [6.0, 8.0]]], [[[1.0, 2.0], [4.0, 6.0]]]]),
            {'0008|0022': ['20240119', '20240119', ], '0008|0032': ['140000.0', '150000.0', ]})
        id1.decay_correction(id2, 1800.0)
        self.assertEqual(id1.voxel_data[0, 0, 0, 0, ], 2.5)
        self.assertEqual(id1.voxel_data[0, 0, 0, 1, ], 5.0)
        self.assertEqual(id1.voxel_data[0, 0, 1, 0, ], 7.5)
        self.assertEqual(id1.voxel_data[0, 0, 1, 1, ], 10.0)

    def test_decay_correction_missing_meta(self):
        id1 = animo.ImageData(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            {'0008|0022': ['20240119'], '0008|0032': ['130000.0']})
        id2 = animo.ImageData(
            np.array([[2.0, 4.0], [6.0, 8.0]]), {})
        self.assertRaises(KeyError, id1.decay_correction, id2, 3600.0)


class TestLoadImageSeriesFromFile(unittest.TestCase):

    def test_load_single_image_no_meta(self):
        id = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'))
        self.assertEqual(np.max(id.voxel_data), 1864120.97838)
        self.assertEqual(id.voxel_data.shape, (1, 256, 256, 256))
        self.assertEqual(id.get_no_frames(), 1)
        self.assertEqual(id.get_matrix_size(), (256, 256, 256))
        self.assertEqual(id.voxel_data[0, 140, 101, 140], 1864120.97838)
        self.assertEqual(id.meta_data, {})

    def test_load_single_image_1_slice_1_meta(self):
        id = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'), ['0008|0022'])
        self.assertEqual(id.get_no_frames(), 1)
        self.assertEqual(id.get_matrix_size(), (256, 256, 256))
        self.assertEqual(id.meta_data, {'0008|0022': ['20230621']})

    def test_load_single_image_9_slice_2_meta(self):
        id = animo.load_image_series_from_file(
            os.path.join('test', 'data', '8_3V'), ['0008|0022', '0008|0032'])
        self.assertEqual(id.voxel_data.shape, (9, 64, 128, 128))
        self.assertEqual(id.get_no_frames(), 9)
        self.assertEqual(id.get_matrix_size(), (64, 128, 128))
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
        self.assertEqual(id.get_no_frames(), 1)
        self.assertEqual(id.get_matrix_size(), (64, 128, 128))
        self.assertEqual(np.sum(id.voxel_data), 1701)
        self.assertEqual(id.meta_data, {})


class TestGetAcqDatetime(unittest.TestCase):

    def test_acq_date_time_1_image(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'),
                                              tags=['0008|0022', '0008|0032'])
        ds = x.get_acq_datetime()
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0], datetime(2023, 6, 21, 10, 51, 44, 0))

    def test_acq_date_time_9_images(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                              tags=['0008|0022', '0008|0032'])
        ds = x.get_acq_datetime()
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
        self.assertRaises(KeyError, x.get_acq_datetime)

    def test_acq_date_time_missing_meta_32(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'),
                                              tags=['0008|0022', '0008|0031'])
        self.assertRaises(KeyError, x.get_acq_datetime)

    def test_acq_date_time_missing_meta(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', 'nema'))
        self.assertRaises(KeyError, x.get_acq_datetime)


class TestGetAcqDuration(unittest.TestCase):

    def test_acq_duration_9_images(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                              tags=['0018|1242'])
        ds = x.get_acq_duration()
        self.assertEqual(len(ds), 9)
        self.assertEqual(ds[0], 3.04)
        self.assertEqual(ds[1], 3.26)
        self.assertEqual(ds[2], 3.26)
        self.assertEqual(ds[3], 3.26)
        self.assertEqual(ds[4], 3.25)
        self.assertEqual(ds[5], 3.26)
        self.assertEqual(ds[6], 3.25)
        self.assertEqual(ds[7], 3.26)
        self.assertEqual(ds[8], 3.26)

    def test_acq_date_time_missing_meta(self):
        x = animo.load_image_series_from_file(os.path.join('test', 'data', '8_3V'),
                                              tags=['0008|0021', '0008|0032'])
        self.assertRaises(KeyError, x.get_acq_duration)
