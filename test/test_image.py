import unittest
import animo
import numpy as np
import os


class TestImageData(unittest.TestCase):

    def test_init(self):
        id = animo.ImageData(np.array([[0.0, 1.0], [1.5, 2.0]]), {'key1': ['X'], 'key2': [2]})
        self.assertEqual(id.voxel_data.shape, (2, 2))
        self.assertEqual(id.voxel_data[0, 0], 0.0)
        self.assertEqual(id.voxel_data[0, 1], 1.0)
        self.assertEqual(id.voxel_data[1, 0], 1.5)
        self.assertEqual(id.voxel_data[1, 1], 2.0)
        self.assertEqual(len(id.meta_data.keys()), 2)
        self.assertEqual(id.meta_data['key1'], ['X'])
        self.assertEqual(id.meta_data['key2'], [2])


class TestLoadImageFromFile(unittest.TestCase):

    def test_load_single_image_no_meta(self):
        id = animo.load_image_from_file(os.path.join('test', 'data', 'nema'))
        self.assertEqual(np.max(id.voxel_data), 1864120.97838)
        self.assertEqual(id.voxel_data.shape, (1, 256, 256, 256))
        self.assertEqual(id.voxel_data[0, 140, 101, 140], 1864120.97838)
        self.assertEqual(id.meta_data, {})

    def test_load_single_image_1_slice_1_meta(self):
        id = animo.load_image_from_file(os.path.join('test', 'data', 'nema'), ['0008|0022'])
        self.assertEqual(id.voxel_data.shape, (1, 256, 256, 256))
        self.assertEqual(id.meta_data, {'0008|0022': ['20230621']})

    def test_load_single_image_9_slice_2_meta(self):
        id = animo.load_image_from_file(os.path.join('test', 'data', '8_3V'),
                                        ['0008|0022', '0008|0032'])
        self.assertEqual(id.voxel_data.shape, (9, 64, 128, 128))
        self.assertEqual(id.meta_data, {'0008|0022': ['20231201', '20231201', '20231201',
                                                      '20231201', '20231201', '20231201',
                                                      '20231201', '20231201', '20231201'],
                                        '0008|0032': ['133028.0', '133031.0', '133034.3',
                                                      '133037.5', '133040.8', '133044.0',
                                                      '133047.3', '133050.5', '133053.8']})

    def test_wrong_meta(self):
        self.assertRaises(ValueError, animo.load_image_from_file,
                          os.path.join('test', 'data', 'nema'), tags=['0004|0022'])


