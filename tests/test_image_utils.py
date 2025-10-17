import numpy as np
import numpy.random as npr
import numpy.testing as nptest
import unittest
import random

from src.utils.image_utils import *

DOUBLE_EPS = 1.0e-14
DOUBLE_EPS_SQ = 1.0e-28
FLOAT_EPS = 1.0e-7
FLOAT_EPS_SQ = 1.0e-14

class TestImageUtils(unittest.TestCase):

    def test_homogeneous_coordinates(self):
        v = np.full(shape=(10,2), fill_value=10)
        expected = np.full(shape=(10,3), fill_value=10, dtype=np.float32)
        expected[:,2] = 1 
        result = homogeneous_coordinates(v)
        self.assertTrue( np.allclose(expected, result), msg='Result should be float32 \
                         and be filled with 1s in [:,2]')
        
    def test_convert_white_rgb_to_gray(self):
        img = np.full(shape=(10,10,3), fill_value=255)
        expected = np.full(shape=(10,10), fill_value=254) #TODO: this is a problem with rounding integer back down
        result = convert_to_gray(img)
        self.assertTrue( np.allclose(expected, result),
                               msg='This test is of a white image, \
                                it should be all 1s')
    
    def test_convert_white_rgb_float_to_gray(self):
        img = np.full(shape=(10,10,3), fill_value=1.)
        expected = np.full(shape=(10,10), fill_value=1.) 
        result = convert_to_gray(img)
        self.assertTrue( np.allclose(expected, result, rtol=1e-3),
                               msg='This test is of a white image, \
                                it should be all 1s')
    def test_convert_already_gray(self):
        img = np.full(shape=(10,10), fill_value=1.)
        expected = img
        result = convert_to_gray(img)
        self.assertTrue( np.allclose(expected, result),
                               msg='This test is of a white image already gray,\
                                  it should be all 1s')
    
    def test_convert_error_weird_shape(self):
        self.assertRaises(ValueError, convert_to_gray, np.random.rand(10,10,10))
        self.assertRaises(ValueError, convert_to_gray, np.random.rand(10))
    
    def test_replace_with_nearest_equal_int(self):
        v = np.full(shape=(5,5), fill_value=1)
        v[2,2] = 0
        condition = '='
        value = 0
        v = replace_with_nearest(v, condition, value)
        result = v
        expected = np.full(shape=(5,5), fill_value=1)
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Replaced result should be filled with 1')
    
    def test_replace_with_nearest_less_than_float(self):
        v = np.full(shape=(5,5), fill_value=1.)
        condition = '<'
        v[2,2] = -1.
        value = 0.
        v = replace_with_nearest(v, condition, value)
        result = v
        expected = np.full(shape=(5,5), fill_value=1.)
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Replaced result should be filled with 1.')
        
    def test_normalize_color_no_ambient(self):
        color = np.full(shape=(5,5), fill_value=1)
        white = np.full_like(color, fill_value=2)
        normalized = normalize_color(color, white)
        result = normalized
        expected = np.full(shape=(5,5), fill_value=.5)
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Normalized result should be filled with .5')
    
    def test_normalize_color_with_ambient(self):
        color = np.full(shape=(5,5), fill_value=1)
        white = np.full_like(color, fill_value=2)
        ambient = np.full(shape=(5,5), fill_value=.5)
        normalized = normalize_color(color, white, black_image=ambient)
        result = normalized
        expected = np.full(shape=(5,5), fill_value=1/3)
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0, places=6,
                               msg='Normalized result should be filled with .33')
    
    def test_crop_empty_roi(self):
        image = np.random.rand(10,10)
        roi = ()
        result = crop(image=image, roi=roi)
        expected = image
        self.assertTrue(id(expected) == id(result),
                        msg='Cropping image with empty roi should do NOTHING')
    
    def test_crop(self):
        image = np.random.rand(10,10)
        x1,y1,x2,y2 = 1, 2, 6, 5 
        roi = (x1,y1,x2,y2)
        result = crop(image=image, roi=roi)
        expected = image[y1:y2,x1:x2]
        self.assertAlmostEqual(np.linalg.norm(result - expected), 0.0,
                               msg='Crop function should match array slicing')

if __name__ == '__main__':
    unittest.main()