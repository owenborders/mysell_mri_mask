import numpy as np 

import cv2
from utils import Utils
from scipy.ndimage import gaussian_filter

import random

class NormalVariations():

    def __init__(self):
        self.utils = Utils()
    
    def warp_image(self,img, initial_points, transformed_points):
        """
        Warps image by transforming the 
        initial points into the transformed points.

        Parameters
        ---------------
        img : np.array
            Input image to be transforms
        initial_points : list
            Initial coordinates that will be moved
        transformed_pints : list
            Coordinates that the initial points arer being transformed into
        """
        rows, cols = img.shape[:2]

        pts1 = np.float32(initial_points)
        pts2 = np.float32(transformed_points)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        img = cv2.warpPerspective(img, M, (cols, rows))

        return img
        
    def convert_to_mp2rage(self):
        smaller_mask = self.resize(self.mask_image, 0.8)
        self.slice_image[(self.slice_image > 0.1) & (self.mask_image < 0.05)] +=0.2
        self.slice_image[(self.slice_image > 0.1) & (self.mask_image > 0.05)] +=0.18
        self.change_brain_contrast(2)
        self.change_skull_contrast(2)
        
        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = 0.7
                min_pixel = random.uniform(0.1, max_pixel)
                if self.slice_image[row][pixel] > 0.1 or self.mask_image[row][pixel] > 0.05:
                    continue
                else:
                    self.slice_image[row][pixel] += random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                max_pixel = 0.7
                min_pixel = random.uniform(0.1, max_pixel)
                if self.slice_image[row][pixel] > 0.1 or self.mask_image[row][pixel] > 0.05:
                    continue
                else:
                    self.slice_image[row][pixel]  += random.uniform(min_pixel , max_pixel)

        self.overlay_random_noise()
        self.slice_image = self.normalize(self.slice_image)
        
    def blur_image(self):
        blur_size = (2 * random.randint(1, 4)) + 1
        self.slice_image = cv2.GaussianBlur(self.slice_image, (blur_size, blur_size), 0)
