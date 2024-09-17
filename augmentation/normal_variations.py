import numpy as np 

import cv2
from utils import Utils
from scipy.ndimage import gaussian_filter

import random
from scipy.ndimage import rotate,zoom

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
    
    def create_warp_ranges(self, num_points, variation_range):

        orig_points = [[50, 50], [200, 50], [50, 200], [200, 200]]
        transformed_points = []
        for point_ind in range(0, num_points):
            transformed_points.append([random.randint(orig_points[point_ind][0],\
            orig_points[point_ind][0] + variation_range),\
            random.randint(orig_points[point_ind][1], \
            orig_points[point_ind][1] + variation_range)])
        
        return orig_points, transformed_points

    def change_brain_contrast(self,contrast_factor) -> None:
        """
        Randomizes the contrast of the brain

        Parameters
        ----------------
        contrast_factor : float
            Determines magnitude of contrast change
        """

        mask = self.mask_image > 0
        brain_region = self.slice_image[mask]
        mean_intensity = np.mean(brain_region)
        self.slice_image[mask] = (self.slice_image[mask] - mean_intensity) \
        * contrast_factor + mean_intensity
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

    def change_skull_contrast(self,contrast_factor) -> None:
        """
        Randomizes the contrast of the skull

        Parameters
        ----------------
        contrast_factor : float
            Determines magnitude of contrast change

        """

        mask = (self.mask_image == 0) & (self.slice_image > 0.1)
        skull_region = self.slice_image[mask]
        mean_intensity = np.mean(skull_region)
        self.slice_image[mask] = (self.slice_image[mask] \
        - mean_intensity) * contrast_factor + mean_intensity
        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def convert_to_mp2rage(self):
              
        smaller_mask = self.utils.resize(self.mask_image, 0.9)
        self.slice_image[(self.slice_image > 0.1) & (self.mask_image < 0.05)] +=0.3
        self.change_brain_contrast(2)
        #self.slice_image[(self.slice_image < 0.1) & (self.mask_image > 0.05) & (smaller_mask < 0.05)] += 0.45
        self.slice_image[(self.slice_image < 0.1) & (self.mask_image > 0.05)] += random.uniform(0.1,0.5)

        #self.slice_image[(self.slice_image < 0.1) & (self.mask_image > 0.05)] *= random.uniform(1.5,3.5)
        self.change_skull_contrast(2)
        
        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = random.uniform(0.3, 0.9)
                min_pixel = random.uniform(0.1, max_pixel)
                if  (self.slice_image[row][pixel] > 0.2 and smaller_mask[row][pixel] <  0.05)\
                or ( smaller_mask[row][pixel] > 0.05):
                    continue
                else:
                    if self.mask_image[row][pixel] > 0.05:
                        min_pixel = 0
                        max_pixel *= 0.5
                    self.slice_image[row][pixel] += random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                max_pixel = random.uniform(0.3, 0.9)
                min_pixel = random.uniform(0.1, max_pixel)
                if (self.slice_image[row][pixel] > 0.2 and self.mask_image[row][pixel] <  0.05) \
                or (smaller_mask[row][pixel] > 0.05):
                    continue
                else:
                    if self.mask_image[row][pixel] > 0.05:
                        min_pixel = 0
                        max_pixel *= 0.3
                    self.slice_image[row][pixel]  += random.uniform(min_pixel , max_pixel)

    def blur_image(self):
        blur_size = (2 * random.randint(1, 4)) + 1
        self.slice_image = cv2.GaussianBlur(self.slice_image, (blur_size, blur_size), 0)
