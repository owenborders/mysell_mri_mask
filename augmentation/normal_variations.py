import numpy as np 

import cv2
from utils import Utils
from scipy.ndimage import gaussian_filter


class NormalVariations():

    def __init__(self):
        self.utils = Utils()

    def elastic_transform(self,image, alpha, sigma):
        random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), \
        sigma, mode='constant', cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), \
        sigma, mode='constant', cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return distorted_image
    
    
