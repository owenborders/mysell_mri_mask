import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate,zoom
import cv2
import scipy.ndimage
from skimage import transform, data
from scipy.ndimage import gaussian_filter
import copy
from skimage.metrics import structural_similarity as compare_ssim
from scipy.ndimage import convolve
from scipy.stats import vonmises

from utils import Utils

class KnownArtifacts():
      def __init__(self):
        self.utils = Utils()
      
      def simulate_ghosting_artifact(self, img):
        """
        Mirrors the MRI image
        and increases its transparency
        to simulate ghosting.
        """

        height, width = img.shape[:2]
        num_copies = random.randint(1, 3)

        for _ in range(num_copies):
            mirrored_copy = cv2.flip(img, flipCode=random.choice([-1, 0, 1]))
            alpha = random.uniform(0.1, 0.2)
            overlay = np.zeros_like(img)
            x_offset = random.randint(0, width - mirrored_copy.shape[1])
            y_offset = random.randint(0, height - mirrored_copy.shape[0])
            overlay[y_offset:y_offset+mirrored_copy.shape[0], \
            x_offset:x_offset+mirrored_copy.shape[1]] = mirrored_copy
            img = cv2.addWeighted(img, 1, overlay, alpha, 0)

        np.clip(img, 0, 1, out=img)

        return img

      def create_ring(
          self,scan_img, mask_img,x_threshold, y_threshold,
          pos_or_neg_x, pos_or_neg_y,brightness_factors,
          ring_distances, mask_cutoff_factor
      ):
        partial = 1
        def resize(img,scale):
            resized_image = zoom(img, (scale, scale),order=1)
            background = np.zeros((256, 256), dtype=img.dtype)
            start_x = max((256 - resized_image.shape[1]) // 2, 0)
            start_y = max((256 - resized_image.shape[0]) // 2, 0)
            resized_image_clipped = resized_image[:min(256, resized_image.shape[0]),
                                                  :min(256, resized_image.shape[1])]

            background[start_y:start_y+resized_image_clipped.shape[0], \
                        start_x:start_x+resized_image_clipped.shape[1]] = resized_image_clipped

            return background

        for ring_size_diff in range(0,len(brightness_factors)):
          brightness_factor = brightness_factors[ring_size_diff]
          scale = ring_distances[ring_size_diff]


          resized_mask = resize(scan_img, scale)
          sobel_x = cv2.Sobel(resized_mask, cv2.CV_64F, 1, 0, ksize=1)
          sobel_y = cv2.Sobel(resized_mask, cv2.CV_64F, 0, 1, ksize=1)

          gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
          gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
          _, edges = cv2.threshold(gradient_magnitude/1, 50, 255, cv2.THRESH_BINARY)
          edges = edges/np.max(edges)
          edges[mask_img == 0] = 0
          small_mask = copy.deepcopy(mask_img)
          small_mask = resize(small_mask,mask_cutoff_factor)

          edges[small_mask == 1] = 0

          if partial == 1:

            if pos_or_neg_x == 0:
              edges[:,x_threshold:] = 0
            else:
              edges[:,:x_threshold] = 0

            if pos_or_neg_y == 0:
              edges[y_threshold:,:] = 0
            else:
              edges[:y_threshold,:] = 0

          indices = np.where(edges > 0.5)

          interval =random.randint(20,100)
          brain_mean = np.mean(scan_img[mask_img>0])
          only_brightness = random.randint(0,0)


          def add_rings(img):
            for ind in range(0,len(indices[0])):
              x = indices[0][ind]
              y = indices[1][ind]
              if only_brightness == 0:
                img[x][y] += brightness_factor
                if img[x][y] > brain_mean * 2:
                  img[x][y] = brain_mean * 2
                if img[x][y] >1:
                  img[x][y] = 1
                elif img[x][y]< 0:
                  img[x][y] = 0
            return img
          scan_img = add_rings(scan_img)
        return scan_img

      def simulate_metal_artifact(
        self, scan_img, mask, brain_only, bright_blur_level,
        dark_blur_level, bright_intensity, dark_intensity, distortion_alpha, distortion_sigma,
        bright_radius,dark_radius, dark_thickness, noise_lower, noise_upper, x_coord, y_coord
      ) -> None:
        """
        simulates artifacts caused by metal
        """
        blur_size = (2 * bright_blur_level) + 1
        overlay = np.zeros_like(scan_img)

        center = (x_coord, y_coord)
        bright_val = bright_intensity
        bright_color = (bright_val, bright_val, bright_val)
        dark_color = (dark_intensity, dark_intensity, dark_intensity)
        overlay = cv2.UMat(overlay)

        cv2.circle(overlay, center, bright_radius, bright_color, thickness=-1)
        overlay = overlay.get()

        overlay = self.elastic_transform_opencv(overlay, alpha=distortion_alpha, sigma=distortion_sigma)
        #overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        noise = np.random.uniform(noise_lower, noise_upper, overlay.shape)
        noise_overlay = np.zeros_like(scan_img)
        noise_overlay[overlay>0] +=noise[overlay>0]

        overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        

        overlay_array = np.array(overlay)
        if brain_only == True:
          #overlay_array[mask < 0.5] = 0
          overlay_array[(mask < 0.5) & (scan_img < 0.1)] = 0
        
        angle = np.random.uniform(0, 360)
        noise_overlay = self.utils.directional_blur(noise_overlay, angle, 7, 10)


        if brain_only == True:
          noise_overlay[(mask < 0.5) & (scan_img < 0.1)] = 0

        scan_img += noise_overlay
        scan_img += overlay_array
        
        #np.clip(scan_img, 0, 1, out=scan_img)
        #scan_img = cv2.addWeighted(scan_img, 1, overlay, 1, 0)
        overlay = np.zeros_like(scan_img)
        overlay = cv2.UMat(overlay)

        blur_size = (2 * dark_blur_level) + 1

        cv2.circle(overlay, center, dark_radius, dark_color, thickness=dark_thickness)
        overlay = overlay.get()

        blur_size = (2 * dark_blur_level) + 1
        overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)

        overlay = self.elastic_transform_opencv(overlay, alpha=distortion_alpha, sigma=distortion_sigma)
        overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
        
        overlay_array = np.array(overlay)
     
        if brain_only == True:
          overlay_array[(mask < 0.5) & (scan_img < 0.1)] = 0
        
        #scan_img = cv2.addWeighted(scan_img, 1, overlay, -1, 0)
        scan_img -= overlay_array
        #np.clip(scan_img, 0, 1, out=scan_img)

        return scan_img


      def elastic_transform_opencv(self,image, alpha, sigma):
        random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return distorted_image


      def overlay_random_noise(self, scan_img, mask_img, absolute_max,absolute_min, brain_only) -> None:
        """
        Adds noise over the
        entire image
        """
        for row in range(0,len(scan_img)):
            for pixel in range(0,len(scan_img[row])):
                if brain_only == True:
                  if mask_img[row][pixel] < 1:
                    continue
                max_pixel = random.uniform(absolute_min, absolute_max)
                min_pixel = random.uniform(absolute_min, max_pixel)
                scan_img[row][pixel]  += random.uniform(min_pixel, max_pixel)

        return scan_img

