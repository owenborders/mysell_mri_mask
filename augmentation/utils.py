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


class Utils():

  def normalize(self, image : np.array) -> np.array:
    """
    Normalizes an image to
    be between 0 and 1 by
    dividing by the max value.

    Parameters
    -------------
    image : numpy.darray
        input image

    Returns
    --------------
    image : numpy.darray
        normalized image
    """
    image[image < 0] = 0
    if np.max(image) > 0:
        image = image/np.max(image)
    
    return image

  def nii_to_numpy(self, nii_file_path : str) -> np.array:
    """
    Convert a .nii MRI file to a NumPy array.

    Parameters
    ----------------------
    nii_file_path : str
        The path to the .nii file.

    Returns
    --------------------
    data : np.ndarray
        The MRI data as a NumPy array.
    """

    nii_image = nib.load(nii_file_path)
    data = np.array(nii_image.get_fdata())
    return data

  def pad_slice_to_square(self,slice_array, desired_size=256):
      """
      Pad a 2D slice to make it square with desired_size x desired_size dimensions.

      Parameters:
      slice_array (numpy.ndarray): The 2D MRI slice.
      desired_size (int): The size of the new square dimensions.

      Returns:
      numpy.ndarray: The padded slice.
      """
      height, width = slice_array.shape

      # Calculate the padding needed for both dimensions
      height_padding = (desired_size - height) // 2
      extra_height_padding = (desired_size - height) % 2

      width_padding = (desired_size - width) // 2
      extra_width_padding = (desired_size - width) % 2

      # Apply padding equally to both sides of each dimension
      padded_slice = np.pad(slice_array,
                            ((height_padding, height_padding + extra_height_padding),
                            (width_padding, width_padding + extra_width_padding)),
                            'constant')

      return padded_slice

  def calculate_ssim(self, imageA, imageB):
      """
      Calculate the Structural Similarity Index (SSIM) between two images.

      Parameters:
      - imageA: NumPy array representing the first image.
      - imageB: NumPy array representing the second image.

      Returns:
      - ssim_index: The calculated SSIM index.
      """
      # Ensure the images are in the same shape
      if imageA.shape != imageB.shape:
          raise ValueError("Input images must have the same dimensions.")

      # Convert images to grayscale if they are in color
      if imageA.ndim == 3:
          imageA = np.dot(imageA[...,:3], [0.2989, 0.5870, 0.1140])
      if imageB.ndim == 3:
          imageB = np.dot(imageB[...,:3], [0.2989, 0.5870, 0.1140])

      # Compute SSIM between two images
      ssim_index, _ = compare_ssim(imageA, imageB, full=True)
      return ssim_index


  def mse(self,imageA, imageB):
      """Compute the Mean Squared Error between two images."""
      err = np.sum((imageA.astype("float") - imageB.astype("float")))
      return (abs(err) ** 0.2)/10
      err /= float(imageA.shape[0] * imageA.shape[1])

      return err

  def calc_combined_score(self,imageA, imageB):
    mse = self.mse(imageA, imageB)
    ssim = (1 - (self.calculate_ssim(imageA, imageB))) * 20
    combined =  ((1.5 * mse) + (ssim))

    return combined

  def calc_mask_boundaries(self, mask_img):
    for row in range(0,len(mask_img)):
      print('***************************')
      print(np.mean(mask_img[row]))

  def calculate_brain_mean(self, scan_img, mask_img):
    brain_mean = np.mean(scan_img[mask_img>0])
    return brain_mean

  def calculate_brain_max(self, scan_img, mask_img):
    brain_max = np.max(scan_img[mask_img>0])
    return brain_max

  def calculate_brain_std(self, scan_img, mask_img):
    brain_std = np.std(scan_img[mask_img>0])
    return brain_std


  def directional_blur(self,img, angle, kernel_size=7, sigma=10):    
    kernel = np.zeros((kernel_size, kernel_size))
    
    center = kernel_size // 2
    
    angle = np.deg2rad(angle) 
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * np.exp(sigma * (np.cos(2 * (np.arctan2(y, x) - angle)) - 1))

    kernel /= np.sum(kernel)
    
    blurred_img = convolve(img, kernel)
    return blurred_img

  def resize(self,img,scale):
          resized_image = zoom(img, (scale, scale),order=1)
          background = np.zeros((256, 256), dtype=img.dtype)
          start_x = max((256 - resized_image.shape[1]) // 2, 0)
          start_y = max((256 - resized_image.shape[0]) // 2, 0)
          resized_image_clipped = resized_image[:min(256, resized_image.shape[0]),
                                                :min(256, resized_image.shape[1])]

          background[start_y:start_y+resized_image_clipped.shape[0], \
                    start_x:start_x+resized_image_clipped.shape[1]] = resized_image_clipped

          return background


  def calculate_mask_edges(self, mask_img):
    binary_mask = mask_img.astype(bool)
    
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    top_row = np.argmax(rows)  
    bottom_row = len(rows) - np.argmax(rows[::-1]) - 1
    
    left_col = np.argmax(cols) 
    right_col = len(cols) - np.argmax(cols[::-1]) - 1  
    
    return {'top': top_row, 'bottom': bottom_row, 'left': left_col,'right': right_col}

