import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate,zoom
import cv2

class Augmentimage():
    """
    Class to apply augmentations to 
    MRI Images.
    """
    def __init__(
        self, slice_image : np.array, 
        mask_image : np.ndarray , 
        extra_artifacts : bool = False
    ) -> None:
        """
        Initializes Variables

        Parameters
        ------------
        slice_image : np.darray
            Current slice of T1 MRI Scan
        mask_image : np.darray
            Mask that corresponds to current slice
        extra_artifacts : bool
            Determines if the extra augmentations
            (noise, signal drop, blurring)
            that go beyond normal variations
            (contrast, size, orientation)
            will be added to the image.
        """

        self.slice_image = slice_image
        self.mask_image = mask_image

        self.artifact_limit = 0
        self.extra_artifacts = extra_artifacts

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

    def pad_slice_to_square(self, slice_array : np.array, desired_size : int = 256) -> np.array:
        """
        Pad an image to increase its dimensions to the desired size.

        Parameters
        ------------------
        slice_array : numpy.ndarray 
            The 2D MRI slice.
        desired_size : int
            The size of the new square dimensions.

        Returns
        ---------------
        padded_slice : numpy.ndarray
            The padded slice.
        """
        original_size = slice_array.shape
        padding = (desired_size - original_size[1]) // 2
        extra_padding = (desired_size - original_size[1]) % 2

        padded_slice = np.pad(slice_array, ((0, 0), (padding, padding + extra_padding)), 'constant')

        return padded_slice

    mri_images = []
    masked_images = []
    x = 0

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

        if np.max(image) > 0:
            image = image/np.max(image)
        return image

    def add_straight_noise(self) -> None:
        """
        Adds random lines to the background 
        of the image
        """

        for row in range(0,len(self.slice_image)):
            max_pixel = random.uniform(0.5, 1)
            min_pixel = random.uniform(0.5, max_pixel)
            for pixel in range(0,len(self.slice_image[row])):
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel] = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel , max_pixel)

    def add_random_noise(self) -> None:
        """
        Adds noise to the background
        of the image
        """

        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel] = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel , max_pixel)


    def overlay_random_noise(self) -> None:
        """
        Adds noise over the
        entire image
        """

        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = random.uniform(0.1, 0.3)
                min_pixel = random.uniform(0, max_pixel)

                self.slice_image[row][pixel]  += random.uniform(min_pixel, max_pixel)



    def rotate_image(self) -> None: 
        """
        Randomly rotates the image
        """
        random_angle = random.uniform(0, 360)

        self.slice_image = rotate(self.slice_image, random_angle, \
        reshape=False, mode='constant',cval=0,order=1)
        self.mask_image = rotate(self.mask_image, random_angle, \
        reshape=False, mode='constant',cval=0,order=1)


    def resize_image(self) -> None:
        """
        Randomly resizes the image
        within a certain range
        """

        scale = random.uniform((0.5),1.2)
        def resize(img):
            resized_image = zoom(img, (scale, scale),order=1)
            background = np.zeros((256, 256), dtype=img.dtype)
            start_x = max((256 - resized_image.shape[1]) // 2, 0)
            start_y = max((256 - resized_image.shape[0]) // 2, 0)
            resized_image_clipped = resized_image[:min(256, resized_image.shape[0]),
                                                  :min(256, resized_image.shape[1])]

            background[start_y:start_y+resized_image_clipped.shape[0], \
                       start_x:start_x+resized_image_clipped.shape[1]] = resized_image_clipped

            return background


        self.slice_image = resize(self.slice_image)
        self.mask_image = resize(self.mask_image)


    def draw_random_shape(self) -> None:
        """
        Draws random shapes over the image
        """
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        height, width = self.slice_image.shape[:2]
        shape_choice = random.choice(['rectangle', 'circle', 'line','random'])
        self.draw_basic_shapes(shape_choice, self.slice_image, random.randint(1, 3))
        
        if shape_choice == 'random':
            num_points = random.randint(3, 10)  # Random number of points
            points = np.array([([random.randint(random.randint(0, width), width),\
            random.randint(random.randint(0, height), height)]) for _ in range(num_points)])
            cv2.polylines(self.slice_image, [points], isClosed=True, \
            color=color, thickness=random.randint(1, 3))

    def draw_basic_shapes(self, shape_choice, target , thickness):
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        height, width = self.slice_image.shape[:2]
        if shape_choice == 'rectangle':
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(start_point[0], width),\
            random.randint(start_point[1], height))
            cv2.rectangle(target, start_point, end_point, color, thickness=thickness)
        elif shape_choice == 'circle':
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(1, min(height, width) // 4)
            cv2.circle(target, center, radius, color, thickness=thickness)
        elif shape_choice == 'line':
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(0, width), random.randint(0, height))
            cv2.line(target, start_point, end_point, color, thickness=random.randint(1, 3))
        
        return target


    def draw_random_shape_with_opacity_grayscale(self) -> None:
        """
        Draws random shapes with
        varying levels of transparency 
        """
        height, width = self.slice_image.shape[:2]
        overlay = np.zeros_like(self.slice_image)
        alpha = random.uniform(0.2, 0.8)
        intensity = random.uniform(-0.5, 0.5)
        shape_choice = random.choice(['rectangle', 'circle', 'line', 'random'])
        overlay = self.draw_basic_shapes(shape_choice, overlay, -1)
        if shape_choice == 'random':
            num_points = random.randint(3, 10)
            points = np.array([([random.randint(random.randint(0, width), width),\
            random.randint(random.randint(0, height), height)]) for _ in range(num_points)])
            cv2.fillPoly(overlay, [points], intensity)

        self.slice_image = cv2.addWeighted(self.slice_image, 1, overlay, alpha, 0)
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

    def change_brain_intensity(self,
        intensity_change : float, 
        add : float
    ) -> None:
        """
        Randomly varies intensity of the brain

        Parameters
        -------------------
        intensity :  float
        """
        
        mask = self.mask_image > 0

        self.slice_image[mask & (add == 0)] += intensity_change
        self.slice_image[mask & (add == 1)] -= intensity_change
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

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


    def add_motion(self) -> None:
        """
        Adds random blurring to
        different parts of the image
        """

        for row in range(len(self.slice_image)):
            if row < 250:
                row_diff = random.randint(0, 2)
                intensity = random.randint(0, 5)
                for pixel in range(len(self.slice_image[row])):
                    target_row = min(row + row_diff, len(self.slice_image) - 1)
                    target_pixel = max(pixel - 2, 0)
                    self.slice_image[target_row][pixel] = \
                    (self.slice_image[target_row][pixel] + intensity * \
                    self.slice_image[target_row][target_pixel]) / (intensity + 1)


    def invert_brain(self) -> None:
        """
        Inverts pixel values 
        of brain region
        """

        mask = self.mask_image > 0
        self.slice_image[mask] = 1 - self.slice_image[mask]


    def simulate_ghosting_artifact(self):
        """
        Mirrors the MRI image
        and increases its transparency
        to simulate ghosting.
        """

        height, width = self.slice_image.shape[:2]
        num_copies = random.randint(1, 3)

        for _ in range(num_copies):
            mirrored_copy = cv2.flip(self.slice_image, flipCode=random.choice([-1, 0, 1])) 
            alpha = random.uniform(0.1, 0.2)  # Semi-transparent
            overlay = np.zeros_like(self.slice_image)
            x_offset = random.randint(0, width - mirrored_copy.shape[1])
            y_offset = random.randint(0, height - mirrored_copy.shape[0])
            overlay[y_offset:y_offset+mirrored_copy.shape[0], \
            x_offset:x_offset+mirrored_copy.shape[1]] = mirrored_copy
            self.slice_image = cv2.addWeighted(self.slice_image, 1, overlay, alpha, 0)

        np.clip(self.slice_image, 0, 1, out=self.slice_image)



    def invert_skull(self):
        """
        Inverts pixel values
        of skull region
        """

        mask = (self.mask_image == 0) & (self.slice_image > 0.01)
        self.slice_image[mask] = 1 - self.slice_image[mask]

    def remove_skull(self):
        """
        Removes skull from image
        """

        mask = self.mask_image == 0
        self.slice_image[mask] = 0


    def change_skull_intensity(self, intensity_change, add):

        condition_mask = (self.mask_image == 0) & (self.slice_image > 0.1)
        if add == 0:
            self.slice_image[condition_mask] += intensity_change
        else:
            self.slice_image[condition_mask] -= intensity_change
        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def change_skull_blackspace(self):

        replacement = random.randint(0,2)
        brain_region = (self.mask_image == 1)
        brain_mean = np.mean(self.slice_image[brain_region])

        condition_mask = (self.mask_image == 0) & (self.slice_image < 0.1)

        if replacement == 0:
          if not np.isnan(brain_mean):
            self.slice_image[condition_mask] = brain_mean
        elif replacement == 1:
          self.slice_image[condition_mask] = \
          np.random.uniform(0, 1, size=np.sum(condition_mask))

        else:
          self.slice_image[condition_mask] = random.uniform(0,1)

        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def change_brain_blackspace(self):
        replacement = random.randint(0,2)
        condition_mask = (self.mask_image == 1) & (self.slice_image < 0.1)
        brain_region =(self.mask_image == 1) & (self.slice_image > 0.1)
        brain_mean = np.mean(self.slice_image[brain_region])
        if replacement == 0:
          if not np.isnan(brain_mean):
            self.slice_image[condition_mask] = brain_mean
        elif replacement == 1:
          self.slice_image[condition_mask] = \
          np.random.uniform(0, 1, size=np.sum(condition_mask))

        else:
          self.slice_image[condition_mask] = random.uniform(0,1)

        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def add_signal_drop(self):

        add_signal_drop = np.random.randint(0, 20, \
        size=(self.slice_image.shape[0], 1))
        mask = (add_signal_drop == 0)
        broadcasted_mask = np.broadcast_to(mask, self.slice_image.shape)
        self.slice_image[broadcasted_mask] = \
        np.random.uniform(0, 0.15, np.sum(broadcasted_mask))

    def calculate_brain_size(self):
        size = np.sum(self.mask_image)
        return size

    def random_execution(self,probability):
        execution_val = random.randint(1,10)
        if self.artifact_limit > 6:
            return False
        if execution_val <= probability:
            return True
        else:
            return False
        self.artifact_limit +=1

    def apply_motion(self,chance = 3):
      if self.random_execution(chance):
        self.add_motion()

    def apply_inversion(self,chance = 1):
      if self.random_execution(chance):
        self.inversion = True
        self.invert_brain()
        self.invert_skull()

    def apply_noise(self,chance = {'regular':1,'straight':1,'overlay':2}):
        if self.random_execution(chance['regular']):
            self.add_random_noise()
        if self.random_execution(chance['straight']):
            self.add_straight_noise()
        if self.random_execution(chance['overlay']):
            self.overlay_random_noise()

    def apply_intensity(self,chance = {'brain':1,'skull':2}):
        if self.random_execution(chance['brain']):
            if not self.inversion:
                self.change_brain_intensity(\
                random.uniform(0, 0.5), random.randint(0,1))
        if self.random_execution(chance['skull']):
            if not self.inversion:
                self.change_skull_intensity(\
                random.uniform(0, 0.5), random.randint(0,1))

    def apply_signal_drop(self,chance = 1):
        if self.random_execution(chance):
            self.add_signal_drop()

    def apply_rotation(self,chance = 2):
        if self.random_execution(chance):
            self.rotate_image()
    
    def apply_resizing(self,chance = 2):
        if self.random_execution(chance):
            self.resize_image()

    def apply_blackspace_randomization(self,
    chance = {'brain':1,'skull':2}):
        if self.random_execution(chance['brain']):
            self.change_brain_blackspace()
        if self.random_execution(chance['skull']):
            self.change_skull_blackspace()

    def apply_ghosting(self,chance = 3):
        if self.random_execution(chance):
            self.simulate_ghosting_artifact()

    def apply_shapes(self,chance = {'high_opacity':5,'low_opacity':4}):
        if self.random_execution(chance['high_opacity']):
                for x in range(0,random.randint(1,7)):
                    self.draw_random_shape()
        if self.random_execution(chance['low_opacity']):
            for x in range(0,random.randint(1,7)):
                self.draw_random_shape_with_opacity_grayscale()

    def modify_images(self):
        brain_inverted = False
        self.brain_size = self.calculate_brain_size()

        def apply_to_all():
            self.mask_image = self.pad_slice_to_square(self.mask_image)
            self.slice_image= self.pad_slice_to_square(self.slice_image)
            self.mask_image = self.normalize(self.mask_image)
            self.slice_image = self.normalize(self.slice_image)
            self.mask_image[self.mask_image < 0.5] = 0
            self.mask_image[self.mask_image > 0.5] = 1
            contrast_change = random.uniform(0.1,1.8)
            intensity_change = 0.2
            intensity_add = 0
            self.change_brain_contrast(contrast_change)
            self.change_skull_contrast(contrast_change)
            self.rotate_image()
            self.resize_image()

         def add_extra_artifacts():
            contrast_change = random.uniform(0.1,1.8)
            self.change_skull_contrast(contrast_change)
            self.inversion = False
            self.apply_motion()
            self.apply_inversion()
            self.apply_noise({'regular':1,'straight':1,'overlay':0})
            self.apply_intensity()
            self.apply_signal_drop()
            self.apply_rotation()
            self.apply_resizing()
            self.apply_blackspace_randomization()
            self.apply_ghosting()
            self.apply_shapes()
            self.apply_noise()
            self.apply_signal_drop()
            self.apply_noise(\
            {'regular':0,'straight':0,'overlay':2})

        apply_to_all()
    
        if self.extra_artifacts:
          add_extra_artifacts()

        return self.slice_image, self.mask_image
