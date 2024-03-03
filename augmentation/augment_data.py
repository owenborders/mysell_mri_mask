import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate,zoom
import cv2



class Augmentimage():

    def __init__(self,slice_image,mask_image,extra_artifacts = False):
        self.slice_image = slice_image
        self.mask_image = mask_image

        self.artifact_limit = 0
        self.extra_artifacts = extra_artifacts

    def nii_to_numpy(self,nii_file_path):
        """
        Convert a .nii MRI file to a NumPy array.

        Parameters:
        nii_file_path (str): The path to the .nii file.

        Returns:
        numpy.ndarray: The MRI data as a NumPy array.
        """
        nii_image = nib.load(nii_file_path)
        data = nii_image.get_fdata()
        return np.array(data)

    # Example usage
    # numpy_array = nii_to_numpy('path_to_your_file.nii')
    def pad_slice_to_square(self,slice_array, desired_size=256):
        """
        Pad a 2D slice to make it square.

        Parameters:
        slice_array (numpy.ndarray): The 2D MRI slice.
        desired_size (int): The size of the new square dimensions.

        Returns:
        numpy.ndarray: The padded slice.
        """
        original_size = slice_array.shape
        # Calculate the padding needed
        padding = (desired_size - original_size[1]) // 2
        extra_padding = (desired_size - original_size[1]) % 2

        # Apply padding equally to both sides of the second dimension
        padded_slice = np.pad(slice_array, ((0, 0), (padding, padding + extra_padding)), 'constant')

        return padded_slice

    mri_images = []
    masked_images = []
    x = 0

    def normalize(self,image):
        if np.min(image) < 0:
            print(np.min(image))
        if np.max(image) >0:
            image = image/np.max(image)
        return image

    def add_straight_noise(self):
        for row in range(0,len(self.slice_image)):
            max_pixel = random.uniform(0.5, 1)
            min_pixel = random.uniform(0.5, max_pixel)
            for pixel in range(0,len(self.slice_image[row])):
                if self.slice_image[row][pixel] < 0:
                   print(self.slice_image[row][pixel])
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel , max_pixel)

    def add_random_noise(self):
        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)

                if self.slice_image[row][pixel] < 0:
                   print(self.slice_image[row][pixel])
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(self.slice_image[row])-1,0,-1):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)
                if self.slice_image[row][pixel] > 0.05:
                    break
                else:
                    self.slice_image[row][pixel]  = random.uniform(min_pixel , max_pixel)


    def overlay_random_noise(self):
        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                max_pixel = random.uniform(0.1, 0.3)
                min_pixel = random.uniform(0, max_pixel)

                self.slice_image[row][pixel]  += random.uniform(min_pixel, max_pixel)



    def rotate_image(self):
        random_angle = random.uniform(0, 360)

        # Rotate the image using scipy.ndimage.rotate
        # 'mode' is set to 'wrap'; you can change this as needed (e.g., 'reflect', 'nearest', etc.)
        self.slice_image = rotate(self.slice_image, random_angle, reshape=False, mode='constant',cval=0,order=1)
        self.mask_image = rotate(self.mask_image, random_angle, reshape=False, mode='constant',cval=0,order=1)


    def resize_image(self):
        # Generate a random scaling factor (between 0.5 and 1.25 for both dimensions)
        scale = random.uniform((0.5),1.2)
        # Resize the image
        def resize(img):
            resized_image = zoom(img, (scale, scale),order=1)

            # Create a black background with the specified dimensions (256x256)
            background = np.zeros((256, 256), dtype=img.dtype)

            # Calculate the position to place the resized image on the background
            start_x = max((256 - resized_image.shape[1]) // 2, 0)
            start_y = max((256 - resized_image.shape[0]) // 2, 0)

            # Adjust the resized image if it's larger than the desired dimensions
            resized_image_clipped = resized_image[:min(256, resized_image.shape[0]),
                                                  :min(256, resized_image.shape[1])]

            # Place the resized (and potentially clipped) image onto the black background
            background[start_y:start_y+resized_image_clipped.shape[0], \
                       start_x:start_x+resized_image_clipped.shape[1]] = resized_image_clipped

            return background


        self.slice_image = resize(self.slice_image)
        self.mask_image = resize(self.mask_image)


    def draw_random_shape(self):
        height, width = self.slice_image.shape[:2]

        # Choose a random color
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

        # Randomly choose a shape to draw
        shape_choice = random.choice(['rectangle', 'circle', 'line','random'])

        if shape_choice == 'rectangle':
            # Draw a rectangle
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(start_point[0], width), random.randint(start_point[1], height))
            cv2.rectangle(self.slice_image, start_point, end_point, color, thickness=random.randint(1, 3))

        elif shape_choice == 'circle':
            # Draw a circle
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(1, min(height, width) // 4)
            cv2.circle(self.slice_image, center, radius, color, thickness=random.randint(1, 3))

        elif shape_choice == 'line':
            # Draw a line
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(0, width), random.randint(0, height))
            cv2.line(self.slice_image, start_point, end_point, color, thickness=random.randint(1, 3))

        elif shape_choice == 'random':
          # Draw a random shape
            num_points = random.randint(3, 10)  # Random number of points
            points = np.array([([random.randint(random.randint(0, width), width),\
                                random.randint(random.randint(0, height), height)]) for _ in range(num_points)])
            cv2.polylines(self.slice_image, [points], isClosed=True, color=color, thickness=random.randint(1, 3))


    def draw_random_shape_with_opacity_grayscale(self):
        height, width = self.slice_image.shape[:2]

        # Create an overlay image for opacity control
        overlay = np.zeros_like(self.slice_image)

        # Adjust alpha for different opacities
        alpha = random.uniform(0.2, 0.8)

        # Choose intensity to brighten or darken
        intensity = random.uniform(-0.5, 0.5)

        # Randomly choose a shape to draw
        shape_choice = random.choice(['rectangle', 'circle', 'line', 'random'])

        if shape_choice == 'rectangle':
            # Draw a rectangle
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(start_point[0], width), random.randint(start_point[1], height))
            cv2.rectangle(overlay, start_point, end_point, intensity, thickness=-1)  # Filled

        elif shape_choice == 'circle':
            # Draw a circle
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(1, min(height, width) // 4)
            cv2.circle(overlay, center, radius, intensity, thickness=-1)  # Filled

        elif shape_choice == 'line':
            # Draw a line
            start_point = (random.randint(0, width), random.randint(0, height))
            end_point = (random.randint(0, width), random.randint(0, height))
            cv2.line(overlay, start_point, end_point, intensity, thickness=random.randint(1, 3))

        elif shape_choice == 'random':
            # Draw a random shape
            num_points = random.randint(3, 10)
            points = np.array([([random.randint(random.randint(0, width), width),\
                                 random.randint(random.randint(0, height), height)]) for _ in range(num_points)])
            cv2.fillPoly(overlay, [points], intensity)

        # Blend the overlay with the original image
        self.slice_image = cv2.addWeighted(self.slice_image, 1, overlay, alpha, 0)

        # Clip the values to ensure they are between 0 and 1
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

    def change_brain_intensity(self,intensity = 'default', add_value = 'default'):
        if add_value == 'default':
            add = random.randint(0, 1)
        else:
            add = add_value
        if intensity == 'default':
            intensity_change = random.uniform(0, 0.5) if add == 0 else random.uniform(0, 0.2)
        else:
            intensity_change = intensity
            add = 0
        mask = self.mask_image > 0

        self.slice_image[mask & (add == 0)] += intensity_change
        self.slice_image[mask & (add == 1)] -= intensity_change
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

    def change_brain_contrast(self,contrast = 'default'):
        # Mask for the brain region
        mask = self.mask_image > 0

        # Extract the brain region from the image
        brain_region = self.slice_image[mask]

        # Calculate mean intensity of the brain region
        mean_intensity = np.mean(brain_region)

        # Random factor for contrast change, between 0.5 (decrease contrast) and 1.5 (increase contrast)
        if contrast == 'default':
            contrast_factor = random.uniform(0.1, 1.8)
        else:
            contrast_factor = contrast


        # Adjust contrast
        # Subtract mean, scale by the factor, and add the mean back
        self.slice_image[mask] = (self.slice_image[mask] - mean_intensity) * contrast_factor + mean_intensity

        # Clip the values to ensure they stay within the range 0 to 1
        np.clip(self.slice_image, 0, 1, out=self.slice_image)

    def change_skull_contrast(self,contrast = 'default'):
        # Mask for the brain region
        mask = (self.mask_image == 0) & (self.slice_image > 0.1)


        # Extract the brain region from the image
        brain_region = self.slice_image[mask]

        # Calculate mean intensity of the brain region
        mean_intensity = np.mean(brain_region)

        # Random factor for contrast change, between 0.5 (decrease contrast) and 1.5 (increase contrast)
        if contrast == 'default':
            contrast_factor = random.uniform(0.1, 1.8)
        else:
            contrast_factor = contrast

        # Adjust contrast
        # Subtract mean, scale by the factor, and add the mean back
        self.slice_image[mask] = (self.slice_image[mask] - mean_intensity) * contrast_factor + mean_intensity

        # Clip the values to ensure they stay within the range 0 to 1
        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def add_motion(self):

        for row in range(len(self.slice_image)):
            if row < 250:
                row_diff = random.randint(0, 2)
                intensity = random.randint(0, 5)
                for pixel in range(len(self.slice_image[row])):
                    target_row = min(row + row_diff, len(self.slice_image) - 1)
                    target_pixel = max(pixel - 2, 0)
                    self.slice_image[target_row][pixel] = (self.slice_image[target_row][pixel] + intensity * self.slice_image[target_row][target_pixel]) / (intensity + 1)



    def add_brain_ghosting(self):

        for row in range(0,len(self.slice_image)):
            for pixel in range(0,len(self.slice_image[row])):
                #if self.mask_image[row][pixel] > 0:
                row_diff = 3
                intensity = random.uniform(0,1)
                if row < 250:
                    self.slice_image[row+row_diff][pixel] = ((self.slice_image[row+row_diff][pixel])  + (intensity*self.slice_image[row][pixel-10]))/(intensity+1 )


    def invert_brain(self):
        mask = self.mask_image > 0
        self.slice_image[mask] = 1 - self.slice_image[mask]


    def simulate_ghosting_artifact(self):
        height, width = self.slice_image.shape[:2]

        # Determine the number of mirrored copies (1 to 3)
        num_copies = random.randint(1, 3)

        for _ in range(num_copies):
            # Create a mirrored copy of the image
            mirrored_copy = cv2.flip(self.slice_image, flipCode=random.choice([-1, 0, 1]))  # Random flip: horizontal, vertical, or both

            # Adjust transparency of the mirrored copy
            alpha = random.uniform(0.1, 0.2)  # Semi-transparent
            overlay = np.zeros_like(self.slice_image)

            # Choose a random location to overlay the mirrored copy
            x_offset = random.randint(0, width - mirrored_copy.shape[1])
            y_offset = random.randint(0, height - mirrored_copy.shape[0])

            # Place the mirrored copy in the overlay
            overlay[y_offset:y_offset+mirrored_copy.shape[0], x_offset:x_offset+mirrored_copy.shape[1]] = mirrored_copy

            # Blend the overlay with the original image
            self.slice_image = cv2.addWeighted(self.slice_image, 1, overlay, alpha, 0)

        # Ensure the final image stays within the 0-1 range
        np.clip(self.slice_image, 0, 1, out=self.slice_image)



    def invert_skull(self):
        mask = (self.mask_image == 0) & (self.slice_image > 0.01)
        self.slice_image[mask] = 1 - self.slice_image[mask]

    def remove_skull(self):
        mask = self.mask_image == 0
        self.slice_image[mask] = 0


    def change_skull_intensity(self,intensity = 'default',add_value = 'default'):


        # Determine the intensity change
        if add_value == 'default':
            add = random.randint(0, 1)
        else:
            add = add_value
        if intensity == 'default':
            intensity_change = random.uniform(0, 0.5) if add == 0 else random.uniform(0, 0.2)
        else:
            intensity_change = intensity
            add = 0


        # Create a mask for the condition
        condition_mask = (self.mask_image == 0) & (self.slice_image > 0.1)

        # Apply the intensity change
        if add == 0:
            self.slice_image[condition_mask] += intensity_change
        else:
            self.slice_image[condition_mask] -= intensity_change

        # Clip values to be within [0, 1]
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
          self.slice_image[condition_mask] = np.random.uniform(0, 1, size=np.sum(condition_mask))

        else:
          self.slice_image[condition_mask] = random.uniform(0,1)


        # Clip values to be within [0, 1]
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
          self.slice_image[condition_mask] = np.random.uniform(0, 1, size=np.sum(condition_mask))

        else:
          self.slice_image[condition_mask] = random.uniform(0,1)


        # Clip values to be within [0, 1]
        np.clip(self.slice_image, 0, 1, out=self.slice_image)


    def add_signal_drop(self):

        add_signal_drop = np.random.randint(0, 20, size=(self.slice_image.shape[0], 1))

        # Create a mask where add_signal_drop equals 0 and broadcast it to match self.slice_image's shape
        mask = (add_signal_drop == 0)
        broadcasted_mask = np.broadcast_to(mask, self.slice_image.shape)

        # Apply mask and assign random uniform values where mask is True
        self.slice_image[broadcasted_mask] = np.random.uniform(0, 0.15, np.sum(broadcasted_mask))

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
                self.change_brain_intensity()
        if self.random_execution(chance['skull']):
            if not self.inversion:
                self.change_skull_intensity()

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


        apply_to_all()


        def add_extra_artifacts():
            self.change_skull_contrast()
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
    


        if self.extra_artifacts:
          add_extra_artifacts()

        return self.slice_image, self.mask_image


folder = 'NFBS_Dataset'

x = 0

def nii_to_numpy(nii_file_path):
    """
    Convert a .nii MRI file to a NumPy array.

    Parameters:
    nii_file_path (str): The path to the .nii file.

    Returns:
    numpy.ndarray: The MRI data as a NumPy array.
    """
    nii_image = nib.load(nii_file_path)
    data = nii_image.get_fdata()
    return np.array(data)

for subject in os.listdir(folder):

    mri_images = []
    masked_images = []
    final_mri_images = []
    final_masked_images = []
    mri_array = []
    masked_array = []
    x+=1
    print(x)
    if x < 101:
      continue


    for file in os.listdir(folder + '/' + subject):
        print(file)
        if 'brain' not in file:
            brain_img = nii_to_numpy(folder+'/'+subject + '/' + file)
            print(brain_img.shape)
            for slice_number in range(0,brain_img.shape[2]):
                mri_images.append((brain_img[:, :, slice_number]))
            for slice_number in range(0,brain_img.shape[1]):
                mri_images.append((brain_img[:, slice_number, :]))
            for slice_number in range(0,brain_img.shape[0]):
                mri_images.append((brain_img[slice_number,:, :]))


    for file in os.listdir(folder + '/' + subject):
        if 'mask' in file:
            mask_img = nii_to_numpy(folder+'/'+subject + '/' + file)
            for slice_number in range(0,mask_img.shape[2]):
                masked_images.append((mask_img[:, :, slice_number]))
            for slice_number in range(0,mask_img.shape[1]):
                masked_images.append((mask_img[:, slice_number, :]))
            for slice_number in range(0,mask_img.shape[0]):
                masked_images.append((mask_img[slice_number,:, :]))

    for slice_number in range(0,len(mri_images)):
      augmented_mri,augmented_mask = Augmentimage(mri_images[slice_number],masked_images[slice_number]).modify_images()
      final_mri_images.append(augmented_mri)
      final_masked_images.append(augmented_mask)
      augmented_mri,augmented_mask = Augmentimage(mri_images[slice_number],masked_images[slice_number],True).modify_images()
      final_mri_images.append(augmented_mri)
      final_masked_images.append(augmented_mask)

    index_list = list(range(len(final_mri_images)))

    # Shuffle the index list
    random.shuffle(index_list)

    # Reorder both lists using the shuffled index list
    mri_array = [final_mri_images[i] for i in index_list]
    masked_array = [final_masked_images[i] for i in index_list]
    mri_array = np.array(mri_array)
    masked_array = np.array(masked_array)
    mri_array = np.array([i.reshape(256, 256, 1) for i in mri_array])
    masked_array = np.array([i.reshape(256, 256, 1) for i in masked_array])

    print(masked_array.shape)
    print(mri_array.shape)

    os.makedirs(f'training_set/subject_{x}', exist_ok=True)

    np.save(f'training_set/subject_{x}/mri_array.npy',mri_array)
    np.save(f'training_set/subject_{x}/mask_array.npy',masked_array)

try:
    mri_array = np.array(mri_images)
    mri_images = []
    #print(mri_array.shape)
    #np.save('/content/drive/MyDrive/mri_array.npy',mri_array)
    mri_array = []

    masked_array = np.array(masked_images)
    masked_images = []
except Exception as e:
    print(e)

