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

from known_artifacts import KnownArtifacts
from utils import Utils
class RestorationTrainSet():

    def __init__(self):

        self.folder = '/data/pnlx/projects/mysell_masking_cnn/NFBS_Dataset'

        self.utils = Utils()
        self.known_artifacts = KnownArtifacts()


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
    
    def run_script(self):
        self.loop_subjects()

    def loop_subjects(self):
        sub_count = 0
        for subject in os.listdir(self.folder):

            mri_images = []
            masked_images = []
            final_orig_mri_images = []
            final_augmented_mri_images = []
            sub_count+=1
            print(f'Subject: {sub_count}')
            if sub_count < 6 or sub_count > 20:
                continue



            for file in os.listdir(self.folder + '/' + subject):
                if 'brain' not in file:
                    brain_img = self.nii_to_numpy(self.folder+'/'+subject + '/' + file)
                    for slice_number in range(0,brain_img.shape[2]):
                        mri_images.append((brain_img[:, :, slice_number]))
                    for slice_number in range(0,brain_img.shape[1]):
                        mri_images.append((brain_img[:, slice_number, :]))
                    for slice_number in range(0,brain_img.shape[0]):
                        mri_images.append((brain_img[slice_number,:, :]))


            for file in os.listdir(self.folder + '/' + subject):
                if 'mask' in file:
                    mask_img = self.nii_to_numpy(self. folder+'/'+subject + '/' + file)
                    for slice_number in range(0,mask_img.shape[2]):
                        masked_images.append((mask_img[:, :, slice_number]))
                    for slice_number in range(0,mask_img.shape[1]):
                        masked_images.append((mask_img[:, slice_number, :]))
                    for slice_number in range(0,mask_img.shape[0]):
                        masked_images.append((mask_img[slice_number,:, :]))

            for slice_number in range(0,len(mri_images)):
                mask_img = masked_images[slice_number]
                if np.sum(mask_img) < 1000:
                    continue
                mri_images[slice_number] = self.utils.pad_slice_to_square(mri_images[slice_number])
                mri_images[slice_number] = self.utils.normalize(mri_images[slice_number])
                mask_img = self.utils.pad_slice_to_square(mask_img)
                orig_mri = copy.deepcopy(mri_images[slice_number])
                augmented_mri = copy.deepcopy(mri_images[slice_number])
                final_orig_mri_images.append(orig_mri)
                final_augmented_mri_images.append(orig_mri)

                brain_max = self.utils.calculate_brain_max(orig_mri, mask_img)
                brain_mean = self.utils.calculate_brain_mean(orig_mri, mask_img)
                brain_std = self.utils.calculate_brain_std(orig_mri, mask_img)
                mask_edges = self.utils.calculate_mask_edges(mask_img)

                if (65 < sub_count < 90) or (sub_count >= 90) or (30< sub_count < 35):
                    noise_max = random.uniform(0.05, 0.4)
                    noise_min = random.uniform(0.01, noise_max/2)
                    augmented_mri = self.known_artifacts.overlay_random_noise(augmented_mri,
                    mask_img, noise_max, noise_min, random.choice([True, False]))

                if (35 < sub_count < 75) or (sub_count >= 90):
                    brightness_factor_list = []
                    ring_distance_list = []
                    for x in range(0,random.randint(3,20)):
                        val = random.uniform(0,0.045)
                        dark = random.randint(0,1)
                        if dark == 1:
                            val *= -1
                        brightness_factor_list.append(val)
                        ring_distance_list.append(1 - (0.045 * x))
                    augmented_mri = self.known_artifacts.create_ring(augmented_mri,
                    mask_img, x_threshold = random.randint(mask_edges['left'],mask_edges['right']),
                    y_threshold = random.randint(mask_edges['top'],mask_edges['bottom']), pos_or_neg_x = random.randint(0,1),
                    pos_or_neg_y = random.randint(0,1), brightness_factors = brightness_factor_list, 
                    ring_distances= ring_distance_list, mask_cutoff_factor = random.uniform(.15,.9))

                dark_blur_level = random.randint(20,30)
                dark_intensity = (brain_mean - (brain_std * random.randint(-2,2))) *(dark_blur_level/5)
                noise_lower = random.uniform(-0.4,-0.05)

                bright_intensity = random.uniform(0, 0.9)

                
                if sub_count < 10: 
                    met_radius = random.randint(5, 12)
                    bright_intensity = random.uniform(0.5, 0.9)
                elif 10 <= sub_count < 20:
                    met_radius = random.randint(13, 20)
                elif 20 <= sub_count < 30:
                    met_radius = random.randint(21, 50)
                else:
                    met_radius = random.randint(5, 50)

                if (sub_count < 50 )or (sub_count >= 90):
                    augmented_mri= self.known_artifacts.simulate_metal_artifact(augmented_mri,
                    mask_img, brain_only = True, bright_blur_level = random.randint(1,5),
                    dark_blur_level= dark_blur_level, bright_intensity = bright_intensity,
                    dark_intensity = dark_intensity, distortion_alpha = random.randint(10, 300), distortion_sigma = random.randint(6,12),
                    bright_radius = met_radius, dark_radius = met_radius, dark_thickness = random.randint(3,7), noise_lower = noise_lower,
                    noise_upper = noise_lower + (random.uniform(0, abs(noise_lower))),\
                    x_coord = random.randint(mask_edges['left'],mask_edges['right']),\
                    y_coord = random.randint(mask_edges['top'],mask_edges['bottom']))

                augmented_mri = self.utils.normalize(augmented_mri)
                final_augmented_mri_images.append(augmented_mri)
                final_orig_mri_images.append(orig_mri)
                index_list = list(range(len(final_augmented_mri_images)))
                random.shuffle(index_list)
                aug_mri_array = [final_augmented_mri_images[i] for i in index_list]
                orig_mri_array = [final_orig_mri_images[i] for i in index_list]
                aug_mri_array = np.array(aug_mri_array)
                orig_mri_array = np.array(orig_mri_array)
                aug_mri_array = np.array([i.reshape(256, 256, 1) for i in aug_mri_array])
                orig_mri_array = np.array([i.reshape(256, 256, 1) for i in orig_mri_array])

                os.makedirs(f'/data/pnlx/projects/mysell_masking_cnn/final_project/augmentations/training_sets/artifact_removal/subject_{sub_count}', exist_ok=True)

                np.save(f'/data/pnlx/projects/mysell_masking_cnn/final_project/augmentations/training_sets/artifact_removal/subject_{sub_count}/aug_mri_array.npy',aug_mri_array)
                np.save(f'/data/pnlx/projects/mysell_masking_cnn/final_project/augmentations/training_sets/artifact_removal/subject_{sub_count}/orig_mri_array.npy',orig_mri_array)



if __name__ == '__main__':
    RestorationTrainSet().run_script()