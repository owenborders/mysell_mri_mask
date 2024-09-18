import numpy as np

import tensorflow as tf
import nibabel as nib
from PIL import Image
import random
from tensorflow.keras.models import load_model
import os
from matplotlib import pyplot as plt
import cv2
import sys
import os, sys

import copy
sys.path.insert(0, os.path.abspath(".."))

from augmentation.utils import Utils

#plan
# Test each slice for each scan for each test set
# collect total dice coefficients and create histograms of avrge performance per slice
# test sets will consist of normal unmodified scans,
# normal scans with normal variations, normal scans with minor to moderate artificial artifacts
# real mp2rage scans, synthetic mp2rage scans, 
# normal scans from other datasets, scans with extreme synthetic augmentations/artifacts


class EvaluateModel():
    def __init__(self, model_path, test_set_directory):
        self.model_path = model_path
        self.performance_per_slice = {'axial':{},'sagittal':{},'coronal':{}}
        self.test_set_directory = test_set_directory
        self.all_dice_losses = []

        self.utils = Utils()
        

    def run_script(self):
        self.load_model()
        #self.loop_directory()
        parent_dir = '/data/pnlx/projects/mysell_masking_cnn/NFBS_Dataset/'
        for sub in os.listdir(parent_dir):
            print(sub)
            sub_dir = f'{parent_dir}{sub}/'
            #sub_dir = '/data/pnlx/projects/mysell_masking_cnn/NFBS_Dataset/A00064081/'

            self.load_niftii_file(f'{sub_dir}/sub-{sub}_ses-NFB3_T1w.nii.gz',\
                                f'{sub_dir}/sub-{sub}_ses-NFB3_T1w_brainmask.nii.gz')
            
            print(self.performance_per_slice)

    def dice_loss(self, y_true, y_pred):
        smooth = 1e-6  
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f)\
        + tf.reduce_sum(y_pred_f) + smooth)


    def load_model(self):
        def dice_loss( y_true, y_pred):
          smooth = 1e-6  
          y_true_f = tf.reshape(y_true, [-1])
          y_pred_f = tf.reshape(y_pred, [-1])
          intersection = tf.reduce_sum(y_true_f * y_pred_f)
          return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f)\
          + tf.reduce_sum(y_pred_f) + smooth)

        def combined_dice_bce_loss( y_true, y_pred, alpha=0.5):
            dice_loss = dice_loss(y_true, y_pred)
            bce_loss = binary_crossentropy(y_true, y_pred)
            combined_loss = (alpha * dice_loss) + ((1 - alpha) * bce_loss)
            return combined_loss

        def apply_threshold(preds, threshold=0.5):
            return tf.cast(tf.greater(preds, threshold), tf.float64)

        def boundary_loss(self,y_true, y_pred):
          sobel_filter = tf.image.sobel_edges
          y_true_edges = sobel_filter(y_true)
          y_pred_edges = sobel_filter(y_pred)
          loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

          return loss

        def combined_loss(y_true, y_pred, alpha=0.3, beta=0.3, gamma=1.5):
          bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
          d_loss = self.dice_loss(y_true, y_pred)
          boundary = self.boundary_loss(y_true, y_pred)

          total_loss = (alpha * bce) + (beta * d_loss) + (gamma * boundary)
          
          return total_loss
        
        def loss(y_true, y_pred):
            pass

        def weighted_boundary_loss(y_true, y_pred):
            pass

        custom_objects = {
            'combined_dice_bce_loss':combined_dice_bce_loss,
            'dice_loss':dice_loss,
            'boundary_loss':boundary_loss,
            'combined_loss':combined_loss,
            'weighted_boundary_loss':weighted_boundary_loss,
            "loss":loss
        }

        self.evaluated_model = load_model(\
        self.model_path,\
        custom_objects=custom_objects)

    def loop_directory(self):
        for sub in os.listdir(self.test_set_directory):
            if int(sub.split('_')[-1]) < 116:
                continue
            mri_array = np.load(self.test_set_directory + f'{sub}/mri_array.npy')
            mask_array = np.load(self.test_set_directory + f'{sub}/mask_array.npy')
            self.test_accuracy(mask_array, mri_array)

    def load_niftii_file(self, mri_path, mask_path):
        nii_data_array = nib.load(mri_path)
        nii_data_array = nii_data_array.get_fdata()
        nii_data_array = np.array(nii_data_array, dtype=np.float64)

        mask_data_array = nib.load(mask_path)
        mask_data_array = mask_data_array.get_fdata()
        mask_data_array = np.array(mask_data_array, dtype=np.float64)


        for slice_count in range(0,max(nii_data_array.shape)):
            print(slice_count)
            # sagittal = [:,:,x] size 240
            # axial = [:,x,:] size 256
            #coronal = [x,:,:]
            view_list =  ['coronal','axial','sagittal']

            for view in range(0,len(view_list)):
                batch = []
                if view == 0 and slice_count < nii_data_array.shape[view]:
                    curr_slice = cv2.resize(nii_data_array[slice_count,:,:], \
                                            (256, 256), interpolation=cv2.INTER_LINEAR)
                    batch.append(self.utils.normalize(\
                    self.utils.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)

                    curr_mask = cv2.resize(mask_data_array[slice_count,:,:], \
                        (256, 256), interpolation=cv2.INTER_LINEAR)

                    curr_mask = self.utils.pad_slice_to_square(curr_mask).reshape(256,256,1)
                    output = float(self.predict(curr_mask,batch))

                    self.performance_per_slice[view_list[view]].setdefault(slice_count,0)
                    self.performance_per_slice[view_list[view]][slice_count] += output
                    

                    

                elif view == 1 and slice_count < nii_data_array.shape[view]:
                    print(view)
                    #print(nii_data_array[:,slice_count,:].shape)
                    curr_slice = cv2.resize(nii_data_array[:,slice_count,:], (256, 256), \
                                            interpolation=cv2.INTER_LINEAR)
                    batch.append(self.utils.normalize(self.utils.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)

                    curr_mask = cv2.resize(mask_data_array[:,slice_count,:], (256, 256), \
                                            interpolation=cv2.INTER_LINEAR)
                    curr_mask = self.utils.pad_slice_to_square(curr_mask).reshape(256,256,1)
                    output = float(self.predict(curr_mask,batch))

                    self.performance_per_slice[view_list[view]].setdefault(slice_count,0)
                    self.performance_per_slice[view_list[view]][slice_count] += output

                    
                elif view == 2 and slice_count < nii_data_array.shape[view]:
                    print(slice_count)
                    print(view)
                    #print(nii_data_array[:,:,slice_count].shape)
                    curr_slice = cv2.resize(nii_data_array[:,:,slice_count], \
                                            (256, 256), interpolation=cv2.INTER_LINEAR)
                    batch.append(self.utils.normalize(\
                    self.utils.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)

                    curr_mask  = cv2.resize(mask_data_array[:,:,slice_count], \
                                            (256, 256), interpolation=cv2.INTER_LINEAR)
                    curr_mask = self.utils.pad_slice_to_square(curr_mask).reshape(256,256,1)
                    output = float(self.predict(curr_mask,batch))

                    self.performance_per_slice[view_list[view]].setdefault(slice_count,0)
                    self.performance_per_slice[view_list[view]][slice_count] += output


                



    def apply_threshold(self, preds, threshold = 0.5):

        return tf.cast(tf.greater(preds,threshold), tf.float64)
    

    def predict(self, true_mask, slice_input):
        prediction = self.evaluated_model.predict(slice_input)
        prediction = self.apply_threshold(prediction)
        true_mask = self.apply_threshold(true_mask) 
        dice_loss = self.dice_loss(true_mask,prediction)

        slice = slice_input.reshape(256,256)

        predicted_mask = np.array(prediction).reshape(256,256)

        true_mask = np.array(true_mask).reshape(256,256)

        true_slice = copy.deepcopy(slice)

        slice[predicted_mask>0] += 2

        true_slice[true_mask>0] += 1

        if dice_loss > 0.1 : 
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

            # Display the first image (poor prediction)
            axs[0].imshow(slice, cmap='gray')
            axs[0].axis('off')  # Turn off axis

            # Display the second image (true prediction)
            axs[1].imshow(true_slice, cmap='gray')
            axs[1].axis('off')  # Turn off axis

            # Save the combined figure
            plt.savefig('combined_predictions.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the figure to free up memory




        return dice_loss


    def test_accuracy(self, mask_array, mri_array):
        for mri_slice in range(0,mri_array.shape[0]):
            prediction = self.evaluated_model.predict(np.array([mri_array[mri_slice]]))
            prediction = self.apply_threshold(prediction)
            true_mask = self.apply_threshold(mask_array[mri_slice]) 
            dice_loss = self.dice_loss(true_mask,prediction)
            self.all_dice_losses.append(dice_loss)


            print(f"dice_loss: {np.mean(self.all_dice_losses)}")


if __name__ == '__main__':
    model_path = '/data/pnlx/projects/mysell_masking_cnn/final_results/models/attention_unet_stage_one_best_only.h5'
    test_set_directory = '/data/pnlx/projects/mysell_masking_cnn/training_set_stage_2/'
    EvaluateModel(model_path,test_set_directory).run_script()
