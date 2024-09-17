import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, Activation, GaussianNoise
import tensorflow.keras.backend as K
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd 
from scipy.ndimage import binary_fill_holes,binary_dilation, binary_erosion




class GenerateMasks():
    def __init__(self, file_path, output_path):
        self.directory = '/data/predict1/data_from_nda/MRI_ROOT/rawdata'

        self.mask_dir = '/data/predict1/data_from_nda/MRI_ROOT/derivatives/masks/t1'

        self.total_score = 0
        self.file_path = file_path
        self.output_path = output_path

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

        self.restored_model1 = load_model(\
            '/data/pnlx/projects/mysell_masking_cnn/mask_6_1_every_epoch_3_29_best_only.h5',\
            custom_objects=custom_objects)
        self.restored_model2 = load_model(\
            '/data/pnlx/projects/mysell_masking_cnn/dice_loss_prioritized_best_only.h5',\
            custom_objects=custom_objects)

        self.restored_model3 = load_model(\
            '/data/pnlx/projects/mysell_masking_cnn/boundary_loss_prioritized.h5',\
            custom_objects=custom_objects)
        
        """self.qc_model = load_model(\
            '/data/pnlx/projects/mysell_masking_cnn/mri_qc/qc_model_ring_test_best_only.h5',\
            custom_objects=custom_objects)""" 

    def run_script(self):
        self.convert_to_array(self.file_path)
        #self.loop_scans()
        """for file in os.listdir(self.val_directory):
          if 'mri' in file:
            scans = np.load(self.val_directory + '/' + file)
            print(scans.shape)
            for slice in scans:
              self.predict(np.array([slice]))"""


    def pad_slice_to_square(self, slice_array, desired_size=256):
        self.original_size = slice_array.shape

        self.padding_height = (desired_size - self.original_size[0]) // 2
        self.extra_padding_height = (desired_size - self.original_size[0]) % 2


        self.padding_width = (desired_size - self.original_size[1]) // 2
        self.extra_padding_width = (desired_size - self.original_size[1]) % 2
        padded_slice = np.pad(slice_array,
                              ((self.padding_height, self.padding_height + self.extra_padding_height),
                               (self.padding_width, self.padding_width + self.extra_padding_width)),
                              'constant')

        return padded_slice

    def reverse_padding(self,padded_slice):
        original_height, original_width = self.original_size

        start_row = (padded_slice.shape[0] - original_height) // 2
        end_row = start_row + original_height

        start_col = (padded_slice.shape[1] - original_width) // 2
        end_col = start_col + original_width

        unpadded_slice = padded_slice[start_row:end_row, start_col:end_col]

        unpadded_slice = unpadded_slice.reshape(self.original_size)

        print(unpadded_slice.shape)

        return unpadded_slice

    def save_as_nii(self):
        original_nii_file = nib.load(self.original_t1_file)
        img = nib.Nifti1Image(self.compiled_slices, original_nii_file.affine)

        nib.save(img, self.output_path )

    def normalize(self,image):
        if np.min(image) < 0:
            print(np.min(image))
        if np.max(image) >0:
            image = image/np.max(image)
        return image

    def convert_to_array(self,path):
        self.original_t1_file = path
        nii_file = nib.load(path)
        nii_data = nii_file.get_fdata()
        nii_data_array = np.array(nii_data, dtype=np.float64)
        print(nii_data_array)
        print(nii_data_array.shape)
        slice_total = len(nii_data_array)
        self.combined_slices = np.zeros((slice_total, slice_total, slice_total))

        print(slice_total)
        self.compiled_slices = np.zeros(nii_data_array.shape)

        for slice_count in range(0,max(nii_data_array.shape)):
            print(slice_count)
            #if slice_count < 200:
             #   continue
            # sagittal = [:,:,x] size 240
            # axial = [:,x,:] size 256
            #coronal = [x,:,:]
            view_list =  ['coronal','axial','sagittal']

            for view in range(0,len(view_list)):
                batch = []
                if view == 0 and slice_count < nii_data_array.shape[view]:
                    curr_slice = cv2.resize(nii_data_array[slice_count,:,:], \
                                            (256, 256), interpolation=cv2.INTER_LINEAR)
                                            
                    batch.append(self.normalize(\
                    self.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)
                    #print(nii_data_array[slice_count,:,:].shape)
                    output = self.predict(np.array(batch), \
                                          nii_data_array[slice_count,:,:].shape,nii_data_array[slice_count,:,:])
                    self.compiled_slices[slice_count,:,:] = output
                    
                elif view == 1 and slice_count < nii_data_array.shape[view]:
                    print(view)
                    #print(nii_data_array[:,slice_count,:].shape)
                    curr_slice = cv2.resize(nii_data_array[:,slice_count,:], (256, 256), \
                                            interpolation=cv2.INTER_LINEAR)
                    
                    batch.append(self.normalize(self.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)
                    output = self.predict(np.array(batch), \
                                          nii_data_array[:, slice_count,:].shape,nii_data_array[:,slice_count,:])
                    self.compiled_slices[:,slice_count,:] = output
                    
                elif view == 2 and slice_count < nii_data_array.shape[view]:
                    print(slice_count)
                    print(view)
                    #print(nii_data_array[:,:,slice_count].shape)
                    
                    curr_slice = cv2.resize(nii_data_array[:,:,slice_count], \
                                            (256, 256), interpolation=cv2.INTER_LINEAR)
                    batch.append(self.normalize(\
                    self.pad_slice_to_square(curr_slice)).reshape(256,256,1))
                    batch = np.array(batch)
                    output = self.predict(np.array(batch), \
                                          nii_data_array[:,:,slice_count].shape,nii_data_array[:,:,slice_count])
                    self.compiled_slices[:,:,slice_count]  = output
                        
        self.save_as_nii()

    def read_files(self):
        t1_path = '/data/predict1/data_from_nda/MRI_ROOT/rawdata/sub-ME66368/ses-202307181/anat/sub-ME66368_ses-202307181_run-1_T2w.nii.gz'

        self.convert_to_array(t1_path)

    def loop_scans(self):
        for file in os.listdir(self.directory):
            if 'sub' in file:
                #if 'BI02450' not in file:
                 #   continue
               # if not any(sub in file for sub in self.list_of_bad_scans):
                #    continue
                self.subject = file
                for session in os.listdir(self.directory +'/' + file):
                    session_path = self.directory +'/' + file +'/' + session
                    self.session = session
                    mask_exists = False
                    for mask_file in os.listdir(self.mask_dir):
                        if mask_file == f'{self.subject}_{self.session}_T1_mask.nii.gz':
                            print(f'{self.subject}_{self.session}_T1_mask.nii.gz')
                            mask_exists = True
                    if mask_exists:
                        continue
                    for scan_type in os.listdir(session_path):
                        try:
                            if scan_type =='anat':
                                for scan in os.listdir(session_path + '/' + scan_type):
                                    if scan.endswith('nii.gz') and 'auxiliary' not in scan and 'T1' in scan:
                                        print(scan)
                                        self.convert_to_array(session_path + '/' + scan_type + '/' + scan)
                        except Exception as e:
                            print(e)

    def predict(self,input, orig_dim, orig_inp):
        ind_outputs = {'first':[],'second':[],'third':[]}
        ind_outputs['first'] = self.restored_model1.predict(input)
        ind_outputs['second'] = self.restored_model2.predict(input)
        ind_outputs['third'] = self.restored_model3.predict(input)
        averaged_output = (ind_outputs['first'] + ind_outputs['second'] + ind_outputs['third'] )/3

        final_image = copy.deepcopy(averaged_output)
        #final_image = self.restored_model3.predict(input)

        for img in range(0,len(final_image)):
            for row in range(0,len(final_image[img])):
              for pixel in range(0,len(final_image[img][row])):
                if final_image[img][row][pixel] < 0.5:
                  final_image[img][row][pixel] = 0
                else:
                  final_image[img][row][pixel] = 1
        print(orig_dim)
        fin_out = np.array(cv2.resize(final_image[0], tuple(reversed(orig_inp.shape)),\
        interpolation=cv2.INTER_LINEAR))

        def fill_holes(binary_mask, structure=None, iterations=1):
            filled_mask = binary_fill_holes(binary_mask)
            
            return filled_mask.astype(np.uint8)

        fin_out = fill_holes(fin_out)

        return fin_out
      
