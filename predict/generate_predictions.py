import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import copy
print(tf.__version__)

class GenerateMasks():
    def __init__(self,directory,subject_list):
        self.directory = directory 
        
        self.subject_list =subject_list
        
        def dice_loss( y_true, y_pred):
          smooth = 1e-6  # Small constant to avoid division by zero
          y_true_f = tf.reshape(y_true, [-1])
          y_pred_f = tf.reshape(y_pred, [-1])
          intersection = tf.reduce_sum(y_true_f * y_pred_f)
          return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

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

          # The weights alpha, beta, and gamma determine the contribution of each loss
          total_loss = (alpha * bce) + (beta * d_loss) + (gamma * boundary)
          return total_loss


        custom_objects = {
            'combined_dice_bce_loss':combined_dice_bce_loss,
            'dice_loss':dice_loss,
            'boundary_loss':boundary_loss,
            'combined_loss':combined_loss
        }


        self.restored_model1 = load_model(\
            '/content/drive/MyDrive/final_model_abstract_swish_activation.h5',\
            custom_objects=custom_objects)
        self.restored_model2 = load_model(\
            '/content/drive/MyDrive/boundary_loss_prioritized.h5',\
            custom_objects=custom_objects)

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
        filename = f'{self.subject}_{self.session}_desc-T1wXcMabsQc_mask.nii.gz'
        if filename not in os.listdir(self.scan_folder):
            nib.save(img, self.scan_folder + '/' + filename )
        else:
            print(self.subject)
            print(filename)
            print('file alrdy exists')

    def normalize(self,image):
        if np.min(image) < 0:
            print(np.min(image))
        if np.max(image) >0:
            image = image/np.max(image)
        return image

    def convert_to_array(self,path):
        nii_file = nib.load(path)
        nii_data = nii_file.get_fdata()
        nii_data_array = np.array(nii_data, dtype=np.float64)
        slice_total = len(nii_data_array)
        self.combined_slices = np.zeros((slice_total, slice_total, slice_total))

        print(slice_total)
        self.compiled_slices = np.zeros((240,256,176))

        for slice_count in range(100,256):
            print(slice_count)
            # sagittal = [:,:,x] size 240
            # axial = [:,x,:] size 256
            #coronal = [x,:,:]
            view_list =  ['coronal','axial','sagittal']

            for view in range(0,len(view_list)):
                #print(slice_count)
                batch = []

                if view == 0 and slice_count < 240:
                    batch.append(self.normalize(self.pad_slice_to_square(nii_data_array[slice_count,:,:]).reshape(256,256,1)))
                    batch = np.array(batch)
                    output = self.predict(np.array(batch))
                    reverse_padded_output = self.reverse_padding(output[0])

                    self.compiled_slices[slice_count,:,:] = reverse_padded_output
                elif view == 1 and slice_count < 256:
                    batch.append(self.normalize(self.pad_slice_to_square(nii_data_array[:,slice_count,:]).reshape(256,256,1)))
                    batch = np.array(batch)
                    output = self.predict(np.array(batch))
                    reverse_padded_output = self.reverse_padding(output[0])
                    self.compiled_slices[:,slice_count,:] = reverse_padded_output
                elif view == 2 and slice_count < 176:
                    batch.append(self.normalize(self.pad_slice_to_square(nii_data_array[:,:,slice_count]).reshape(256,256,1)))
                    batch = np.array(batch)
                    output = self.predict(np.array(batch))
                    reverse_padded_output = self.reverse_padding(output[0])
                    self.compiled_slices[:,:,slice_count] = reverse_padded_output

        #self.save_as_nii()

    def read_files(self):
        for subject in os.listdir(self.directory):
            #if subject in self.subject_liste
            if not any(x in subject for x in self.subject_list):
                continue
            for session in  os.listdir(self.directory + '/' + subject):
                for scan_type in os.listdir(self.directory + '/'\
                + subject + '/' + session):
                    scan_folder = self.directory + '/' +\
                    subject + '/' + session + '/' + scan_type
                    for file in os.listdir(scan_folder):
                        print(file)
                        if 'Xc_T1w' in file and 'mask' not in file and file.endswith('nii.gz'):
                            if 't1' in file.split('_')[0]:
                                self.subj_suffix = 't1'
                            elif 't0' in file.split('_')[0]:
                                self.subj_suffix = 't0'
                            self.original_t1_file = scan_folder + '/'+ file
                            self.scan_folder = scan_folder
                            self.session = session
                            self.subject = subject
                            self.convert_to_array(scan_folder + '/' + file)

    def predict(self,input):

        first_output = self.restored_model1.predict(input)
        second_output =  self.restored_model2.predict(input)

        averaged_output = (first_output +second_output)/2

        final_image = copy.deepcopy(averaged_output)

        for img in range(0,len(final_image)):
            for row in range(0,len(final_image[img])):
              for pixel in range(0,len(final_image[img][row])):
                if final_image[img][row][pixel] < 0.5:
                  final_image[img][row][pixel] = 0
                else:
                  final_image[img][row][pixel] = 1

        plt.imshow(first_output[0],cmap='gray')
        plt.show()
        plt.imshow(second_output[0],cmap='gray')
        plt.show()
        plt.imshow(final_image[0],cmap='gray')
        plt.show()
        plt.imshow(input[0],cmap='gray')
        plt.show()

        return final_image



if __name__ =='__main__':
    GenerateMasks().read_files()
