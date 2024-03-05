"""import tensorflow as tf

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    print(gpu)"""

import tensorflow as tf


# Assuming you want to use the second GPU (ID 1)
gpu_to_use = 0  

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Specify the GPU to be used
        tf.config.experimental.set_visible_devices(gpus[gpu_to_use], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_to_use], True)

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)



import nibabel as nib
from PIL import Image
import random
from tensorflow.keras import layers, models
import numpy as np
import cv2
import tarfile
import os
import matplotlib.pyplot as plt
import pydicom
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.losses import binary_crossentropy
import math
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate,zoom
import cv2

import copy
from scipy.ndimage import distance_transform_edt as distance
from keras import backend as K

class train_data():
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = np.prod(img_shape)
        print('-------------')
        print(self.latent_dim)
        self.width = 256
        self.height = 256

    def run_script(self):
        """self.sample_scan = np.load('mri_scan_sample.npy')
        self.sample_scan_slices = []

        for slice_count in range(0,len(self.sample_scan)):
          try:
            self.sample_scan_slices.append(np.pad(self.sample_scan[slice_count,:, :][::-1, :], ((0, 0), (0, 80)), mode='constant', constant_values=0).reshape(self.width, self.height, 1))
            self.sample_scan_slices.append(np.pad(self.sample_scan[:,slice_count, :], ((0, 16), (0, 80)), mode='constant', constant_values=0).reshape(self.width, self.height, 1))
            self.sample_scan_slices.append(np.pad(self.sample_scan[:, :,slice_count][:, ::-1], ((0, 16), (0, 0)), mode='constant', constant_values=0).reshape(self.width, self.height, 1))
          except Exception as e:
            print(e)

        self.sample_scan_slices = self.normalize_images(self.sample_scan_slices)
        self.prediction_tests()"""

        """self.masked_images = np.load('subject_1/mask_array.npy')
        self.masked_images = self.masked_images
        self.mri_images = np.load('subject_1/mri_array.npy')
        self.mri_images = self.mri_images """
        #self.generator = self.build_custom_dilated_variable_filter_generator('elu')

        custom_objects = {
            'combined_dice_bce_loss':self.combined_dice_bce_loss,
            'dice_loss':self.dice_loss,
            'boundary_loss':self.boundary_loss,
            'combined_loss':self.combined_loss
        }
        self.generator = load_model('/content/drive/MyDrive/dice_loss_prioritized.h5',custom_objects=custom_objects)
        self.generator.compile(loss=self.combined_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5),metrics=[self.dice_loss])
        self.print_gpu_memory_usage()
        print(self.generator.summary())
        self.print_gpu_memory_usage()
        

        #self.generator = self.load_old_model()
        #self.generator.load_weights('/content/drive/MyDrive/final_model_abstract_swish_activation.h5')

        self.train(150)
        #self.test_sobel()

    def normalize_images(self,images):
      normalized_images = []
      s = []
      max = 0
      for img in images:
        if np.max(img) > max:
          max = np.max(img)
      for img in range(0,len(images)):
          images[img] = images[img].astype(np.float32)
          if np.max(images[img]) >0:
            normalized_img = images[img]/np.max(images[img])
          else:
            normalized_img = images[img]
          normalized_images.append(normalized_img)

      return np.array(normalized_images)

    def boundary_loss(self,y_true, y_pred):
      sobel_filter = tf.image.sobel_edges
      y_true_edges = sobel_filter(y_true)
      y_pred_edges = sobel_filter(y_pred)
      loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

      return loss


    def surface_loss_keras(self,y_true, y_pred):
        def calc_dist_map(seg):
          res = np.zeros_like(seg)
          posmask = seg.astype(np.bool)

          if posmask.any():
              negmask = ~posmask
              res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

          return res


        def calc_dist_map_batch(y_true):
            y_true_numpy = y_true.numpy()
            return np.array([calc_dist_map(y)
                            for y in y_true_numpy]).reshape(y_true.shape).astype(np.float32)

        y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                        inp=[y_true],
                                        Tout=tf.float32)
        multipled = y_pred * y_true_dist_map
        return K.mean(multipled)


    def combined_loss(self,y_true, y_pred, alpha=0.5, beta=1.5, gamma=0.5):
      bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
      d_loss = self.dice_loss(y_true, y_pred)
      boundary = self.boundary_loss(y_true, y_pred)

      # The weights alpha, beta, and gamma determine the contribution of each loss
      total_loss = (alpha * bce) + (beta * d_loss) + (gamma * boundary)
      return total_loss


    def test_sobel(self):
      data_dir ='/content/drive/MyDrive/augmented_training_sets'
      subjects = [os.path.join(data_dir, 'suject_' + str(i)) for i in range(1, 101)]
      while True:  # Loop indefinitely
          np.random.shuffle(subjects)
          for subject in subjects:
              mri_path = os.path.join(subject, 'mri_array.npy')
              mask_path = os.path.join(subject, 'mask_array.npy')
              mri_data = np.load(mri_path)
              mask_data = np.load(mask_path)
              mask = mask_data[150]
              # Apply Sobel filter in both horizontal and vertical directions
              sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
              sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

              # Calculate the gradient magnitude
              gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
              gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)

              # (Optional) Apply a threshold to visualize the edges better
              _, edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
              # Display the edges
              plt.figure(figsize=(10, 5))
              plt.subplot(1, 2, 1)
              plt.title('Original Mask')
              plt.imshow(mask_data[150], cmap='gray')
              plt.subplot(1, 2, 2)
              plt.title('Detected Edges')
              plt.imshow(edges, cmap='gray')
              plt.show()


    def training_generator(self,data_dir, batch_size,first_sub,last_sub):
      subjects = [os.path.join(data_dir, 'subject_' + str(i)) for i in range(first_sub, last_sub)]
      while True:  # Loop indefinitely
          np.random.shuffle(subjects)
          for subject in subjects:
              mri_path = os.path.join(subject, 'mri_array.npy')
              mask_path = os.path.join(subject, 'mask_array.npy')
              mri_data = np.load(mri_path)
              mask_data = np.load(mask_path)
              permutation = np.random.permutation(len(mri_data))
              mri_data = mri_data[permutation]
              mask_data = mask_data[permutation]
              for i in range(0, len(mri_data), batch_size):
                  mri_batch = mri_data[i:i + batch_size]
                  mask_batch = mask_data[i:i + batch_size]
                  yield (np.array(mri_batch), np.array(mask_batch))


    def dice_loss(self, y_true, y_pred):
      smooth = 1e-6
      y_true_f = tf.reshape(y_true, [-1])
      y_pred_f = tf.reshape(y_pred, [-1])
      intersection = tf.reduce_sum(y_true_f * y_pred_f)
      dice_score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
      return 1 - tf.clip_by_value(dice_score, 0, 1)  # Ensures loss is non-negative

    def combined_dice_bce_loss(self, y_true, y_pred, alpha=0.5):
        dice_loss = self.dice_loss(y_true, y_pred)
        bce_loss = binary_crossentropy(y_true, y_pred)
        combined_loss = (alpha * dice_loss) + ((1 - alpha) * bce_loss)
        return combined_loss

    def print_gpu_memory_usage(self,device_name='/GPU:1'):
        """
        Prints the current GPU memory usage.
        
        Parameters:
        - device_name: str, the name of the device for which to print memory usage.
        """
        return ''
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    memory_info = tf.config.experimental.get_memory_info(device_name)
                    print(f"Current memory usage on {device_name}:")
                    print(f"- Total memory: {memory_info['current'] / (1024 ** 3):.2f} GB")
                    print(f"- Free memory: {memory_info['peak'] / (1024 ** 3):.2f} GB")
            except RuntimeError as e:
                print(e)
        else:
            print("GPU device not found.")


    def build_custom_dilated_variable_filter_generator(self,activ_func):
          def dilation_block(input):
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(input)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            for dilation_rate in range(1, 16, 1):
                print(dilation_rate)
                filter_size = 64
                output = layers.Conv2D(filter_size, (3, 3),\
                dilation_rate=dilation_rate, padding='same')(output)
                output = layers.BatchNormalization()(output)
                output = layers.Activation(activ_func)(output)
            return output

          def max_pooling_block(input):
              output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(input)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.concatenate([output,dilated_output])
              output = layers.MaxPooling2D((2, 2))(output)
              output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)

              return output

          def conv_block(input,filter_size):
              output = layers.Conv2D(64,filter_size, dilation_rate=1, padding='same')(input)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.concatenate([output,dilated_output,max_pooling_output])
              output = layers.Conv2D(128,filter_size, dilation_rate=1, padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)
              output = layers.Conv2D(64,filter_size, dilation_rate=1, padding='same')(output)
              output = layers.BatchNormalization()(output)
              output = layers.Activation(activ_func)(output)

              return output


          inputs = layers.Input(shape=(self.width, self.height, 1))

          dilated_output = dilation_block(inputs)
          max_pooling_output = max_pooling_block(inputs)
          small_filter_output = conv_block(inputs,(3,3))


          merged_outputs = layers.concatenate(\
          [small_filter_output,dilated_output,max_pooling_output])

          final_layer = layers.Conv2D(64, (3, 3), padding='same')(merged_outputs)

          outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_layer)
          model = models.Model(inputs=[inputs], outputs=[outputs])
          return model




    def invert_t1_list(self,t1_list):
      modified_list = []
      for slice_array in t1_list:
        modified_list.append(self.invert_contrast(slice_array))
      return modified_list

    def rescale_image(self, image, new_min=0, new_max=255):
       rescaled_image = (image * self.global_std) + self.global_mean
       return rescaled_image

    def show_images(self, real_images, generated_images, num_examples=5):
        plt.figure(figsize=(10, 4))
        for i in range(num_examples):
            plt.subplot(2, num_examples, i + 1)
            plt.imshow(real_images[i], cmap='gray')
            plt.axis('off')
            plt.subplot(2, num_examples, num_examples + i + 1)
            plt.imshow(generated_images[i], cmap='gray')
            #plt.imshow(generated_images[i].reshape(320, 320), cmap='gray')
            plt.axis('off')
        plt.show()


    def train(self, epochs, batch_size=8, save_interval=20, num_examples=10):
        checkpoint_cb_best = tf.keras.callbacks.ModelCheckpoint(\
        'dice_loss_prioritized_best_only.h5',\
        save_best_only=True)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(\
        'dice_loss_prioritized_every_epoch.h5',\
        save_best_only=False)
        batch_size = 8
        self.print_gpu_memory_usage()

        #self.generator.fit(self.mri_images,self.masked_images,epochs=100,batch_size=batch_size,callbacks=[checkpoint_cb],validation_split = 0.2,shuffle=True)
        train_data_dir = '/data/pnlx/projects/mysell_masking_cnn/training_set'
        val_data_dir = '/data/pnlx/projects/mysell_masking_cnn/training_set'
        batch_size = 8
        total_training_samples = 281600
        train_gen = self.training_generator(train_data_dir, batch_size,1,101)
        val_gen = self.training_generator(val_data_dir, batch_size,101,116)
        csv_logger = CSVLogger('training_log.csv', append=True, separator=',')

        self.generator.fit(
            train_gen,
            steps_per_epoch=total_training_samples // batch_size,
            epochs=100,
            validation_data=val_gen,
            validation_steps=28160 // batch_size,
            callbacks=[checkpoint_cb,csv_logger,checkpoint_cb_best]
        )



img_shape = (256, 256, 1)
trainer = train_data(img_shape)

trainer.run_script()
