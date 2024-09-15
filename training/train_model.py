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

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.losses import binary_crossentropy

import math
import os
from scipy.ndimage import rotate,zoom
import copy
from scipy.ndimage import distance_transform_edt as distance
from keras import backend as K
from model_architectures import Architectures

class TrainModel():
    def __init__(self):
        self.architectures = Architectures()
        self.configure_gpu()

    def configure_gpu(self, gpu_to_use : int = 0) -> None:
        """
        Sets up GPU that tensorflow 
        will use to train model.

        Parameters
        -------------
        gpu_to_use: int
            index of GPU
        """
        gpus = tf.config.experimental. \
        list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=61440)]
                    )
                tf.config.experimental.set_visible_devices( \
                gpus[gpu_to_use], 'GPU')
            except RuntimeError as e:
                print(e)

    def run_script(self, load_old_model : bool = True) -> None:
        """
        Defines architecture and 
        calls training functions.

        Parameters
        -------------
        load_old_model: bool
            If set to True,
            an existing model will be loaded
        """

        if load_old_model:
            self.generator = self.load_trained_model(\
            '/data/pnlx/projects/mysell_masking_cnn/mask_attn_stand_attndrop_every_epoch.h5')
            self.generator.compile(
                loss=self.combined_loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5),
                metrics=[self.dice_loss, self.weighted_boundary_loss]
            )
        else:
            self.generator = self.architectures.atrous_attn_unet()
            self.generator.compile(
                loss=self.combined_loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5),
                metrics=[self.dice_loss, self.weighted_boundary_loss]
            )

            print(self.generator.summary())

        self.train()

    def load_trained_model(self,path_to_model : str) -> tf.keras.models.Model:
        """
        Loads a previously trained model
        in order to continue training.

        Parameters 
        -------------
        path_to_model: str
            path to the saved model

        Returns
        ----------
        model : tf.keras.models.Model 
            loaded model
        """
        
        def loss(y_true,y_pred):
            pass
        custom_objects = {
            'dice_loss':self.dice_loss,
            'boundary_loss':self.boundary_loss,
            'combined_loss':self.combined_loss,
            'loss':loss,
            'weighted_boundary_loss': self.weighted_boundary_loss
        }
        
        model = load_model(\
        path_to_model,\
        custom_objects=custom_objects)

        return model
        
    def boundary_loss(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        """
        Loss function that prioritizes the 
        boundaries of the masks.

        Parameters
        ----------
        y_true : tf.Tensor
            The true segmentation masks.
        y_pred : tf.Tensor
            The predicted segmentation masks.

        Returns
        -------
        loss : tf.Tensor
            The calculated boundary loss.
        """

        sobel_filter = tf.image.sobel_edges

        # create outline of masks using sobel filters
        y_true_edges = sobel_filter(y_true) 
        y_pred_edges = sobel_filter(y_pred)

        # calculates mean squared error between boundaries
        loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

        return loss

    def weighted_boundary_loss(self,y_true: tf.Tensor, y_pred: tf.Tensor,\
        weight_fn: float = 3.0, weight_fp: float = 1.0\
    ) -> tf.Tensor:
        """
        Weighted boundary loss function that prioritizes
        the boundaries of the masks and emphasizes false negatives.

        Parameters
        ----------
        y_true : tf.Tensor
            The true segmentation masks.
        y_pred : tf.Tensor
            The predicted segmentation masks.
        weight_fn : float
            Weight for false negatives.
        weight_fp : float
            Weight for false positives.

        Returns
        -------
        loss : tf.Tensor
            The calculated weighted boundary loss.
        """
        
        sobel_filter = tf.image.sobel_edges
        y_true_edges = sobel_filter(y_true)
        y_pred_edges = sobel_filter(y_pred)
        edge_diff = y_true_edges - y_pred_edges
        weight_map = tf.where(y_true_edges > y_pred_edges, weight_fn, weight_fp)
        weighted_diff = weight_map * tf.square(edge_diff)
        loss = tf.reduce_mean(weighted_diff)
        
        return loss

    def combined_loss(self, y_true, y_pred, bce_weight : float = 0.5,
        dice_weight : float = 0.8, boundary_weight = 0.8
    ) -> tf.Tensor:
        """
        Loss function that combined binary cross-entropy,
        dice coefficient, and boundary loss.

        Parameters
        ------------------
        y_true : tf.Tensor
            ground truth mask
        y_pred : tf.Tensor
            predicted mask
        bce_weight : float
            weight added to binary cross-entropy
        dice_weight : float
            weight added to dice coefficient 
        bound_weight : float
            weight added to boundary loss 

        Returns
        ----------------
        total_loss : tf.Tensor
            calculated combined loss between
            predicted mask and ground truth 
        """

        bce = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25,gamma=2.0,
        from_logits=False)(y_true, y_pred)
        d_loss = self.dice_loss(y_true, y_pred)
        boundary_loss = self.weighted_boundary_loss(y_true, y_pred)

        bce = tf.reduce_mean(bce)
        d_loss = tf.reduce_mean(d_loss)
        #boundary_loss =tf.reduce_mean(boundary_loss)

        tf.print("BCE:", bce)
        tf.print("Dice Loss:", d_loss)
        tf.print("Boundary Loss:", boundary_loss)
        
        total_loss = (bce_weight * bce) + (dice_weight * d_loss) + (boundary_loss * boundary_weight)

        return total_loss

    def training_generator(
        self, data_dir : str, batch_size : int,
        first_sub : int, last_sub : int
    ) -> tuple:
        """
        Function to read data from each subject folder
        to be fed into the neural network as a training
        set or validation set. 

        Parameters
        -----------------
        data_dir : str
            directory to subject folders 
        batch_size : int
            batch size being used for training
        first_sub : int
            minimum subject in range of subjects being used 
        last_sub : int 
            maximum subject in range of subjects being used 

        Yields
        ------------------
        combined_batch : tuple
            Tuple containing a batch of MRI Data
        """

        subjects = [os.path.join(data_dir, 'subject_' + str(i)) for i in range(first_sub, last_sub)]
        while True:
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
                    mri_batch = mri_data[i:i + batch_size].astype(np.float32)
                    mask_batch = mask_data[i:i + batch_size].astype(np.float32)
                    combined_batch = (mri_batch, mask_batch)
                    yield combined_batch

    def dice_loss(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        """
        Loss function that calculates
        overlap between ground truth 
        masks and predicted masks.

        Parameters
        ----------------
        y_true : tf.Tensor
            ground truth masks
        y_pred : tf.Tensor
            predicted mask

        Returns
        ------------
        dice_score : tf.Tensor
            Dice coefficient of
            predicted and ground
            truth masks
        """
    
        smooth = 1e-6
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (\
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_score = 1 - tf.clip_by_value(dice_score, 0, 1) 
        dice_score = dice_score 

        return dice_score
    
    def boundary_dice_loss(self, y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
        """
        Loss function that calculates
        overlap between ground truth 
        masks and predicted masks.

        Parameters
        ----------------
        y_true : tf.Tensor
            ground truth masks
        y_pred : tf.Tensor
            predicted mask

        Returns
        ------------
        dice_score : tf.Tensor
            Dice coefficient of
            predicted and ground
            truth masks
        """
    
        smooth = 1e-6
        sobel_filter = tf.image.sobel_edges
        y_true_edges = sobel_filter(y_true)
        y_pred_edges = sobel_filter(y_pred)

        y_true_f = tf.reshape(y_true_edges, [-1])
        y_pred_f = tf.reshape(y_pred_edges, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (\
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_loss = 1 - tf.clip_by_value(dice_score, 0, 1)

        return dice_loss

    def train(self, num_epochs : int = 50, batch_size : int = 8) -> None:
        """
        Function to begin the training process
        for the defined architecture.

        Parameters
        -------------
        num_epochs : int
            how many times the model will
            loop through the entire training set
        batch_size : int
            how many elements from the training
            set are fed into the model at a time.
        """

        checkpoint_cb_best = tf.keras.callbacks.ModelCheckpoint(\
        'mask_attn_stand_attndrop_best_only_iteration_2.h5',\
        save_best_only=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(\
        'mask_attn_stand_attndrop_every_epoch.h5',\
        save_best_only=False)
        sub_data_dir = '/data/pnlx/projects/mysell_masking_cnn/training_set'
        total_training_samples = 281600
        train_gen = self.training_generator(sub_data_dir, batch_size,1,101)
        val_gen = self.training_generator(sub_data_dir, batch_size,101,116)
        csv_logger = CSVLogger('training_log_mask_attn_stand_attndrop.csv', append=True, separator=',')

        self.generator.fit(
            train_gen,
            steps_per_epoch=total_training_samples // batch_size,
            epochs=num_epochs,
            validation_data=val_gen,
            validation_steps=28160 // batch_size,
            callbacks=[checkpoint_cb,csv_logger,checkpoint_cb_best]
        )


if __name__ == '__main__':
    TrainModel().run_script()
