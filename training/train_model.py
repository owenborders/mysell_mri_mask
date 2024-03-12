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
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GaussianNoise
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

    def configure_gpu(self,gpu_to_use : int = 0) -> None:
        """
        Sets up GPU that tensorflow 
        will use to train model.

        Parameters
        -------------
        gpu_to_use: int
            index of GPU
        """
        gpus = tf.config.experimental.\
        list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(\
                gpus[gpu_to_use], 'GPU')
                logical_gpus = tf.config.experimental.\
                list_logical_devices('GPU')
                tf.config.experimental.\
                set_memory_growth(gpus[gpu_to_use], True)
                print(len(gpus), "Physical GPUs,", \
                len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                print(e)

    def run_script(self,load_old_model : bool = False) -> None:
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
            '/data/pnlx/projects/mysell_masking_cnn/dice_loss_prioritized.h5')
        else:
            self.generator = \
            self.architectures.hybrid_dilated_model('elu',256,256)
            self.generator.compile(loss=self.combined_loss,\
            optimizer=tf.keras.optimizers.Adam(\
            learning_rate=0.002,beta_1=0.5),\
            metrics=[self.dice_loss])

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

        custom_objects = {
            'combined_dice_bce_loss':self.combined_dice_bce_loss,
            'dice_loss':self.dice_loss,
            'boundary_loss':self.boundary_loss,
            'combined_loss':self.combined_loss
        }
        model = load_model(\
        path_to_model,\
        custom_objects=custom_objects)

        return model
        
    def boundary_loss(self,y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
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
        y_true_edges = sobel_filter(y_true)
        y_pred_edges = sobel_filter(y_pred)
        loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

        return loss

    def combined_loss(self,y_true : tf.Tensor,
        y_pred : tf.Tensor, bce_weight : float = 0.5,
        dice_weight : float = 1.5, bound_weight : float = 0.5
    ) -> tf.Tensor:
        """
        Loss function that combined binary cross-entropy,
        dice coefficient, and boundary loss.

        Parameters
        ------------------
        y_true : tf.Tensorf
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

        bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        d_loss = self.dice_loss(y_true, y_pred)
        boundary = self.boundary_loss(y_true, y_pred)

        total_loss = (bce_weight * bce) + (dice_weight * d_loss)\
        + (bound_weight * boundary)
        return total_loss

    def training_generator(
        self, data_dir:str, batch_size:int,
        first_sub:int, last_sub:int
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

        subjects = [os.path.join(data_dir, 'subject_' + str(i)) \
        for i in range(first_sub, last_sub)]
        while True: 
            np.random.shuffle(subjects)
            for subject in subjects:
                mri_data = self.load_and_randomize(subject, 'mri_array.npy')
                mask_data = self.load_and_randomize(subject, 'mask_array.npy')
                for i in range(0, len(mri_data), batch_size):
                    mri_batch = mri_data[i:i + batch_size]
                    mask_batch = mask_data[i:i + batch_size]
                    combined_batch = (np.array(mri_batch), np.array(mask_batch))
                    yield combined_batch

    def load_and_randomize(self,subject_folder:str, array_filename:str) -> np.ndarray:
        """
        Loads array from specified filepath
        and randomizes the order of it.

        Parameters
        --------------
        subject_folder : str
            Folder for current subject
        array_filename : str
            Name of file containing array
        
        Returns
        --------------
        array_data : np.ndarray
            Loaded array with randomized order
        """
        filepath = os.path.join(subject_folder, array_filename)
        array_data = np.load(filepath)
        permutation = np.random.permutation(len(array_data))
        array_data = array_data[permutation]

        return array_data

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

        return dice_score

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
        'dice_loss_prioritized_best_only.h5',\
        save_best_only=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(\
        'dice_loss_prioritized_every_epoch.h5',\
        save_best_only=False)
        sub_data_dir = '/data/pnlx/projects/mysell_masking_cnn/training_set'
        total_training_samples = 281600
        train_gen = self.training_generator(sub_data_dir, batch_size,1,101)
        val_gen = self.training_generator(sub_data_dir, batch_size,101,116)
        csv_logger = CSVLogger('training_log.csv', append=True, separator=',')
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
