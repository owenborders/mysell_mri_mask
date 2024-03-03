#testing on test set
import tensorflow as tf
import nibabel as nib
from PIL import Image
import random
from tensorflow.keras import layers, models
import numpy as np
import cv2
import tarfile
import os
import matplotlib.pyplot as plt
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GaussianNoise

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

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')

def dice_loss( y_true, y_pred):
  smooth = 1e-6  # Small constant to avoid division by zero
  y_true_f = tf.reshape(y_true, [-1])
  y_pred_f = tf.reshape(y_pred, [-1])
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def iou_loss(y_true, y_pred):
    smooth = 1e-6  # Small constant to avoid division by zero
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return 1 - (intersection + smooth) / (union + smooth)
def boundary_loss(y_true, y_pred):
  sobel_filter = tf.image.sobel_edges
  y_true_edges = sobel_filter(y_true)
  y_pred_edges = sobel_filter(y_pred)
  loss = tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

  return loss


def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=0.5):
  bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
  d_loss = dice_loss(y_true, y_pred)
  boundary = boundary_loss(y_true, y_pred)

  # The weights alpha, beta, and gamma determine the contribution of each loss
  total_loss = (alpha * bce) + (beta * d_loss) + (gamma * boundary)
  return total_loss

def rotate_image(img,angle,reverse = False):
    random_angle = angle

    # Rotate the image using scipy.ndimage.rotate
    # 'mode' is set to 'wrap'; you can change this as needed (e.g., 'reflect', 'nearest', etc.)
    if not reverse:
      img = rotate(img, random_angle, reshape=False, mode='constant',cval=0,order=1)
    else:
      img = rotate(img, random_angle * -1, reshape=False, mode='constant',cval=0,order=1)
    return img


def combined_dice_bce_loss( y_true, y_pred, alpha=0.5):
    # Calculate Dice loss
    dice_loss = self.dice_loss(y_true, y_pred)

    # Calculate Binary Cross-Entropy loss
    bce_loss = binary_crossentropy(y_true, y_pred)

    # Combine the losses
    combined_loss = (alpha * dice_loss) + ((1 - alpha) * bce_loss)
    return combined_loss
def change_brain_contrast(img,contrast = 'default'):

    img = np.array(img)
    # Calculate mean intensity of the brain region
    mean_intensity = np.mean(img)
    original_dtype = img.dtype

    # Random factor for contrast change, between 0.5 (decrease contrast) and 1.5 (increase contrast)
    if contrast == 'default':
        contrast_factor = random.uniform(0.5, 2.8)
    else:
        contrast_factor = contrast


    # Adjust contrast
    # Subtract mean, scale by the factor, and add the mean back
    #print(f"Original mean intensity: {mean_intensity}, Contrast factor: {contrast_factor}")

    img= (img- mean_intensity) * contrast_factor + mean_intensity

    # Clip the values to ensure they stay within the range 0 to 1
    np.clip(img, 0, 1, out=img)
    return img

custom_objects = {
    'combined_dice_bce_loss':combined_dice_bce_loss,
    'dice_loss':dice_loss,
    'combined_loss':combined_loss

}
t1_test_set = np.load(f'/content/drive/MyDrive/transfer_learning_test_sets/mri_array.npy')
mask_test_set = np.load(f'/content/drive/MyDrive/transfer_learning_test_sets/mask_array.npy')

t1_test_set = np.array([i.reshape(256, 256,1) for i in t1_test_set])

mask_test_set = np.array([i.reshape(256, 256,1) for i in mask_test_set])
print(t1_test_set.shape)
print(np.max(t1_test_set))
restored_model2 = load_model('/content/drive/MyDrive/final_model_abstract_swish_activation.h5',custom_objects=custom_objects)
restored_model1 = load_model('/content/drive/MyDrive/final_model_abstract_elu_activation.h5',custom_objects=custom_objects)

restored_model3 = load_model('/content/drive/MyDrive/final_model_abstract.h5',custom_objects=custom_objects)
restored_model4 = load_model('/content/drive/MyDrive/final_model_abstract_leakyrelu_activation.h5',custom_objects=custom_objects)
restored_model5 = load_model('/content/drive/MyDrive/final_model_abstract_elu_activation.h5',custom_objects=custom_objects)


test_dataset = tf.data.Dataset.from_tensor_slices((t1_test_set, mask_test_set))
test_dataset = test_dataset.batch(8)  # Replace 32 with your batch size
def apply_threshold(preds, threshold=0.5):
    return tf.cast(tf.greater(preds, threshold), tf.float64)
t1_test_set = []
mask_test_set = []

dice_losses = []
for batch in test_dataset:
    images, labels = batch


    # Predict with each model
    #preds1 = restored_model1.predict(images)
    all_predictions = []

    for rotation in [90,180,270]: #[90,180,270]:
      for contrast in [1]: #[1,1.05,1.075,1.1]:
        mod_images = []
        for image in images:
          rotated_img =rotate_image(change_brain_contrast(image,contrast),rotation)
          mod_images.append(rotated_img)

        mod_images = np.array(mod_images)
        """preds2 = restored_model2.predict(mod_images)
        for pred in range(0,len(preds2)):
          preds2[pred] = rotate_image(preds2[pred],rotation,True)
        all_predictions.append(preds2)"""
        preds5 = restored_model5.predict(mod_images)
        for pred in range(0,len(preds5)):
          preds5[pred] = rotate_image(preds5[pred],rotation,True)
      all_predictions.append(preds5)
    #preds3 = restored_model3.predict(images)
    #preds4 = restored_model4.predict(images)


    # Average the predictions
    avg_preds = np.mean(all_predictions, axis=0)

    # Apply threshold
    thresholded_predictions = apply_threshold(avg_preds)
    """for x in range(0,8):
      plt.subplot(3,1,1)
      plt.imshow(thresholded_predictions[x], cmap='gray')
      plt.subplot(3,1,2)
      plt.imshow(images[x], cmap='gray')
      plt.subplot(3,1,3)
      plt.imshow(labels[x], cmap='gray')
      plt.show()"""

    # Compute Dice loss
    loss = dice_loss(labels, thresholded_predictions)
    dice_losses.append(loss.numpy())


# Calculate the average Dice loss over all batches
average_dice_loss = np.mean(dice_losses)
print(f"Average Dice Loss on the Test Set: {average_dice_loss}")
