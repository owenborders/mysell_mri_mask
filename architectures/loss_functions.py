
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.losses import binary_crossentropy


class LossFunctions():

    def __init__(self):
        pass


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


