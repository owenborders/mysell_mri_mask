import numpy as np

import tensorflow as tf
import nibabel as nib
from PIL import Image
import random
from tensorflow.keras.models import load_model

class EvaluateModel():
    def __init__(self):
        self.model_path = ''

    def run_script(self):
        self.load_model()

    def load_model(self):
        pass


if __name__ == '__main__':
    EvaluateModel().run_script()
