import numpy as np

import tensorflow as tf
import nibabel as nib
from PIL import Image
import random
from tensorflow.keras.models import load_model


#plan
#Test each slice for each scan for each test set
# collect total dice coefficients and create histograms of avrge performance per slice
# test sets will consist of normal unmodified scans,
# normal scans with normal variations, normal scans with minor to moderate artificial artifacts
# real mp2rage scans, synthetic mp2rage scans, 
# normal scans from other datasets, scans with extreme synthetic augmentations/artifacts


class EvaluateModel():
    def __init__(self):
        self.model_path = ''

    def run_script(self):
        self.load_model()

    def load_model(self):
        pass


if __name__ == '__main__':
    EvaluateModel().run_script()
