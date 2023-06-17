import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard

# VGG19 forward
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import numpy as np
import os
import sys
import argparse
import cv2
import glob
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

vgg19 = VGG19(weights='imagenet', include_top=False)

# process image
def process_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

image = cv2.imread('/Users/precious/Intelligent-Computing-Systems.git/VGG/Beautiful_demoiselle_(Calopteryx_virgo)_male_3.jpg')
image = process_image(image)

import time
start = time.time()
features = vgg19.predict(image)
end = time.time()

print("Time: ", end - start)
