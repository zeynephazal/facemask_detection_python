import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import sys
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
# from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
import matplotlib.pyplot as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report