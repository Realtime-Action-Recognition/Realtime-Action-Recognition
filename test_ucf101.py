# ==============================================================================================
# Title:   test_ucf101.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

import numpy as np
# Import to load pre-trained model
from keras.applications.densenet import DenseNet121

# Imports for callbacks during training
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Imports for the layers used
from keras.layers import Concatenate, Flatten, Dense
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Import to create the model
from keras.models import Model, load_model

# Import Optimizers for the model
from keras.optimizers import Adam, SGD

import _pickle as cPickle

# Import for real-time plotting of the Training Loss and Training Accuracy
# from livelossplot.keras import PlotLossesCallback

# Save in current Folder by default
res = [0, 5]
model_save_path = "."

import argparse
parser = argparse.ArgumentParser(description='Command-line parameters to test the Three-Stream Algorithm on UCF-101')

parser.add_argument('model_path', metavar='mp', type=str, default="UCF_Trained_Model_for_Paper_1929_05.h5", help='Enter the model path')
parser.add_argument('batch_size', metavar='bs', type=int, default=32, help='Enter batch size')
command_line_args = parser.parse_args()

test_hog_data = []
test_temporal_data = []
test_spatial_data = []
test_labels = []

def load_testing_data():
    global test_hog_data
    global test_temporal_data
    global test_spatial_data
    global test_labels

    rfp = open('ucf101_split_1_hog_data.pyc', 'rb')
    test_hog_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_temporal_data.pyc', 'rb')
    test_temporal_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_spatial_data.pyc', 'rb')
    test_spatial_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_train_labels.pyc', 'rb')
    test_labels = cPickle.load(rfp)
    rfp.close()

load_testing_data()

X_test_combined = np.stack((test_hog_data, test_temporal_data, test_spatial_data), axis=4)
X_test_hog = X_test_combined[:,:,:,:,0]

X_test_temporal = X_test_combined[:,:,:,:,1]

X_test_spatial = X_test_combined[:,:,:,:,2]

X_test_hog = [cv2.merge((i, i, i)) for i in X_test_hog]
X_test_temporal = [cv2.merge((i, i, i)) for i in X_test_temporal]
X_test_spatial = [cv2.merge((i, i, i)) for i in X_test_spatial]

model_name = command_line_args.model_path
model = load_model(model_name, compile=False)

res += model.evaluate([np.array(X_test_hog), np.array(X_test_temporal), np.array(X_test_spatial)], test_labels, batch_size=32)
print(model_name, 'Test Loss, Test Accuracy:', res)