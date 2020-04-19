# ==============================================================================================
# Title:   train_ucf101.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

# Import to load pre-trained model
import cv2
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import _pickle as cPickle

from three_stream_model import ThreeStreamModel

parser = argparse.ArgumentParser(description='Command-line parameters to train the Three-Stream Algorithm on UCF-101')
parser.add_argument('-bs','--batch_size', default=32, help='The batch size to use while training. (default=32)')
parser.add_argument('-e','--epochs', default=100, help='Number of epochs to train. (default=100)')
parser.add_argument('-oc','--output_classes', default=101, help='Number of output classes. (default=101)')
parser.add_argument('-lr','--learning_rate', default=0.000001, help='Learning rate of the algorithm. (default=0.000001)')

command_line_args = parser.parse_args()

three_stream_model = ThreeStreamModel(output_classes=int(command_line_args.output_classes),
                    input_learning_rate=float(command_line_args.learning_rate),
                    input_epochs=int(command_line_args.epochs))

train_hog_data = []
train_temporal_data = []
train_spatial_data = []
train_labels = []

def load_training_data():
    global train_hog_data
    global train_temporal_data
    global train_spatial_data
    global train_labels

    rfp = open('ucf101_split_1_hog_data.pyc', 'rb')
    train_hog_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_temporal_data.pyc', 'rb')
    train_temporal_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_spatial_data.pyc', 'rb')
    train_spatial_data = cPickle.load(rfp)
    rfp.close()

    rfp = open('ucf101_split_1_train_labels.pyc', 'rb')
    train_labels = cPickle.load(rfp)
    rfp.close()

load_training_data()

def process_training_data():
    X_train_combined = np.stack((train_hog_data, 
                                train_temporal_data, 
                                train_spatial_data), axis=4)

    X_train, X_test, Y_train, Y_test = train_test_split(X_train_combined,
                                        train_labels, test_size = 0.2, random_state= 42)

    X_train_hog = X_train[:,:,:,:,0]
    X_test_hog = X_test[:,:,:,:,0]

    X_train_temporal = X_train[:,:,:,:,1]
    X_test_temporal = X_test[:,:,:,:,1]

    X_train_spatial = X_train[:,:,:,:,2]
    X_test_spatial = X_test[:,:,:,:,2]

    if len(train_hog_data[0][0][0] == 1):
        X_train_hog = [cv2.merge((i, i, i)) for i in X_train_hog]
        X_test_hog = [cv2.merge((i, i, i)) for i in X_test_hog]

    if len(train_temporal_data[0][0][0] == 1):
        X_train_temporal = [cv2.merge((i, i, i)) for i in X_train_temporal]
        X_test_temporal = [cv2.merge((i, i, i)) for i in X_test_temporal]

    if len(train_spatial_data[0][0][0] == 1):
        X_train_spatial = [cv2.merge((i, i, i)) for i in X_train_spatial]
        X_test_spatial = [cv2.merge((i, i, i)) for i in X_test_spatial]

    return  X_train_hog, X_test_temporal, X_test_spatial, Y_train, \
            X_test_hog, X_test_temporal, X_test_spatial, Y_test


def train_model():
    X_train_hog, X_train_temporal, X_train_spatial, Y_train, \
    X_test_hog, X_test_temporal, X_test_spatial, Y_test = process_training_data()

    three_stream_model.train_model(X_train_hog, X_train_temporal, X_train_spatial,
                                    Y_train, X_test_hog, X_test_temporal,
                                    X_test_spatial, Y_test, 
                                    input_epochs=int(command_line_args.epochs),
                                    input_batch_size=int(command_line_args.batch_size))