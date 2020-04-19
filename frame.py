# ==============================================================================================
# Title:   frame.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

# OS import to read Guest Operating System files and folders
import os

# OpenCV2 import for preprocessing of the individual frames
# Tested with OpenCV v3.4.4
import cv2

# Time import to track the time taken by different components
# of the system
import time

# Import pyflow (custom library) to generate optical flow
import pyflow

# Warnings import to warn user of certain pitfalls of the system
import warnings

# Numpy import for array manipulation
import numpy as np

# Import TensorFlow
import tensorflow as tf

# Randint import to randomly sample frames at the specified
# sampling rate
from random import randint

# Matplotlib import to plot the Model Training Accuracy
# and Training Loss
import matplotlib.pyplot as plt

# Keras imports for the Neural Netork Framework
import keras
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import backend as KerasBackend
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Scikit Learn import for the splitting of the dataset into
# training and testing sets
# from sklearn.model_selection import train_test_split

# Time to load and print keras warnings
time.sleep(3)

# Variable to identify individual frames
global_frame_id = 1


# class_names = {0:'STEP 6 RIGHT', 1:'STEP 7 RIGHT', 2:'STEP 7 LEFT', 3:'STEP 4 RIGHT', 4:'STEP 2 LEFT', 5:'STEP 6 LEFT', 6:'STEP 5 LEFT', 7:'STEP 3', 8:'STEP 2 RIGHT', 9:'STEP 5 RIGHT', 10:'STEP 4 LEFT', 11:'STEP 1'}

class_names = {0:'STEP 1', 1:'STEP 2 LEFT', 2:'STEP 2 RIGHT', 3:'STEP 3', 4:'STEP 4 LEFT', 5:'STEP 4 RIGHT', 6:'STEP 5 LEFT', 7:'STEP 5 RIGHT', 8:'STEP 6 LEFT', 9:'STEP 6 RIGHT', 10:'STEP 7 LEFT', 11:'STEP 7 RIGHT'}

# Loading the pre-trained prediction model
# fusion_model = load_model("final_combined_fused_model_pyflow_demo_10_1.00.h5")
model = None
graph = None


''' 
Setting Image Parameters
'''
image_height = 112
image_width = 112

def process_flow(im1, flow_vector):

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_vector[..., 0], flow_vector[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    h, s, grayscale = cv2.split(hsv)

    return(grayscale)

def crop_to_height(image_array):
    '''
    Crop the stream to select only the Region of Interest
    In our system, the region of interest is the middle of the screen
    '''
    height, width, channels = image_array.shape

    if height == width:
        return image_array

    image_array = np.array(image_array)

    assert height < width, "Height of the image is greater than width!"
    excess_to_crop = int((width - height)/2)
    cropped_image = image_array[0:height, excess_to_crop:(height+excess_to_crop)]
    return cropped_image

model = load_model("handwash_step_model.h5")
graph = tf.get_default_graph()


class Frame:
    '''
    Class to contain operations with respect to individual
    Frames
    '''
    

    # The following properties are defined as follows
    # to ensure that when called, the associated Frame
    # Object will have the most up-to-date value of the
    # attribute.
    #
    # This is done by changing a placeholder variable for
    # each property, and returning the placeholder variable's
    # value when a call is made to the real variable.
    #
    # If not implemented the way it is, the attributes of the
    # associated Frame object remains the same as the attributes
    # initialised by the __init__() function

    @property
    def pixel_values_array(self):
        return np.array(self.placeholder_pixel_values_array)


    @property
    def shape(self):
        return self.pixel_values_array.shape


    @property
    def frame_height(self):
        return self.pixel_values_array.shape[0]


    @property
    def frame_width(self):
        return self.pixel_values_array.shape[1]


    @property
    def number_of_channels(self):
        if len(self.pixel_values_array.shape) == 2:
            return 1
        return self.pixel_values_array.shape[2]


    @property
    def dense_optical_flow_vector(self):
        return self.placeholder_dense_optical_flow_vector


    @property
    def class_predicted(self):
        return self.placeholder_class_predicted


    @property
    def confidence_score(self):
        return self.placeholder_confidence_score


    # Constructor 
    def __init__(self, pixel_values_array_input):
        '''
        Initialize the Frame, to its pixel values array
        
        Parameters:
        @pixel_values_array : The value returned from cv2.imread()
        '''

        global global_frame_id

        # Try-Except block to handle empty images being passed to the class
        # try:
        #     assert not(pixel_values_array_input == None), "The frame passed was empty!"
        # except ValueError:
        #     print("Frame Object Created- FrameID: "+str(global_frame_id)+", and shape:", end="")
        
        # Assign an ID to the frame
        self.frame_id = global_frame_id
        global_frame_id = global_frame_id + 1

        # Initialise dense_optical_flow_vector to False to avoid
        # creation of a wrong Placeholder Optical Flow Vector 
        self.placeholder_dense_optical_flow_vector = False

        # placeholder_pixel_values_array is used so the variable
        # pixel_values_array always has the updated value associated
        # with the object
        self.placeholder_pixel_values_array = pixel_values_array_input

        # Predication attributes initialised to default initial values
        self.placeholder_class_predicted = -1
        self.placeholder_confidence_score = -1

        # Print Shape attributes, when Frame Object is initialized
        # print(self.frame_height,"x", self.frame_width, "x", self.number_of_channels)


    def resize_frame(self, new_image_width, new_image_height):
        '''
        Resizing an image to the new height and new width
        Parameters:
        @new_image_height : Height the image must be resized to
        @new_image_width  : Width the image must be resized to
        '''

        self.placeholder_pixel_values_array = np.array(cv2.resize(self.pixel_values_array, (new_image_height, new_image_width)))
        # print("Frame successfully resized to ", new_image_height, "x", new_image_width)


    def convert_to_grayscale(self):
        '''
        Convert the frame to grayscale from color
        '''

        assert (self.number_of_channels > 1), "The image to convert_to_grayscale is already Grayscale"

        self.placeholder_pixel_values_array = cv2.cvtColor(self.pixel_values_array, cv2.COLOR_BGR2GRAY)

    # THIS WORKS WOW
    # def check_me(self, g):
    #     print(g.hello)

    def crop_to_region(self):
        '''
        Crop the frame to select only the exitRegion of Interest
        In our system, the region of interest is the middle of the screen
        '''

        assert self.frame_height != self.frame_width, "Frame is already a cropped to region of interest."
        # assert self.pixel_values_array.shape[0] != self.pixel_values_array.shape[1], "Frame is already a cropped to region of interest."

        assert self.frame_height <= self.frame_width, "Height of the frame is greater than width!"
        # assert self.pixel_values_array.shape[0] <= self.self.pixel_values_array.shape[1], "Height of the frame is greater than width!"
        
        excess_to_crop = int((self.frame_width - self.frame_height)/2)
        cropped_image = self.pixel_values_array[0:self.frame_height, excess_to_crop:(self.frame_height+excess_to_crop)]
        
        # excess_to_crop = int((self.pixel_values_array.shape[1] - self.pixel_values_array.shape[0])/2)
        # cropped_image = self.pixel_values_array[0:self.pixel_values_array.shape[0], excess_to_crop:(self.pixel_values_array.shape[0]+excess_to_crop)]
        

        self.placeholder_pixel_values_array = cropped_image
    

    def generate_optical_flow(self, previous_frame):
        '''
        Generate the Dense Optical Flow between the current frame, and the previous frame
        Parameters:
        @previous_frame : A parameter of type Frame, that contains the previous frame
        '''

        '''
        Flow Options
        '''
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        assert (self.dense_optical_flow_vector == False), "Optical Flow already generated for the current frame."

        # Cropping the previous_frame
        previous_frame = crop_to_height(previous_frame)

        # Resizing the previous frame
        previous_frame = cv2.resize(previous_frame, (image_height, image_width))

        # Numpy Arrays
        previous_temporal_image = previous_frame.astype(float) / 255.
        next_temporal_image = self.placeholder_pixel_values_array.astype(float) / 255.

        # previous_temporal_image = previous_temporal_image.astype(np.float32)
        # next_temporal_image = next_temporal_image.astype(np.float32)

        # print("ACTUAL:")
        # print(len(self.pixel_values_array),"x",len(self.pixel_values_array[0]), "x", len(self.pixel_values_array[0][0][0]))

        # Adding new method
        u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
        temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)
        temporal_image = process_flow(previous_temporal_image, temporal_image)

        
        # temporal_image_grayscale = cv2.cvtColor(temporal_image, cv2.COLOR_BGR2GRAY)
        temporal_image_after_reshape = np.reshape(temporal_image, (1, image_width, image_height, 1))
        
        self.placeholder_dense_optical_flow_vector = temporal_image_after_reshape


    def preprocess(self, final_frame_side):
        '''
        Single function to process all the default pre-processing
        Parameters:
        @final_frame_side : A parameter of type Frame, that contains the previous frame
        '''

        self.crop_to_region()
        self.resize_frame(final_frame_side, final_frame_side)


    def show_frame(self):
        '''
        Function to show the current frame using cv2.imshow()
        '''

        print("Frame: "+str(self.frame_id)+" is being displayed.")
 
        # show the image and wait for the 'Q' key to be pressed
        cv2.imshow("Frame: "+str(self.frame_id), self.pixel_values_array)
        cv2.waitKey(0)
        cv2.destroyWindow("Frame: "+str(self.frame_id))


    def show_details(self):
        '''
        Function to display the details associated with the Frame
        '''

        # FrameID and Shape Attributes
        print("ImageID\t\t\t\t:", self.frame_id)
        print("Frame Height\t\t\t:", self.frame_height)
        print("Frame Width\t\t\t:", self.frame_width)
        print("Number of Channels\t\t:", self.number_of_channels)
        
        # Check if Class has been pridicted already;
        # if no, print No
        if self.class_predicted == -1: 
            print("Class Predicted\t\t\t: No")
        # if yes, print predicted class and confidence score
        else:                          
            print("Class Predicted\t\t\t:", self.class_predicted)
            print("Confidence Score\t\t:", self.confidence_score)
        
        # Check if Dense Optical Flow has been generated already;
        # if no, print No
        if self.dense_optical_flow_vector == False:
            print("Dense Optical Flow Generated\t: No")
        # if yes, print Yes
        else:
            print("Dense Optical Flow Generated\t: Yes")
        

    def predict_frame(self):
        '''
        Function to actually predict the frame
        '''

        # Convert the image to grayscale
        self.convert_to_grayscale()
        spatial_image_after_reshape = np.reshape(self.placeholder_pixel_values_array, (1, image_height,image_width,1))

        with graph.as_default():
            current_prediction = model.predict([np.array(spatial_image_after_reshape), np.array(self.dense_optical_flow_vector)])

        # Get the class with maximum confidence
        class_prediction = np.argmax(current_prediction)

        # Rounding the class probability
        class_probability = round(current_prediction[0][class_prediction], 4)

        predicted_class = class_names[class_prediction]

        # print("FrameID:",self.frame_id,"predicted as:", predicted_class)

        self.placeholder_class_predicted = predicted_class
        self.placeholder_confidence_score = class_probability



    def get_frame_id(self):
        '''
        Function to show the Current Frame's FrameID
        '''

        return (self.frame_id)


    def frame_predictions(self, class_predicted_input, confidence_score_input):
        '''
        Function to assign predictions to the Frame object
        Use this function as the interface to change predicted
        class and confidence score
        Parameters:
        @class_predicted_input  : The class that is predicted by the model
        @confidence_score_input : The confidence score that the predicted
                                  class is correct   
        '''
        self.placeholder_class_predicted = class_predicted_input
        self.placeholder_confidence_score = confidence_score_input


    # Destructor
    def __del__(self): 
        '''
        Destructor Function to destroy, and cleanup object properties
        '''

        # TODO: Perform Cleanup here
        
        # Uncomment to add logging:
        # print("Frame with FrameID: "+str(self.frame_id)+" was destroyed successfully.")
        pass