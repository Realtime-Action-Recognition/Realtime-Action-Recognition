# ==============================================================================================
# Title:   realtime_action_recognition.py
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

# Warnings import to warn user of certain pitfalls of the system
import warnings

# Import threading to multi-thread the application
import threading

# Numpy import for array manipulation
import numpy as np

# Randint import to randomly sample frames at the specified
# sampling rate
from random import randint

# Matplotlib import to plot the Model Training Accuracy
# and Training Loss
import matplotlib.pyplot as plt

# Import the Frame Module
import frame as Frame

# Import the FrameBuffer Module
import frame_buffer as FrameBuffer

# Import the ActionSteps Module
import action_steps as ActionSteps

#Load Model
from keras.models import load_model

# Import argument parser
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description='Command-line parameters to train the Three-Stream Algorithm on UCF-101')
parser.add_argument('-bs','--batch_size', default=32, help='The batch size to use while training. (default=32)')
parser.add_argument('-e','--epochs', default=100, help='Number of epochs to train. (default=100)')
parser.add_argument('-oc','--output_classes', default=101, help='Number of output classes. (default=101)')
parser.add_argument('-lr','--learning_rate', default=0.000001, help='Learning rate of the algorithm. (default=0.000001)')


# Initialize a previous temporal image
previous_frame = np.array([0])

step_images = [
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done"]

class ActionSystem:

    # Constructor
    def __init__(self):
        '''
        Constructor to initialize the Action system
        '''

        # Initialize the Frame Buffer of size 15
        self.frame_buffer = FrameBuffer.FrameBuffer(15)

        # ActionSteps
        # self.action_steps = ActionSteps.ActionSteps()

        # Initialize the Video Stream
        self.live_stream = cv2.VideoCapture(0)
        print("Video Stream started successfully.")
        print("Changing video stream resolution..")
        self.live_stream.set(3, 1080)
        self.live_stream.set(4, 1080)
        time.sleep(1)

        print("Video Stream resolution changed to: 1080x1080")

        # Previous frame stored to generate optical flow
        self.previous_frame = None

        # Circular system counter to keep count of frames
        self.frame_count = 1

        # Sampling rate of frames for prediction
        # In a 30 FPS Video, a sampling rate of 5
        # would mean: 1 in 5 frames are sampled for prediction
        # Therefore, every second, 6 frames are sampled
        self.sampling_rate = 5

        # Property to set default FPS of the video stream
        self.video_fps = 30

        # Interval to check buffer in (milliseconds)
        self.check_buffer_interval = 1000

        self.step_name =   {"step_1"       : "STEP 1",
                            "step_2_left"  : "STEP 2 LEFT",
                            "step_2_right" : "STEP 2 RIGHT",
                            "step_3"       : "STEP 3",
                            "step_4_left"  : "STEP 4 LEFT",
                            "step_4_right" : "STEP 4 RIGHT",
                            "step_5_left"  : "STEP 5 LEFT",
                            "step_5_right" : "STEP 5 RIGHT",
                            "step_6_left"  : "STEP 6 LEFT",
                            "step_6_right" : "STEP 6 RIGHT",
                            "step_7_left"  : "STEP 7 LEFT",
                            "step_7_right" : "STEP 7 RIGHT"}



    # Function to return frames to display on the Flask Server
    # DO NOT CHANGE
    def get_frame(self):
        '''
        Returns frames from the live video stream, encoded as jpeg bytes
        jpeg bytes is the format used by the Flask server to display the 
        live stream
        '''

        global previous_frame
        global step_images
        
        # Read the live feed frame
        success, image = self.live_stream.read()
        
        # Sample frames according to the sampling rate
        if self.frame_count % (int)(self.video_fps / self.sampling_rate) == 0:
            # Create a Frame Object

            # def manipulate_frame():
            frame_object = Frame.Frame(image)
            # Preprocess the frame for prediction
            frame_object.preprocess(112)

            # Generate the optical flow for the image
            frame_object.generate_optical_flow(previous_frame)

            # Predict the frame object's class here
            frame_object.predict_frame()

            self.frame_buffer.add_to_buffer(frame_object)

            # frame_thread = threading.Thread(target=manipulate_frame)
            # # t.daemon = True
            # frame_thread.start()

        ret, jpeg = cv2.imencode('.jpg', image)

        # Store the current Image as the previous image
        previous_frame = image

        # Increment frame_count
        self.frame_count += 1

        if( self.frame_count % 60 == 0 and self.frame_count > 90 ):
            step_completed = self.frame_buffer.get_step_predicted()
            return [jpeg.tobytes(), step_completed]
            # self.action_steps.add_step(step_completed)
        
        return [jpeg.tobytes()]

    def get_frame_buffer_instance(self):
        '''
        Function to pass the FrameBuffer Instance
        '''
        return self.frame_buffer


    def check_buffer(self):
        '''
        Function to check FrameBuffer, and interface it with the
        ActionSteps module.
        This fuction is called repeatedly, with an interval of: check_buffer_interval
        '''

        step_completed = self.frame_buffer.get_step_predicted()





    # Destructor
    def __del__(self):
        '''
        Destructor to delete the ActionSystem
        '''

        # Release the video stream
        self.live_stream.release()
        print("Live Stream successfully released.")

        # Delete the FrameBuffer
        del(self.frame_buffer)

# Tomorrow:
# - Only 5 of them, nowhere right now.
# - Varshini was asked to find people.

# I am in Cisco now, offer from GS. I need some time to get settled. 
# 1st person to be contacted.
# Only caveat is that I cannot join immediately.
# Talk about research, AI and ML. Caveat. I sprung on the opportunity.
# Definitely interested in coming on-board.


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vZm3LDdJLwBP-1LNVCV9oP8a77lRUjvp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vZm3LDdJLwBP-1LNVCV9oP8a77lRUjvp" -O handwash_step_model.h5 && rm -rf /tmp/cookies.txt