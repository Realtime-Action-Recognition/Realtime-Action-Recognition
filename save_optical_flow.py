# ==============================================================================================
# Title:   save_optical_flow.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

import os
import cv2
import time
import pyflow
import _pickle as pickle
import marshal
import numpy as np

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

image_height = 224
image_width = 224

data_folder = "TestingData/"
save_folder = "ColorTrainTemporal/"

spatial_train_images = []
temporal_train_images = []
hog_train_images = []
train_labels = []
class_dict = {}
count = 0

already_done_classes = ["ApplyEyeMakeup",
"ApplyLipstick",
"Archery",
"BabyCrawling",
"BalanceBeam",
"BandMarching",
"BaseballPitch",
"Basketball"]
# "BasketballDunk",
# "BenchPress",
# "Biking",
# "Billiards",
# "BlowDryHair",
# "BlowingCandles",
# "BodyWeightSquats",
# "Bowling",
# "BoxingPunchingBag",
# "BoxingSpeedBag",
# "BreastStroke",
# "BrushingTeeth",
# "CleanAndJerk",
# "CliffDiving",
# "CricketBowling",
# "CricketShot",
# "CuttingInKitchen",
# "Diving",
# "Drumming",
# "Fencing",
# "FieldHockeyPenalty",
# "FloorGymnastics",
# "FrisbeeCatch",
# "FrontCrawl",
# "GolfSwing",
# "Haircut",
# "Hammering",
# "HammerThrow",
# "HandstandPushups",
# "HandstandWalking",
# "HeadMassage",
# "HighJump",
# "HorseRace",
# "HorseRiding",
# "HulaHoop",
# "IceDancing",
# "JavelinThrow",
# "JugglingBalls",
# "JumpingJack",
# "JumpRope",
# "Kayaking",
# "Knitting",
# "LongJump",
# "Lunges",
# "MilitaryParade",
# "Mixing",
# "MoppingFloor",
# "Nunchucks",
# "ParallelBars",
# "PizzaTossing",
# "PlayingCello",
# "PlayingDaf",
# "PlayingDhol",
# "PlayingFlute",
# "PlayingGuitar",
# "PlayingPiano",
# "PlayingSitar",
# "PlayingTabla",
# "PlayingViolin",
# "PoleVault",
# "PommelHorse",
# "PullUps",
# "Punch",
# "PushUps",
# "Rafting",
# "RockClimbingIndoor",
# "RopeClimbing",
# "Rowing",
# "SalsaSpin",
# "ShavingBeard",
# "Shotput",
# "SkateBoarding",
# "Skiing",
# "Skijet",
# "SkyDiving",
# "SoccerJuggling",
# "SoccerPenalty",
# "StillRings",
# "SumoWrestling",
# "Surfing",
# "Swing",
# "TableTennisShot"]

import matplotlib.pyplot as plt

from skimage.feature import *
from skimage import data, exposure


image = data.astronaut()


def get_hog(image):
    H = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

    (H, hogImage) = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")

    return hogImage
    
    # cv2.imshow("HOG Image", hogImage)
    # cv2.waitKey()

# im = cv2.imread('test1.jpg', 1)
# im = cv2.resize(im, (224, 224))
# print("L:", len(im))

# hog_start = time.time()
# get_hog(im)
# hog_end = time.time()

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

# t1 = cv2.imread('test1.jpg', 1)
# t2 = cv2.imread('test2.jpg', 1)
# previous_temporal_image = cv2.resize(t1, (224, 224))
# next_temporal_image = cv2.resize(t2, (224, 224))

# previous_temporal_image = previous_temporal_image.astype(float) / 255.
# next_temporal_image = next_temporal_image.astype(float) / 255.

# temp_start = time.time()

# u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)

# temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)

# temporal_image = process_flow(next_temporal_image, temporal_image)

# temp_end = time.time()

# print("HOG Time =", hog_end - hog_start)
# print("Temporal Time =", temp_end - temp_start)

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

video_classes = os.listdir(data_folder)

print("UCF-101 Classes:",video_classes)

classes_done = 0

for current_class in video_classes:
    
    if(current_class not in already_done_classes):
        continue

    classes_done += 1

    if current_class not in class_dict.keys():
        class_dict[current_class] = count
        count += 1

    class_image_counter = 0

    current_class_images = os.listdir(data_folder+str(current_class))
    min_frame_numbers = {}

    c = 0
    for image_name in current_class_images:
        group = image_name.split('_')[2]
        clip = image_name.split('_')[3].split('frame')[0]

        if group+":"+clip not in min_frame_numbers.keys():
            min_frame_numbers[group+":"+clip] = -1
        
        frame = int(image_name.split('frame')[1].split('.')[0])

        if frame < min_frame_numbers[group+":"+clip]:
            continue

        image_name_without_frame = image_name.split('frame')[0]
        # print(group, clip, frame, image_name)

        # Get the 5th next frame
        next_image_frame = str(int(frame)+10)

        next_image_path = image_name_without_frame + 'frame'
        if(len(next_image_frame)==1):
            next_image_path += '00000'
        elif(len(next_image_frame)==2):
            next_image_path += '0000'
        elif(len(next_image_frame)==3):
            next_image_path += '000'

        next_image_path += next_image_frame + '.jpg'

        min_frame_numbers[group+":"+clip] = int(next_image_frame)

        print(image_name, next_image_path)
        previous_temporal_image = cv2.imread(data_folder+current_class+'/'+image_name, 1)
        next_temporal_image = cv2.imread(data_folder+current_class+'/'+next_image_path, 1)


        if next_temporal_image is None:
            continue

        previous_temporal_image = cv2.resize(previous_temporal_image, (image_width, image_height))
        next_temporal_image = cv2.resize(next_temporal_image, (image_width, image_height))

        spatial_image_grayscale = cv2.cvtColor(previous_temporal_image, cv2.COLOR_BGR2GRAY)

        spatial_image_after_reshape = np.reshape(spatial_image_grayscale, (image_width, image_height, 1))

        hog_image = get_hog(previous_temporal_image)
        hog_image_after_reshape = np.reshape(hog_image, (image_width, image_height, 1))

        previous_temporal_image = cv2.resize(previous_temporal_image, (image_width, image_height))
        next_temporal_image = cv2.resize(next_temporal_image, (image_width, image_height))

        previous_temporal_image = previous_temporal_image.astype(float) / 255.
        next_temporal_image = next_temporal_image.astype(float) / 255.

        u, v, im2W = pyflow.coarse2fine_flow(previous_temporal_image, next_temporal_image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)

        temporal_image = np.concatenate((u[..., None], v[..., None]), axis=2)

        temporal_image = process_flow(next_temporal_image, temporal_image)

        # temporal_image_grayscale = cv2.cvtColor(temporal_image, cv2.COLOR_BGR2GRAY)

        temporal_image_after_reshape = np.reshape(temporal_image, (image_width, image_height, 1))

        spatial_train_images.append(np.array(spatial_image_after_reshape))
        temporal_train_images.append(np.array(temporal_image_after_reshape))

        hog_train_images.append(np.array(hog_image_after_reshape))
        train_labels.append(float(class_dict[current_class]))

        # # Visual Inspection
        # cv2.imshow('First Frame', previous_temporal_image)
        # cv2.imshow('Second Frame', next_temporal_image)
        # cv2.imshow('Temporal Image', temporal_image_after_reshape)
        # cv2.waitKey()

        # cv2.imwrite(save_folder+current_class+'/'+image_name_without_frame+str(frame)+"TO"+str(next_image_frame)+".jpg", temporal_image_after_reshape)

        # print("Shape Inside: ", len(temporal_train_images),"x",len(temporal_train_images[0]),"x",len(temporal_train_images[0][0]))

    # TODO: Clean 2 for longJump


    # pickle_start = time.time()
    # with open('temporal_pickle_test.pickle', 'wb') as handle:
    #     pickle.dump(temporal_train_images, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle1 = time.time()

    # with open('spatial_pickle_test.pickle', 'wb') as handle:
    #     pickle.dump(spatial_train_images, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle2 = time.time()

    # with open('pickle_test.pickle', 'wb') as handle:
    #     pickle.dump(train_labels, handle)#, protocol=pickle.HIGHEST_PROTOCOL)
    # pickle3 = time.time()

    

    # print("Pickle 1 :", pickle1 - pickle_start)
    # print("Pickle 2 :", pickle2 - pickle1)
    # print("Pickle 3 :", pickle3 - pickle2)

    # print("Marshal 1 :", marshal1 - marshal_start)
    # print("Marshal 2 :", marshal2 - marshal1)
    # print("Marshal 3 :", marshal3 - marshal2)


pickle_start = time.time()
f = open("temporal_test_pickle_8_classes.pyc", "wb")
pickle.dump(temporal_train_images, f)
f.close()
pickle1 = time.time()

f = open("spatial_test_pickle_8_classes.pyc", "wb")
pickle.dump(spatial_train_images, f)
f.close()
pickle2 = time.time()

f = open("pickle_test_label_8_classes.pyc", "wb")
pickle.dump(train_labels, f)
f.close()
pickle3 = time.time()

f = open("pickle_test_hog_8_classes.pyc", "wb")
pickle.dump(hog_train_images, f)
f.close()
pickle4 = time.time()

f = open("pickle_class_test_mapping_8_classes.pyc", "wb")
pickle.dump(class_dict, f)
f.close()
pickle5 = time.time()

print("pickle 1 :", pickle1 - pickle_start)
print("pickle 2 :", pickle2 - pickle1)
print("pickle 3 :", pickle3 - pickle2)
print("pickle 5 :", pickle4 - pickle3)
print("pickle 4 :", pickle5 - pickle4)

print("Shape Outside: ", len(temporal_train_images),"x",len(temporal_train_images[0]),"x",len(temporal_train_images[0][0]))

        # print(image_name, next_image_path)
    # previous_temporal_image = np.array([0])

    # for current_image in current_class_images:

    #     # if(class_image_counter < max_images_from_second_class):
    #     image_number = str(int(str(current_image).split('e')[1].split('.')[0]))
    #     image_path = data_folder + str(current_class) + '/' + str(current_image)

    #     read_image = cv2.imread(image_path, 1)
    #     read_image = cv2.resize(read_image, (image_width, image_height))
    #     # spatial_image_before_reshape = cv2.resize(read_image, (image_width, image_height))
    #     spatial_image_grayscale = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
    #     spatial_image_after_reshape = np.reshape(spatial_image_grayscale, (image_width, image_height, 1))

    #     print(image_number, ":", image_path)

    # if(classes_done == 1):
    #     break
