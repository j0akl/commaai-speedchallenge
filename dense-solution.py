import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("data/train.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

total_flow = []

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 20, 3, 5, 1.2, 0)
    # add all flow values to array to be saved for training
    total_flow.append(flow)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()

labels = open('data/train.txt', "r").readlines()
label_array = []
for i in labels:
    label_array.append(i[:-2])
print(label_array[0])
training_dataset = []
for i in range(100):
    training_dataset.append([total_flow[i], label_array[i]])
print(training_dataset[0])
val_ds_length = 4000
train_ds_length = len(training_dataset) - val_ds_length

train_ds, val_ds = random_split(training_dataset, [train_ds_length,
                                                   val_ds_length])
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)

input_size = 480 * 640

img, label = training_dataset[1]
# get range
max_value = torch.max(img)
min_value = torch.min(img)
print(max_value, min_value)
