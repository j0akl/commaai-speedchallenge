from __future__ import print_function
import cv2 as cv
import numpy as np

video_path = "data/train.mp4"

cam = cv.VideoCapture(video_path)
_ret, prev = cam.read()
height, width, layers = prev.shape
new_h = height // 4
new_w = width // 4
prev = cv.resize(prev, (new_w, new_h))
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

#file_counter = 0
#while file_counter < 10:
counter = 0
total_flow = np.zeros((10798, new_h, new_w, 2))
while counter < 10798:
    _ret, pre_resizing = cam.read()
    if _ret == True:
        resized = cv.resize(pre_resizing, (new_w, new_h))
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        total_flow[counter] = flow
        prevgray = gray
    else:
        pass

    print("frame: ", counter)
    counter += 1
data = np.asarray(total_flow)
# flow_path = "data/flows/test_flow.npz"
# flow_path = "data/flows/train_flow" + str(file_counter) + ".npz"
# f = open(flow_path, "a")
# f.close()
np.savez_compressed(flow_path, data)
# print("finished round ", str(file_counter))
# file_counter += 1
print('done!')
