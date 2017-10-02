#!/usr/bin/env python
import freenect
import cv2
import numpy as np
import frame_convert2
import os

# Calibration measurement is approximately 79cm away for the rover
cv2.namedWindow('Depth')
cv2.namedWindow('RGB')
keep_running = True 
ROOT_PATH = './data/lane_change'
RGB_PATH = 'rgb'
DEPTH_PATH = 'depth'

depth_index = 0
rgb_index = 0

def collect_depth(dev, data, timestamp):
    global keep_running
    global depth_index
    np.save('{0}/{1}/{2}'.format(ROOT_PATH, DEPTH_PATH, depth_index), data)
    cv2.imshow('Depth', frame_convert2.pretty_depth_cv(data))
    if cv2.waitKey(10) == 27:
        keep_running = False
    depth_index += 1


def collect_rgb(dev, data, timestamp):
    global keep_running
    global rgb_index
    np.save('{0}/{1}/{2}'.format(ROOT_PATH, RGB_PATH, rgb_index), data)
    rgb_index += 1
    cv2.imshow('RGB', frame_convert2.video_cv(data))
    if cv2.waitKey(10) == 27:
        keep_running = False


def body(*args):
    if not keep_running:
        raise freenect.Kill


if not os.path.exists(ROOT_PATH):
    os.makedirs(ROOT_PATH)
    os.makedirs('{0}/{1}'.format(ROOT_PATH, DEPTH_PATH))
    os.makedirs('{0}/{1}'.format(ROOT_PATH, RGB_PATH))
print('Press ESC in window to stop')
freenect.runloop(depth=collect_depth,
                 video=collect_rgb,
                 body=body)
