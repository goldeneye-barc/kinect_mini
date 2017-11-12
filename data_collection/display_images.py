import cv2
import numpy as np
import frame_convert2
import sys
import time

ROOT_PATH = sys.argv[1] if len(sys.argv) > 1 else '../data/calibration'
DELAY = int(sys.argv[2]) if len(sys.argv) > 2 else 10
RGB_PATH = 'rgb'
DEPTH_PATH = 'depth'
DEBUG = len(sys.argv) > 3 and sys.argv[3] == "debug"


def display_depth(delay):
    i = 0
    while 1:
        try:
            data = np.load('{0}/{1}/{2}.npy'.format(ROOT_PATH, DEPTH_PATH, i))
            cv2.imshow('Depth', frame_convert2.pretty_depth_cv(data))
            if cv2.waitKey(delay) == 27:
                break
        except Exception:
            break
        if DEBUG:
            print i
        i += 1


def display_rgb(delay):
    i = 0
    while 1:
        try:
            data = np.load('{0}/{1}/{2}.npy'.format(ROOT_PATH, RGB_PATH, i))
            cv2.imshow('RGB', frame_convert2.video_cv(data))
            if cv2.waitKey(delay) == 27:
                break
        except Exception:
            break
        if DEBUG:
            print i
        i += 1


if __name__ == '__main__':
    display_depth(DELAY)
    display_rgb(DELAY)


