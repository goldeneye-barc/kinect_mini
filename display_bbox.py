import cv2
import numpy as np
from bboxstep import BoundingBoxSim
images_path, cfg_path, weights_path, data_path, num_images ="./data/straight_line_forward/rgb/", "./neural_net/yolo-obj.cfg", "./neural_net/rover.weights", "./neural_net/obj.data", 200
image_sim = BoundingBoxSim(images_path=images_path, weights_path=weights_path, data_path=data_path, num_images=num_images, cfg_path=cfg_path)

image_sim.simulate(True)

while image_sim.step():
    curr_state = image_sim.curr_state
    print "Current state is", curr_state
    img = cv2.imread(curr_state['image'])
    cv2.rectangle(img, curr_state['top_left'], curr_state['bottom_right'], (0,255,0), 3)
    cv2.imshow('image', img)
    cv2.waitKey(200)
cv2.destroyAllWindows()



