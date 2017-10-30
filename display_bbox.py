import cv2
import numpy as np
import frame_convert2
from bboxstep import BoundingBoxSim

def argminmax_constrained(arr, x_s, y_s, x_e, y_e):
    curr_min, curr_max, max_ind, min_ind = float('infinity'), -float('infinity'), [0,0], [0,0]
    #x_s, y_s, x_e, y_e = max(x_s, 0), max(y_s, 0), min(x_e, len(arr[0]) - 1), min(y_e, len(arr) - 1)
    for i in range(y_s, y_e + 1):
        for j in range(x_s, x_e + 1):
            if arr[i][j] > curr_max:
                curr_max = arr[i][j]
                max_ind = [i, j]
            if arr[i][j] < curr_min:
                curr_min = arr[i][j]
                min_ind = [i, j]
    return max_ind, min_ind #now do arr[max_ind[0]][max_ind[1]] for max and similar for min

def main():
    global depth
    all_images_path= "./data/calibration"
    images_path, cfg_path, weights_path, data_path, num_images = all_images_path+"/rgb/", "./neural_net/yolo-obj.cfg", "./neural_net/rover.weights", "./neural_net/obj.data", 50
    image_sim = BoundingBoxSim(images_path=images_path, weights_path=weights_path, data_path=data_path, num_images=num_images, cfg_path=cfg_path)

    image_sim.simulate()

    while image_sim.step():
        curr_state = image_sim.curr_state
        step_num = image_sim.curr_step
        print "Step number ", step_num
        print "Current state is ", curr_state

        img = cv2.imread(curr_state['image'])
        cv2.rectangle(img, curr_state['top_left'], curr_state['bottom_right'], (0,255,0), 3)
        #cv2.rectangle(img, (0,0), (640, 480), (0,255,0), 3)

        depth = np.load(all_images_path + '/depth/' + str(step_num)+ '.npy')
        x1, y1= curr_state['top_left']
        x2, y2= curr_state['bottom_right']
        #depth[y1:y2,x1:x2] = 255
        #print x1,y1,x2,y2
        truncated_depth = depth[y1:y2,x1:x2]
        global truncated_depth
        print truncated_depth
        depth_m = 0.1236 * np.tan(truncated_depth / 2842.5 + 1.1863)
        depth_m[depth_m < 0] = np.inf
        print depth_m.min()
        (i,j) = np.unravel_index(depth_m.argmin(), depth_m.shape) 
        z = depth_m[i][j]
        i, j = i + y1, j + x1

        cv2.rectangle(depth, (j-3, i-3), (j+3, i+3), (0,255,0), 2)
        assert z == 0.1236 * np.tan(depth[i][j] / 2842.5 + 1.1863)
        minDistance = -10
        scaleFactor = .0021
        w = 640
        h = 480
        bbox_j, bbox_i = (y1+y2)//2, (x1+x2)//2
        x = (bbox_i - w / 2) * (z + minDistance) * scaleFactor
        y = (bbox_j - h / 2) * (z + minDistance) * scaleFactor
        print "World Coordinates", x, y, z

        """
        #cv2.rectangle(depth, curr_state['top_left'], curr_state['bottom_right'], (-1,0,255), 3)

        y1, x1= curr_state['top_left']
        y2, x2= curr_state['bottom_right']
        print x1, y1, x2, y2
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, len(depth_m[0]) - 1), min(y2, len(depth_m) - 1)
        max_ind, min_ind = argminmax_constrained(depth_m, x1, y1, x2, y2)
        max_ind_g, min_ind_g = argminmax_constrained(depth_m, 0, 0, 639, 479)
        #print argminmax_constrained(depth_m, 0, 0, 639, 479)
        cv2.rectangle(depth_m, (min_ind[0] - 5, min_ind[1] - 5), (min_ind[0] + 5, min_ind[1] + 5), (0,255,0), 2)
        #cv2.rectangle(depth_m, (max_ind[0] - 5, max_ind[1] - 5), (max_ind[0] + 5, max_ind[1] + 5), (0,255,0), 2)
        cv2.rectangle(depth_m, (min_ind_g[0] - 3, min_ind_g[1] - 3), (min_ind_g[0] + 3, min_ind_g[1] + 3), (0,255,0), 1)
        #cv2.rectangle(depth_m, (max_ind_g[0] - 3, max_ind_g[1] - 3), (max_ind_g[0] + 3, max_ind_g[1] + 3), (0,255,0), 1)
        """

        cv2.imshow('image', img)
        cv2.imshow('Depth', frame_convert2.pretty_depth_cv(depth))
        cv2.waitKey(1000)
        
if __name__=="__main__":
    main()



