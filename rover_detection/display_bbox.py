from __future__ import division
import cv2
import numpy as np
import frame_convert2
import matplotlib.pyplot as plt
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

def rgb_rect(state):
    img = cv2.imread(state['image'])
    cv2.rectangle(img, state['top_left'], state['bottom_right'], (0,255,0), 3)
    return img

def calc_coords(depth_path, state):
    depth = np.load(depth_path)
    x1, y1 = state['top_left']
    x2, y2 = state['bottom_right']
    truncated_depth = depth[y1:y2,x1:x2]
    depth_m = 0.1236 * np.tan(truncated_depth / 2842.5 + 1.1863)
    depth_m[depth_m < 0] = np.inf

    (i,j) = np.unravel_index(depth_m.argmin(), depth_m.shape) 
    z = depth_m[i][j]
    i, j = i + y1, j + x1
    assert z == 0.1236 * np.tan(depth[i][j] / 2842.5 + 1.1863)

    cv2.rectangle(depth, (j-3, i-3), (j+3, i+3), (0,255,0), 2)
    w, h = 640, 480
    bbox_j, bbox_i = (y1+y2)//2, (x1+x2)//2
    x = (bbox_i - w / 2)* z * 0.00173667 * 2
    y = (bbox_j - h / 2)* z * 0.00173667 * 2

    return (depth, (x,y,z))

def analyze_coords(coords):
    pass

def plot_states(**kwargs):
    for i, kw in enumerate(kwargs.keys()):
        arg = kwargs[kw]
        end_time = 1 / 30 * len(arg)
        t = np.linspace(0, end_time, len(arg))
        assert len(t) == len(arg)
        plt.figure(i)
        plt.plot(arg)
        plt.title(kw)
    plt.show()

def main():
    all_images_path= "../data/u_turn"
    images_path, cfg_path, weights_path, data_path, num_images = all_images_path+"/rgb/", "./neural_net/yolo-obj.cfg", "./neural_net/rover.weights", "./neural_net/obj.data", 10
    image_sim = BoundingBoxSim(images_path=images_path, weights_path=weights_path, data_path=data_path, num_images=num_images, cfg_path=cfg_path)

    image_sim.simulate()

    # Analyzing variables
    global world_coords
    world_coords = []
    xvel = []
    yvel = []
    zvel = []

    while image_sim.step():
        # Get current step and state
        curr_state = image_sim.curr_state
        step_num = image_sim.curr_step
        print "Step number ", step_num
        print "Current state is ", curr_state

        # Compute image rects and coords
        img = rgb_rect(curr_state)
        depth_path = all_images_path + '/depth/' + str(step_num)+ '.npy'
        depth, coords = calc_coords(depth_path, curr_state)

        # Analyze obtained coordinates
        analyze_coords(coords)
        print "World Coordinates (cm)", coords
        world_coords.append([100 * coord for coord in coords])
        if len(world_coords) > 1:
            T = 1 / 30
            xvel.append((world_coords[-1][0]-world_coords[-2][0])/ T )
            yvel.append((world_coords[-1][1]-world_coords[-2][1])/ T )
            zvel.append((world_coords[-1][2]-world_coords[-2][2]) /T )
            print "Velocities (cm / s)", xvel[-1], yvel[-1], zvel[-1]


        # Display images
        cv2.imshow('image', img)
        cv2.imshow('Depth', frame_convert2.pretty_depth_cv(depth))
        cv2.waitKey(10)

    cv2.destroyAllWindows()

    # Plotting functions
    world_coords = np.array(world_coords)
    plot_states(x=world_coords[:, 0], y=world_coords[:, 1], z=world_coords[:, 2], xvel=xvel, yvel=yvel, zvel=zvel)
        
if __name__=="__main__":
    main()



