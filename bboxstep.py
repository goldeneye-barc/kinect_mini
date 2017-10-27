from __future__ import division
import pickle
import time
import darknet as dn

class BoundingBoxSim:
    def __init__(self, **kwargs):
        self.images_path = kwargs['images_path']
        self.cfg_path = kwargs['cfg_path']
        self.weights_path = kwargs['weights_path']
        self.data_path = kwargs['data_path']
        self.num_images = kwargs['num_images']
        try:
            self.bbox_data = pickle.load(open('{}/bboxdata.p'.format(self.images_path), 'rb'))
            print "Loaded existing file with " + str(len(self.bbox_data)) + " images precomputed"
        except IOError:
            self.bbox_data = []

    def simulate(self, from_scratch=False):
        """
        Precompute the simulation to save time while running it. 

        Args:
            from_scratch (bool): Recompute the entire simulation and don't use saved data

        Returns:
            None

        """
        if from_scratch:
            self.bbox_data = []
        net = dn.load_net(self.cfg_path, self.weights_path, 0)
        meta = dn.load_meta(self.data_path)
        max_val = self.num_images
	time_taken = []
        for i in range(self.num_images + 1):
	    start = time.clock()
	    try:
		    if i < len(self.bbox_data):
			continue
		    image = "{}{}.jpg".format(self.images_path, i)
		    result = dn.detect(net, meta, image)
		    time_taken.append(time.clock() - start)
		    coords = result[0][2]
		    x,y,w,h = [int(coord) for coord in coords]
		    self.bbox_data.append(
			    {
				'image': image,
				'center': (x, y),
				'width': w,
				'height': h,
				'top_left': (max(x-w//2, 0), max(y-h//2, 0)), 
				'bottom_right': (x+w//2, y+h//2)
			    })
		    pickle.dump(self.bbox_data, open('{}/bboxdata.p'.format(self.images_path), 'wb'))
		    print image
		    print self.bbox_data[i]
		    print "Saved array to image path"
		    print "-------------------------"
	    except Exception:
		    print "Error with image"
        print "Done simulating"
	print "Average time take was " + str(sum(time_taken)/len(time_taken))+ "seconds"

    def reset(self):
        """
        Resets the simulation to the initial state
        Args:
            None
        Returns:
            None
        """
        self.curr_step = 0

    def step(self):
        """
        Moves the simulation forward by one timestep and update curr_state
        Args:
            None
        Returns:
            curr_step (list)
        """
        if not hasattr(self, 'curr_step'):
            self.curr_step = 0
        if self.curr_step >= self.num_images:
            print 'Completed loop. Use the reset function to start the steps over'
            self.curr_state = None
            return 
        self.curr_state = self.bbox_data[self.curr_step]
        self.curr_step += 1
        return self.curr_state

