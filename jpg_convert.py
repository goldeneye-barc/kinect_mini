from PIL import Image
import numpy as np
import sys
ROOT_PATH = sys.argv[1] if len(sys.argv) > 1 else './data/calibration/rgb'

i = 0
try:
    while True:
        data = np.load('{}/{}.npy'.format(ROOT_PATH, i))
        img = Image.fromarray(data, 'RGB')
        img.save('{}/{}.jpg'.format(ROOT_PATH, i))
        i += 1
except Exception as E:
    print('done')

