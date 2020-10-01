#!/usr/bin/python3

import os
import sys

from PIL import Image
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

script_dir = os.path.dirname(__file__)

image_dimensions = (10, 10)


if __name__ == "__main__":
    print("EECS 332 - MP#1 - Andrew Quinn random bmp file generator")

    print(os.path.join(script_dir, "mid_test", "mid_01.bmp"))

    for i in range(0, 10):
        new_bmp = np.random.choice(a=[False, True], size=image_dimensions)
        print(new_bmp)
        im = Image.fromarray(new_bmp)

        file_index = str(i).zfill(3)
        im.save(os.path.join(script_dir, "mid_tests", "mid_{}.bmp".format(file_index)))
