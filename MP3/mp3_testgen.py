#!/usr/bin/python3

 # Anything not directly related to processing here
from mp3_helper import *

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from math import floor


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


image_dimensions=(5,5)


if __name__ == "__main__":
    hello()
    print("\n\nThis is the test image generator for MP #3.")
    print("We are going to generate a bunch of small bitmaps with only")
    print("very light greys, for you to practice histogram equalization")
    print("on.\n\n")

    for i in range(0, 10):
        new_bmp = np.random.choice(a=[0, 1], size=image_dimensions).astype(np.uint8)
        print(new_bmp)
        im = Image.fromarray(new_bmp, 'L')

        file_index = str(i).zfill(3)
        im.save(os.path.join(test_images_dir, "test_image_{}.bmp".format(file_index)))
