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
    print("We are going to generate a bunch of small bitmaps with colors")
    print("differing by small values.")
    print("\n\nThe name \"test_image_4_xxx.bmp\" indicates 4 different")
    print("shades of very dark black; once you histogram EQ it they should")
    print("fill all or almost all of the whole spectrum. For example,")
    print("a properly histogram EQ'd test_image_2_000.bmp should be pure")
    print("black and pure white.")

    for x in range(2, 5):
        for i in range(0, 5):
            new_bmp = np.random.choice(a=list(range(0,x)), size=image_dimensions).astype(np.uint8)
            print(new_bmp)
            im = Image.fromarray(new_bmp, 'L')

            file_index = str(i).zfill(3)
            im.save(os.path.join(test_images_dir, "test_image_{}_{}.bmp".format(x,file_index)))
