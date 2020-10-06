#!/usr/bin/python3


import os
import sys
import itertools
from PIL import Image
import numpy as np

from mp2 import *


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
images_dir = os.path.join(script_dir, "test_images")
images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])


se_identity_1 = np.array([[True]])
se_identity_3 = np.array([[False, False, False],
                          [False, True,  False],
                          [False, False, False]])
se_identity_5 = np.array([[False, False, False, False, False],
                          [False, False, False, False, False],
                          [False, False,  True, False, False],
                          [False, False, False, False, False],
                          [False, False, False, False, False]])
se_cross_3    = np.array([[False, True,  False],
                          [True,  True,   True],
                          [False, True,  False]])

if __name__ == "__main__":
    print("Andrew Quinn - EECS 332 - MP#2, Testing Framework")
    print("=" * 80)

    for image in images:
        print(image)
        img_in = np.array(Image.open(image).convert('1'))
        print(img_in)
        print("")


        print("\n\n\n============ EROSION TEST ==============\n\n\n")
        print("SE 1x1 identity (should all be True):")
        print(img_in == erode(img_in, se_identity_1))
        print("SE 3x3 identity (should all be True):")
        print(img_in == erode(img_in, se_identity_3))
        print("SE 5x5 identity (should all be True):")
        print(img_in == erode(img_in, se_identity_5))

        print("SE 3x3 cross (only Trues which have True on all sides):")
        print(img_in)
        print(erode(img_in, se_cross_3))

        print("\n\n\n============ DILATION TEST ==============\n\n\n")
        print("SE 1x1 identity (should all be True):")
        print(img_in == dilate(img_in, se_identity_1))
        print("SE 3x3 identity (should all be True):")
        print(img_in == dilate(img_in, se_identity_3))
        print("SE 5x5 identity (should all be True):")
        print(img_in == dilate(img_in, se_identity_5))

        print("SE 3x3 cross (only Trues which have True on any sides):")
        print(img_in)
        print(dilate(img_in, se_cross_3))
