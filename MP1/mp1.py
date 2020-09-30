#!/usr/bin/python3

import os
import sys
from PIL import Image
import numpy as np


np.set_printoptions(threshold=sys.maxsize)

script_dir = os.path.dirname(__file__)

rel_image_paths = ['test.bmp',
                   'face.bmp',
                   'gun.bmp']

abs_image_paths = list(map(lambda s: os.path.join(script_dir, s),
                           rel_image_paths))


def binary_bmp_to_txt(bmp_array):
    """Prints x's where the color white is, and whitespace where black is."""
    for i in range(0, bmp_array.shape[0]):
        for j in range(0, bmp_array.shape[1]):
            if p[i, j]:
                print('x', end="")
            else:
                print(' ', end="")
        print("")




if __name__ == "__main__":
    print("EECS 332 - MP#1 - Andrew Quinn solution")
    print("---------------------------------------")

    p = np.array(Image.open(abs_image_paths[0]))

    binary_bmp_to_txt(p)

    #
    # p2 = np.zeros(p.shape)
    # print(p2)
    #
    # eq = initial_ccl_pass(p, p2)
    #
    # print(p2)
    # print(eq)
