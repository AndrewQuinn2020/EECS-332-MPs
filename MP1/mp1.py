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


def check_binary_pixel(x, y, old_array, new_array, next_region):
    """
    Does a single iteration of the CCL algorithm, on a single, binary pixel.

    On a black and white bitmap file, the command


        # from PIL import Image
        # import numpy as np

        np.array( Image.open( image_path ) )


    will return an array of booleans (True, False). False corresponds to the
    black value, and True the white, respectively. This makes running the
    CCL pass algorithm on a single pixel very straightforward.

    If the pixel is False (black), it simply gets a zero. We are only interested
    in contiguous white regions in this lab.

    Else, we look first at the above pixel (if it exists). If the above pixel
    has a nonzero value, we take that as our new value. Then we look at the
    pixel to the left of us; if that has a nonzero value, *and* we have a
    nonzero value (from the above pixel), we return a 2-tuple (to remark that)
    these two regions are contiguous). If the left pixel has a nonzero value but
    we are still 0, we take its value. Finally, if both the left and the above
    pixel are 0, we assume we've discovered a new potential region, and we
    return current_region + 1.
    """

    if not old_array[x, y]:
        new_array[x, y] = 0
        return next_region
    else:
        if y > 0:
            if new_array[x, y-1] == 0:
                pass
            else:
                new_array[x, y] = new_array[x, y-1]
                if x > 0 and new_array[x-1, y] > 0:
                    return (next_region, new_array[x, y-1], new_array[x-1, y])
                return next_region
        if x > 0:
            if new_array[x-1, y] == 0:
                new_array[x, y] = next_region + 1
                return next_region + 1
            else:
                new_array[x, y] = new_array[x-1, y]
                return next_region


if __name__ == "__main__":
    print("EECS 332 - MP#1 - Andrew Quinn solution")
    print("---------------------------------------")

    p = np.transpose(np.array(Image.open(abs_image_paths[0])))
    print(p)
    print(p.shape)

    p2 = np.zeros(p.shape)
    print(p2)

    equiv_table = []

    nr = 1
    for i in range(0, p.shape[0]):
        for j in range(0, p.shape[1]):
            nr = check_binary_pixel(i, j, p, p2, nr)
            if isinstance(nr, type((1,2,3))):
                print("TUPLE FOUND")
                equiv_table.append((nr[1], nr[2]))
                nr = nr[0]

    print(equiv_table)
