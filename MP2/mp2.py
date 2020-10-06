#!/usr/bin/python3


import os
import sys
import itertools
from PIL import Image
import numpy as np


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
images_dir = os.path.join(script_dir, "images")
images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]


def neighborhood_coordinates(img, x, y, dx=1, dy=1):
    """Given a distance out, returns the checkable neighborhood around a
    pixel for an image."""
    x_nbhd = list(range( x - dx, x + dx + 1))
    y_nbhd = list(range( y - dy, y + dy + 1))

    x_nbhd = filter(lambda x: x >= 0 and x < img.shape[0], x_nbhd)
    y_nbhd = filter(lambda x: x >= 0 and x < img.shape[1], y_nbhd)

    return list(itertools.product(list(x_nbhd), list(y_nbhd)))


def check_if_se_coordinate_valid(img, se, x, y, i, j):
    """Given a structural element, an image, an x-y coordinate for the image,
    and an i-j coordinate for the structural element, checks to see whether
    the element needs to be operated on."""
    x_adjusted = x + i - se.shape[0] // 2
    y_adjusted = y + j - se.shape[1] // 2

    x_yes = (x_adjusted >= 0) and (x_adjusted < img.shape[0])
    y_yes = (y_adjusted >= 0) and (y_adjusted < img.shape[1])

    return x_yes and y_yes

def eroded_pixel(img, se, x, y, verbose=False):
    """Returns the post-erosion value of the pixel.

    Note that the structuring element *must* be an odd number-by-odd number
    2D numpy matrix of booleans."""
    assert se.shape[0] % 2 == 1
    assert se.shape[1] % 2 == 1

    return_bool = img[x, y]

    # where_to_check = neighborhood_coordinates(img, x, y, dx=se.shape[0]//2,
    #                                                      dy=se.shape[1]//2)
    # for (x, y) in where_to_check:

    if verbose:
        print("Determining whether {} is active...".format((x,y)))

    for i in range(0, se.shape[0]):
        if not return_bool:
            break
        for j in range(0, se.shape[1]):
            if not return_bool:
                break
            if check_if_se_coordinate_valid(img, se, x, y, i, j):
                comp = (x + i - se.shape[0]//2,
                        y + j - se.shape[1]//2)
                if verbose:
                    print("  Cross-checking absolute image pixel ({}, {}) against SE pixel ({}, {}).".format(comp[0],comp[1],i,j), end="    ")

                if se[i, j]:
                    if verbose:
                        print("Pixel active, assigning... {} && {} == {}.".format(img[comp[0], comp[1]], se[i, j], (img[comp[0], comp[1]] and se[i, j])))
                    return_bool = return_bool and (img[comp[0], comp[1]] and se[i, j])
                else:
                    if verbose:
                        print("SE pixel not active. Moving on.")

    if verbose:
        print("Return value: {} is {}.".format((x, y), return_bool))

    return return_bool


def erode(img, se, verbose=False):
    out = np.zeros((img.shape[0], img.shape[1])).astype(type(True))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            out[i, j] = eroded_pixel(img, se, i, j, verbose=verbose)

    return out



def dilated_pixel(img, se, x, y, verbose=False):
    """Returns the post-dilation value of the pixel.

    Note that the structuring element *must* be an odd number-by-odd number
    2D numpy matrix of booleans."""
    assert se.shape[0] % 2 == 1
    assert se.shape[1] % 2 == 1

    return_bool = img[x, y]

    # where_to_check = neighborhood_coordinates(img, x, y, dx=se.shape[0]//2,
    #                                                      dy=se.shape[1]//2)
    # for (x, y) in where_to_check:

    if verbose:
        print("Determining whether {} is active...".format((x,y)))

    for i in range(0, se.shape[0]):
        if return_bool:
            break
        for j in range(0, se.shape[1]):
            if return_bool:
                break
            if check_if_se_coordinate_valid(img, se, x, y, i, j):
                comp = (x + i - se.shape[0]//2,
                        y + j - se.shape[1]//2)
                if verbose:
                    print("  Cross-checking absolute image pixel ({}, {}) against SE pixel ({}, {}).".format(comp[0],comp[1],i,j), end="    ")

                if se[i, j]:
                    if verbose:
                        print("Pixel active, assigning... {} && {} == {}.".format(img[comp[0], comp[1]], se[i, j], (img[comp[0], comp[1]] and se[i, j])))
                    return_bool = return_bool or (img[comp[0], comp[1]] and se[i, j])
                else:
                    if verbose:
                        print("SE pixel not active. Moving on.")

    if verbose:
        print("Return value: {} is {}.".format((x, y), return_bool))

    return return_bool


def dilate(img, se, verbose=False):
    out = np.zeros((img.shape[0], img.shape[1])).astype(type(True))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            out[i, j] = dilated_pixel(img, se, i, j, verbose=verbose)

    return out


def opening(img, se, verbose=False):
    return dilate(erode(img, se, verbose=verbose), se, verbose=verbose)


def closing(img, se, verbose=False):
    return erode(dilate(img, se, verbose=verbose), se, verbose=verbose)


if __name__ == "__main__":
    print("Andrew Quinn - EECS 332 - MP#2\n" + ("-" * 80))

    if not os.path.exists(os.path.join(script_dir, 'results')):
        os.makedirs(os.path.join(script_dir, 'results'))

    test_img_array_1 = np.array([[True, True, True],
                                 [True, True, True],
                                 [True, True, True]])

    test_se_1 = np.array([[False, True, True],
                          [True, True, True],
                          [True, True, True]])

    print("Testing")
    print(erode(test_img_array_1, test_se_1, verbose=True))

    # for image in images:
    #     img_in = np.array(Image.open(image).convert('1'))
    #     print(img_in)
    #
    #     se = np.array([[True, True, True],
    #                    [True, True, True],
    #                    [True, True, True]])
    #
    #     eroded_pixel(img_in, se, 0, 0)
