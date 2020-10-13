#!/usr/bin/python3


import os
import sys
import itertools
from PIL import Image
import numpy as np
from pathlib import Path


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
images_dir = os.path.join(script_dir, "images")
images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]

se_identity_1 = np.array([[True]])
se_cross_3    = np.array([[False, True,  False],
                          [True,  True,   True],
                          [False, True,  False]])
se_north_3    = np.array([[False, True,  False],
                          [False, True,  False],
                          [False, False, False]])
se_glider_3    = np.array([[False, True,  False],
                           [True, False,  False],
                           [True, True,  True]])
se_block_3    = np.array([[True,  True,   True],
                          [True,  True,   True],
                          [True,  True,   True]])

structural_elements = [se_identity_1, se_cross_3, se_north_3, se_glider_3]

def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]


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


def boundary(img, se, verbose=False):
    out = np.zeros((img.shape[0], img.shape[1])).astype(type(True))
    diff = erode(img, se, verbose=verbose)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if not img[i, j]:
                out[i, j] = False
            elif diff[i, j]:
                out[i, j] = False
            else:
                out[i, j] = img[i, j]

    return out





if __name__ == "__main__":
    print("Andrew Quinn - EECS 332 - MP#2\n" + ("-" * 80))
    results_dir = os.path.join(script_dir, 'results')
    se_dir = os.path.join(script_dir, 'structure_elems')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(se_dir):
        os.makedirs(se_dir)

    for se in structural_elements:
        se_name = namestr(se)[0]
        se_save_location = os.path.join(se_dir, se_name + ".bmp")
        im = Image.fromarray(se.astype(np.uint8) * 255, 'L')
        im.save(se_save_location)

    print("You can see the various SEs used by checking structure_elems/.")

    for image in images:
        print(image)
        img_in = np.array(Image.open(image))
        for se in structural_elements:
            for op in [erode, dilate, opening, closing, boundary]:
                rel_filename = Path(image).stem + "-" + namestr(se)[0] + "-" + namestr(op)[0] + ".bmp"
                abs_filename = os.path.join(results_dir, rel_filename)

                im = Image.fromarray(op(img_in, se))
                im.save(abs_filename)
