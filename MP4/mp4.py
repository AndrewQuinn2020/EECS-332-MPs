#!/usr/bin/python3

import logging
import os
import sys
from colorsys import hsv_to_rgb

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

from PIL import Image
import numpy as np
import colorlog

logger = logging.getLogger()
logger.setLevel(colorlog.colorlog.logging.DEBUG)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=10000)

script_dir = os.path.dirname(__file__)
hist_output_dir = os.path.join(script_dir,
                               "histogram_data")
images_dir = os.path.join(script_dir, "images")
results_dir = os.path.join(script_dir, 'results')


def array2tuples(img_array):
    """Given a 2-dimensional array of vectors length 3, returns a 2d array
    of tuples length 3.

    I just prefer to work with tuples because they're hashable and play nice
    with dictionaries in Python. The up front cost isn't that great."""
    return_array = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=(type((1,2,3))))
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            tuple_in = (img_array[i, j, 0], img_array[i, j, 1], img_array[i, j, 2])
            return_array[i, j] = tuple_in

    return return_array


def create_hard_mask_3d(img, hist, threshold=0):
    """Given a dictionary of 3-tuples and a dictionary of 3-tuples which
    returns histogram values, return an array of 0s and 1s with the same
    dimensions as img based on whether the pixel at img[i, j] met or
    exceeded the threshold.

    If the pixel value isn't found in the dictionary at all, it defaults
    to 0."""
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] in hist:
                mask[i, j] = 1
    return mask


def apply_hard_mask(img_array, mask):
    """Given img_array and a mask value, returns a new array like image_array
    but all pixels with 0s in the mask have been turned off."""
    img_out_array = np.zeros_like(img_array)
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            img_out_array[i, j] = img_array[i, j] * mask[i, j]
    return img_out_array


def array_hsv2rgb(img_array):
    """Given an array of vectors of length 3 of HSV values, rewrites it in
    place to be RGB values."""
    for i in range(0, img_array.shape[0]):
        for j in range(0, img_array.shape[1]):
            rgb = hsv_to_rgb(img_array[i,j,0]/255, img_array[i,j,1]/255, img_array[i,j,2]/255)
            img_array[i,j,0] = rgb[0] * 255
            img_array[i,j,1] = rgb[1] * 255
            img_array[i,j,2] = rgb[2] * 255
    return img_array

if __name__ == "__main__":
    logger.info("Andrew Quinn - EECS 334 - MP 4")
    logger.info("-" * 80)
    logger.info("This file should be run *after* `mp4_histogram_training.py`")
    logger.info("has generated histogram training data in `histogram_data/`.")

    # Load in the histogram files.
    hist_rgb = pickle.load( open( os.path.join(hist_output_dir, "hist_rgb.pickle"), "rb"))
    hist_hsv = pickle.load( open( os.path.join(hist_output_dir, "hist_hsv.pickle"), "rb"))
    hist_rg = pickle.load( open( os.path.join(hist_output_dir, "hist_rg.pickle"), "rb"))
    hist_rb = pickle.load( open( os.path.join(hist_output_dir, "hist_rb.pickle"), "rb"))
    hist_gb = pickle.load( open( os.path.join(hist_output_dir, "hist_gb.pickle"), "rb"))
    hist_hs = pickle.load( open( os.path.join(hist_output_dir, "hist_hs.pickle"), "rb"))
    hist_hv = pickle.load( open( os.path.join(hist_output_dir, "hist_hv.pickle"), "rb"))
    hist_sv = pickle.load( open( os.path.join(hist_output_dir, "hist_sv.pickle"), "rb"))
    
    for path, subdirs, files in os.walk(images_dir):
        for name in files:
            img = os.path.join(path, name)
            logger.debug("Now operating on {}".format(img))

            image_array_rgb = np.array(Image.open(img).convert("RGB"))
            image_array_hsv = np.array(Image.open(img).convert("HSV"))
            image_array_rgb_tupled = array2tuples(image_array_rgb)
            image_array_hsv_tupled = array2tuples(image_array_hsv)

            mask_rgb = create_hard_mask_3d(image_array_rgb_tupled, hist_rgb)
            image_out_rgb = apply_hard_mask(image_array_rgb, mask_rgb)

            save_loc = os.path.join(results_dir, name[:-4] + "_rgb_mask.bmp")
            logger.debug("Saving RGB mask to {}".format(save_loc))
            im = Image.fromarray(image_out_rgb)
            im.save(save_loc)


            mask_hsv = create_hard_mask_3d(image_array_hsv_tupled, hist_hsv)
            image_out_hsv = apply_hard_mask(image_array_hsv, mask_hsv)

            save_loc = os.path.join(results_dir, name[:-4] + "_hsv_mask.bmp")
            logger.debug("Saving HSV mask to {}".format(save_loc))
            im = Image.fromarray(array_hsv2rgb(image_out_hsv))
            im.save(save_loc)
