#!/usr/bin/python3

import logging
import os
import sys

try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

from PIL import Image
import numpy as np
from pathlib import Path

import colorlog


logger = logging.getLogger()
logger.setLevel(colorlog.colorlog.logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

# logger.debug("Debug message")
# logger.info("Information message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical message")


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=10000)


script_dir = os.path.dirname(__file__)
training_data_dir = os.path.join(script_dir,
                                 "histogram_training_images",
                                 "sfa",
                                 "SKIN",
                                 "5")
# training_data_dir = os.path.join(script_dir,
#                                  "histogram_training_images",
#                                  "sfa_small_test")
hist_output_dir = os.path.join(script_dir,
                               "histogram_data")


def img2hists(img_path, hist_rgb={}, hist_hsv={}, total_pixels=0):
    """Given a Pillow image, return the number of pixels in it, and two
    dictionaries, containing the histogram data of the image as an RGB
    file and an HSV file respectively.

    By default it gives dictionaries for just the current image, but if
    you want to collect information for a sequence of images you can pass
    it non-empty dictionaries for hist_rgb and hist_hsv."""

    image_array_rgb = np.array(Image.open(img).convert("RGB"))
    image_array_hsv = np.array(Image.open(img).convert("HSV"))

    total_pixels += image_array_rgb.shape[0] * image_array_rgb.shape[1]

    for i in range(0, image_array_rgb.shape[0]):
        for j in range(0, image_array_rgb.shape[1]):
            rgb = (image_array_rgb[i,j,0],
                   image_array_rgb[i,j,1],
                   image_array_rgb[i,j,2])
            hsv = (image_array_hsv[i,j,0],
                   image_array_hsv[i,j,1],
                   image_array_hsv[i,j,2])

            if rgb in hist_rgb:
                hist_rgb[rgb] += 1
            else:
                hist_rgb[rgb] = 1

            if hsv in hist_hsv:
                hist_hsv[hsv] += 1
            else:
                hist_hsv[hsv] = 1


    return (total_pixels, hist_rgb, hist_hsv)


def slice_hist(hist):
    hist_xy = dict()
    hist_xz = dict()
    hist_yz = dict()

    for key in hist.keys():
        if (key[0], key[1]) in hist_xy:
            hist_xy[(key[0], key[1])] += hist[key]
        else:
            hist_xy[(key[0], key[1])] = hist[key]

        if (key[0], key[2]) in hist_xz:
            hist_xz[(key[0], key[2])] += hist[key]
        else:
            hist_xz[(key[0], key[2])] = hist[key]

        if (key[1], key[2]) in hist_yz:
            hist_yz[(key[1], key[2])] += hist[key]
        else:
            hist_yz[(key[1], key[2])] = hist[key]

    logger.debug("XY hist created - {}".format(hist_xy))
    logger.debug("XZ hist created - {}".format(hist_xz))
    logger.debug("YZ hist created - {}".format(hist_yz))
    return (hist_xy, hist_xz, hist_yz)


if __name__ == "__main__":
    # Change this if you want more or fewer logging messages.

    logger.info("Andrew Quinn - EECS 332, MP 4 - Histogram training module")
    logger.info("-" * 80)

    big_total_pixels = 0
    big_hist_rgb = {}
    big_hist_hsv = {}

    logger.warning("Constructing un-normalized histograms for directory (this might take a while): {}".format(training_data_dir))

    for path, subdirs, files in os.walk(training_data_dir):
        for name in files:
            img = os.path.join(path, name)
            logger.debug("Now analyzing {}".format(img))
            logger.debug("Constructing individual for {}".format(img))
            (img_total_pixels, img_hist_rgb, img_hist_hsv) = img2hists(img)
            logger.debug("Histogram construction complete for {}".format(img))
            logger.debug("Total pixels :: {}".format(img_total_pixels))
            logger.debug("RGB individual histogram dict :: {}".format(img_hist_rgb))
            logger.debug("HSV individual histogram dict :: {}".format(img_hist_hsv))

            logger.debug("Adding to cumulative histogram data for {}".format(img))
            (big_total_pixels, big_hist_rgb, big_hist_hsv) = img2hists(img, big_hist_rgb, big_hist_hsv, big_total_pixels)
            logger.debug("Histogram construction complete for {}".format(img))
            logger.debug("Total cumulative pixels :: {}".format(big_total_pixels))
            logger.debug("RGB cumulative histogram dict :: {}".format(big_hist_rgb))
            logger.debug("HSV cumulative histogram dict :: {}".format(big_hist_hsv))
            logger.debug("  -> File competed: {}".format(img))

    logger.info("Non-normalized histograms have been constructed for directory: {}".format(training_data_dir))
    logger.info("Constructiong 2-tuple slices of RGB and HSV hists.")
    (big_hist_rg, big_hist_rb, big_hist_gb) = slice_hist(big_hist_rgb)
    (big_hist_hs, big_hist_hv, big_hist_sv) = slice_hist(big_hist_hsv)

    rgb_count_check = 0
    for key in big_hist_rgb.keys():
        rgb_count_check += big_hist_rgb[key]

    rg_count_check = 0
    for key in big_hist_rg.keys():
        rg_count_check += big_hist_rg[key]

    rb_count_check = 0
    for key in big_hist_rb.keys():
        rb_count_check += big_hist_rb[key]

    gb_count_check = 0
    for key in big_hist_gb.keys():
        gb_count_check += big_hist_gb[key]

    hs_count_check = 0
    for key in big_hist_hs.keys():
        hs_count_check += big_hist_hs[key]

    hv_count_check = 0
    for key in big_hist_hv.keys():
        hv_count_check += big_hist_hv[key]

    sv_count_check = 0
    for key in big_hist_sv.keys():
        sv_count_check += big_hist_sv[key]

    hsv_count_check = 0
    for key in big_hist_hsv.keys():
        hsv_count_check += big_hist_hsv[key]

    try:
        if (rgb_count_check != hsv_count_check or
            rgb_count_check != big_total_pixels or
            hsv_count_check != big_total_pixels):
            raise ValueError
    except ValueError:
        logger.warning("Histogram counts don't match up!")
        logger.warning("   -> big_total_pixels = {}".format(big_total_pixels))
        logger.warning("   -> rgb_count_check  = {}".format(rgb_count_check))
        logger.warning("   -> hsv_count_check  = {}".format(hsv_count_check))

    try:
        if (hs_count_check != big_total_pixels or
            hv_count_check != big_total_pixels or
            sv_count_check != big_total_pixels):
            raise ValueError
    except ValueError:
            logger.warning("HSV 2-tuple slice counts don't match up!")
            logger.warning("   -> big_total_pixels = {}".format(big_total_pixels))
            logger.warning("   -> hv_count_check   = {}".format(hv_count_check))
            logger.warning("   -> hs_count_check   = {}".format(hs_count_check))
            logger.warning("   -> sv_count_check   = {}".format(sv_count_check))

    try:
        if (rb_count_check != big_total_pixels or
            rg_count_check != big_total_pixels or
            gb_count_check != big_total_pixels):
            raise ValueError
    except ValueError:
            logger.warning("RGB 2-tuple slice counts don't match up!")
            logger.warning("   -> big_total_pixels = {}".format(big_total_pixels))
            logger.warning("   -> rg_count_check   = {}".format(rg_count_check))
            logger.warning("   -> rb_count_check   = {}".format(rb_count_check))
            logger.warning("   -> gb_count_check   = {}".format(gb_count_check))


    logger.info("Pickling non-normalized histogram dicts with 'size' key added.")
    big_hist_rgb['size'] = big_total_pixels
    big_hist_hsv['size'] = big_total_pixels
    big_hist_rg['size'] = big_total_pixels
    big_hist_rb['size'] = big_total_pixels
    big_hist_gb['size'] = big_total_pixels
    big_hist_hs['size'] = big_total_pixels
    big_hist_hv['size'] = big_total_pixels
    big_hist_sv['size'] = big_total_pixels

    rgb_hist_location = os.path.join(hist_output_dir, "hist_rgb.pickle")
    hsv_hist_location = os.path.join(hist_output_dir, "hist_hsv.pickle")

    rg_hist_location = os.path.join(hist_output_dir, "hist_rg.pickle")
    rb_hist_location = os.path.join(hist_output_dir, "hist_rb.pickle")
    gb_hist_location = os.path.join(hist_output_dir, "hist_gb.pickle")
    hs_hist_location = os.path.join(hist_output_dir, "hist_hs.pickle")
    hv_hist_location = os.path.join(hist_output_dir, "hist_hv.pickle")
    sv_hist_location = os.path.join(hist_output_dir, "hist_sv.pickle")

    with open(rgb_hist_location, 'wb') as fp:
        pickle.dump(big_hist_rgb, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_rgb pickled to {}".format(fp))

    with open(hsv_hist_location, 'wb') as fp:
        pickle.dump(big_hist_hsv, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_hsv pickled to {}".format(fp))

    with open(rg_hist_location, 'wb') as fp:
        pickle.dump(big_hist_rg, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_rg pickled to {}".format(fp))

    with open(rb_hist_location, 'wb') as fp:
        pickle.dump(big_hist_rb, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_rb pickled to {}".format(fp))

    with open(gb_hist_location, 'wb') as fp:
        pickle.dump(big_hist_gb, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_gb pickled to {}".format(fp))

    with open(hs_hist_location, 'wb') as fp:
        pickle.dump(big_hist_hs, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_hs pickled to {}".format(fp))

    with open(hv_hist_location, 'wb') as fp:
        pickle.dump(big_hist_hv, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_hv pickled to {}".format(fp))

    with open(sv_hist_location, 'wb') as fp:
        pickle.dump(big_hist_sv, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("big_hist_sv pickled to {}".format(fp))

    logger.info("Histogram data for RGB and HSV has been pickled.")
    logger.warning("Removing 'size' key from histograms.")
    del big_hist_rgb['size']
    del big_hist_hsv['size']
    del big_hist_rg['size']
    del big_hist_rb['size']
    del big_hist_gb['size']
    del big_hist_hs['size']
    del big_hist_hv['size']
    del big_hist_sv['size']
