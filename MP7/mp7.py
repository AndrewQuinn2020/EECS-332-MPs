#!/usr/bin/python3

# mp5.py

import logging
import os
import sys
from itertools import product
from functools import reduce

import colorlog
from PIL import Image
import numpy as np
from scipy import signal
import cv2 as cv

logger = logging.getLogger(__name__)
# Change this to get more, or fewer, error messages.
#   DEBUG = Show me everything.
#   INFO = Only the green text and up.
#   WARNING = Only warnings.
#   ERROR = Only (user coded) error messages.
#   CRITICAL = Only (user coded) critical error messages.
logger.setLevel(colorlog.colorlog.logging.DEBUG)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=10000)
np.set_printoptions(precision=4)

# Some constants so that we know where all of our stuff is.
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, "images")
results_dir = os.path.join(script_dir, "results")

BOUNDING_BOX_SIZE_X = 5
BOUNDING_BOX_SIZE_Y = 5
STARTING_X = 55
STARTING_Y = 25

dirs = [script_dir, images_dir, results_dir]


def subimage(img, x, y):
    return img[y : y + BOUNDING_BOX_SIZE_Y, x : x + BOUNDING_BOX_SIZE_X].copy()


def exhaustive_search(f, template, candidate_field, verbose=False):
    """Applies f(template, candidate) to every candidate in candidate_field."""
    if not verbose:
        logger.disabled = True

    min_f, min_i, min_j, count = np.inf, 0, 0, 0

    logger.debug(candidate_field.shape)
    logger.debug(template.shape)
    for i in range(candidate_field.shape[0] - template.shape[0] + 1):
        for j in range(candidate_field.shape[1] - template.shape[1] + 1):
            count += 1
            candidate = candidate_field[
                i : i + template.shape[0], j : j + template.shape[1]
            ]
            assert candidate.shape == template.shape
            if f(template, candidate) <= min_f:
                logger.debug("New f found! {}".format(f(template, candidate)))
                min_f = f(template, candidate)
                min_i = i
                min_j = j

    if not verbose:
        logger.disabled = False

    return (min_f, min_i, min_j, count)


def zero_diff(template, candidate):
    return 0


def ssd(template, candidate):
    return (template - candidate).sum() ** 2


def draw_bounding_box(img, x, y):
    """Returns an image with a bounding box drawn over it."""
    return cv.rectangle(
        img,
        (x, y),
        (x + BOUNDING_BOX_SIZE_X, y + BOUNDING_BOX_SIZE_Y),
        (200, 200, 100),
        2,
    )


if __name__ == "__main__":
    logger.info("Andrew Quinn - EECS 332 - Machine Problem #7")
    logger.info("-" * (88 - 11))

    for dir in dirs:
        if not os.path.exists(dir):
            logger.warning("\t\t{}\t\t doesn't exist... Creating.".format(dir))
            os.makedirs(dir)

    count = 3
    for path, subdirs, files in os.walk(images_dir):
        first_image = True
        for og_name in sorted(files):
            logger.info("Now working on:      {}".format(os.path.join(path, og_name)))
            img = cv.imread(os.path.join(path, og_name), cv.IMREAD_GRAYSCALE)
            location_out = os.path.join(results_dir, og_name)
            template_out = os.path.join(results_dir, og_name[:-4] + "_template.jpg")

            if first_image:
                template = subimage(img, STARTING_X, STARTING_Y)
                draw_bounding_box(img, STARTING_X, STARTING_Y)
                first_image = False

            else:
                f, next_x, next_y, count = exhaustive_search(zero_diff, template, img)
                assert count == (img.shape[0] - template.shape[0] + 1) * (
                    img.shape[1] - template.shape[1] + 1
                )
                logger.info("  Exhaustive search complete!")
                logger.info(
                    "  Next boundary box will start at ({}, {}).".format(next_x, next_y)
                )
                template = subimage(img, next_x, next_y)
                draw_bounding_box(img, next_x, next_y)

            # if type(prev_img) is not type(img):
            #     prev_img = img
            #     draw_bounding_box(img, 55, 25)
            #     comparison_img = subimage(img, 55, 25)
            # else:
            #     prev_img = img
            #
            # print(exhaustive_search(zero_diff, comparison_img, prev_img))
            # print(exhaustive_search(ssd, comparison_img, prev_img))
            # print(prev_img.shape)
            #
            logger.warning("Writing template to:   {}".format(template_out))
            cv.imwrite(template_out, template)
            logger.warning("Writing results to:    {}".format(location_out))
            cv.imwrite(location_out, img)

            count -= 1
            if count < 1:
                break
