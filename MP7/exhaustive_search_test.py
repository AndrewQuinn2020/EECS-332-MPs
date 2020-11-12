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


def ssd(template, candidate):
    return ((template - candidate) ** 2).sum()


if __name__ == "__main__":
    logger.info("Andrew Quinn - EECS 332 - Machine Problem #7")
    logger.info("Testing the exhaustive search code.")
    logger.info("-" * (88 - 11))

    field = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 0, 3, 3],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ]
    )

    template = np.zeros((3, 3))

    assert exhaustive_search(lambda a, b: 0, template, field)[3] == 9

    template_2 = np.array([[2, 2, 2], [3, 0, 3], [4, 4, 4]])

    min_f, i, j, min_count = exhaustive_search(ssd, template_2, field)

    candidate = field[i : i + template_2.shape[0], j : j + template_2.shape[1]]
    print(candidate)
    print(template_2 - candidate)
