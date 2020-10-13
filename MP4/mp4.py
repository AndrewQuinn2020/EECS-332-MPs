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
import colorlog

logger = logging.getLogger()
logger.setLevel(colorlog.colorlog.logging.INFO)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=10000)

script_dir = os.path.dirname(__file__)
hist_output_dir = os.path.join(script_dir,
                               "histogram_data")


if __name__ == "__main__":
    print("Andrew Quinn - EECS 332, MP 4")
    print("-" * 80)
