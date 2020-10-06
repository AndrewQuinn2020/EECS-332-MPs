#!/usr/bin/python3


import os
import sys
from PIL import Image
import numpy as np


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
mids_dir = os.path.join(script_dir, "tests")


if __name__ == "__main__":
    print("Andrew Quinn - EECS 332 - MP#2\n" + ("-" * 80))
