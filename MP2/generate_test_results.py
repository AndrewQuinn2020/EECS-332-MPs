#!/usr/bin/python3

import itertools
import os
import sys
from test import *

import numpy as np
from mp2 import *
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

script_dir = os.path.dirname(__file__)
images_dir = os.path.join(script_dir, "test_images")
images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])


se_identity_1 = np.array([[True]])
se_cross_3 = np.array([[False, True, False], [True, True, True], [False, True, False]])
se_north_3 = np.array(
    [[False, True, False], [False, True, False], [False, False, False]]
)
se_glider_3 = np.array([[False, True, False], [True, False, False], [True, True, True]])

structural_elements = [se_identity_1, se_cross_3, se_north_3]


def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj]


if __name__ == "__main__":
    print("Andrew Quinn - EECS 332 - Generate test results for report")
    print("=" * 80)
