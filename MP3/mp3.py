#!/usr/bin/python3

 # Anything not directly related to processing here
from mp3_helper import *

from PIL import Image
import numpy as np
import sys
from pathlib import Path


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


if __name__ == "__main__":
    hello()

    for image in images:
        img_data = load_gs(image)
        print(img_data)
        print(Path(image).stem)
        print(save_gs(img_data, Path(image).stem))
