#!/usr/bin/python3

 # Anything not directly related to processing here
from mp3_helper import *

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from math import floor


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


def img2dict(data):
    """Construct a histogram of the count of every value found in the
    2D array data, in a dictionary."""
    hist_data = {}
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            if data[i, j] not in hist_data.keys():
                hist_data[data[i, j]] = 1
            else:
                hist_data[data[i, j]] += 1
    return hist_data


def hist2matrix(hist):
    """Returns a |hist|-by-2 matrix, with keys on top and values on bottom."""

    m = np.zeros((len(hist), 2)).astype(int)
    # print(m.shape)

    c = 0
    for key in sorted(hist.keys()):
        m[c, 0] = key
        m[c, 1] = hist[key]
        c += 1

    return m

def matrix2cmd(matrix_in):
    """Given an n*2 matrix of keys in column 0 and integer values in column
    1, returns a new n*2 matrix of the same keys, but the values have been
    summed up."""

    matrix_out = np.zeros_like(matrix_in)
    c = 0
    for i in range(0,matrix_in.shape[0]):
        c += matrix_in[i, 1]
        # print(c)
        matrix_out[i, 0] = matrix_in[i, 0]
        matrix_out[i, 1] = c
    return matrix_out


def cmd2dict(cmd_in):
    """Returns a dictionary of what to replace each value by."""
    cmd_max = cmd_in[cmd_in.shape[0]-1, cmd_in.shape[1]-1]
    scaling_dict = dict()

    for i in range(0, cmd_in.shape[0]):
        scaling_dict[cmd_in[i, 0]] = floor(cmd_in[i, 0] * (cmd_in[i, 1] / cmd_max))

    return scaling_dict




if __name__ == "__main__":
    hello()

    for image in test_images:
        img_data = load_gs(image)
        hist_data = img2dict(img_data)
        # plt.bar(list(hist_data.keys()), hist_data.values(), color='g')
        # plt.show(block=False)
        # plt.close()

        hist = hist2matrix(hist_data)
        # plt.bar(list(hist[:,0]), list(hist[:,1]))
        # plt.show(block=False)

        hist_cmd = matrix2cmd(hist)
        # print(hist)
        print(hist_cmd)

        hist_eq_dict = cmd2dict(hist_cmd)
        print(hist_eq_dict)

        results_data = np.zeros_like(img_data).astype(int)

        for i in range(0, results_data.shape[0]):
            for j in range(0, results_data.shape[1]):
                results_data[i,j] = hist_eq_dict[img_data[i,j]]

        print(results_data)

        test_results_path = save_gs(results_data, Path(image).stem,
                                    dir=test_results_dir)
        print("Processed image saved: {}".format(test_results_path))

    for image in images:
        img_data = load_gs(image)
        hist_data = img2dict(img_data)
        # plt.bar(list(hist_data.keys()), hist_data.values(), color='g')
        # plt.show(block=False)
        # plt.close()

        hist = hist2matrix(hist_data)
        # plt.bar(list(hist[:,0]), list(hist[:,1]))
        # plt.show(block=False)

        hist_cmd = matrix2cmd(hist)
        # print(hist)
        # print(hist_cmd)

        hist_eq_dict = cmd2dict(hist_cmd)

        results_data = np.zeros_like(img_data).astype(int)

        for i in range(0, results_data.shape[0]):
            for j in range(0, results_data.shape[1]):
                results_data[i,j] = hist_eq_dict[img_data[i,j]]

        # print(results_data)

        results_path = save_gs(results_data, Path(image).stem)
        print("Processed image saved: {}".format(results_path))
