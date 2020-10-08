#!/usr/bin/python3

 # Anything not directly related to processing here
from mp3_helper import *

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


def img2dict(data):
    """Construct a histogram of the count of every value found in the
    2D array data, in a dictionary, along with the minimum and maximum
    greyscale value it found in the data."""
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


def cmd2plottable(cmd_in):
    """Given the output of matrix2cmd, constructs a 256*2 matrix for plotting
    the cumulative distribution function."""
    matrix_out = np.zeros((256, 2))

    cumval = 0
    nextval = 0
    for i in range(0, 256):
        matrix_out[i, 0] = i
        matrix_out[i, 1] = cumval
        if matrix_out[i, 0] == cmd_in[nextval, 0]:
            matrix_out[i, 1] = cmd_in[nextval, 1]
            cumval = matrix_out[i, 1]
            if nextval < cmd_in.shape[0] - 1:
                nextval += 1

    return matrix_out

def cmd2dict(cmd):
    """Returns a dictionary of what to replace each value by."""
    pixel_count = cmd[cmd.shape[0]-1, cmd.shape[1]-1]
    scaling_dict = dict()

    for i in range(0, cmd.shape[0]):
        scaling_dict[cmd[i, 0]] = round(((cmd[i, 1] - cmd[0,1])/(pixel_count - cmd[0,1])) * 255)
    return scaling_dict




if __name__ == "__main__":
    hello()

    for image in test_images:
        img_data = load_gs(image)
        hist_data = img2dict(img_data)

        hist = hist2matrix(hist_data)
        hist_cmd = matrix2cmd(hist)
        plottable_cmd = cmd2plottable(hist_cmd)
        hist_eq_dict = cmd2dict(hist_cmd)

        results_data = np.zeros_like(img_data).astype(int)

        for i in range(0, results_data.shape[0]):
            for j in range(0, results_data.shape[1]):
                results_data[i,j] = hist_eq_dict[img_data[i,j]]

        results_hist = img2dict(results_data)
        results_hist_matrix = hist2matrix(results_hist)
        results_hist_cmd = matrix2cmd(results_hist_matrix)
        plottable_results_cmd = cmd2plottable(results_hist_cmd)


        test_results_path = save_gs(results_data, Path(image).stem,
                                    dir=test_results_dir)

        print("Processed image saved to: {}".format(test_results_path))

        # Let's plot some histograms.
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("Pixels per color value: `{}`".format(Path(image).stem + ".bmp"))
        ax1.set_xlabel("Pixel color value ($v \in \{0, 1, \dots, 255\}$)")
        ax1.set_ylabel("# / pixels per grayscale value")
        ln1 = ax1.bar(hist_data.keys(), hist_data.values(), alpha=0.6, color='r', label="Original")
        ln2 = ax1.bar(results_hist.keys(), results_hist.values(), alpha=0.6, color='b', label="EQ'd")

        ax2 = ax1.twinx()
        ln3 = ax2.bar(plottable_cmd[:,0], plottable_cmd[:,1], alpha=0.1,
                          color='g', label="cdf (Original)")
        ln4 = ax2.bar(plottable_results_cmd[:,0], plottable_results_cmd[:,1],
                          alpha=0.1, color='purple', label="cdf (EQ'd)")

        plt.legend([ln1, ln2, ln3, ln4], [ln1.get_label(), ln2.get_label(), ln3.get_label(), ln4.get_label()])
        plt.tight_layout()

        plt.savefig(os.path.join(test_results_dir, Path(image).stem + "_hist.svg"))
        plt.savefig(os.path.join(test_results_dir, Path(image).stem + "_hist.jpg"))
        plt.close()

    for image in images:
        img_data = load_gs(image)
        hist_data = img2dict(img_data)

        hist = hist2matrix(hist_data)
        hist_cmd = matrix2cmd(hist)
        plottable_cmd = cmd2plottable(hist_cmd)

        hist_eq_dict = cmd2dict(hist_cmd)

        results_data = np.zeros_like(img_data).astype(int)

        for i in range(0, results_data.shape[0]):
            for j in range(0, results_data.shape[1]):
                results_data[i,j] = hist_eq_dict[img_data[i,j]]

        results_hist = img2dict(results_data)
        results_hist_matrix = hist2matrix(results_hist)
        results_hist_cmd = matrix2cmd(results_hist_matrix)
        plottable_results_cmd = cmd2plottable(results_hist_cmd)

        results_path = save_gs(results_data, Path(image).stem)
        print("Processed image saved: {}".format(results_path))

        # Let's plot some histograms.
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title("Pixels per color value: `{}`".format(Path(image).stem + ".bmp"))
        ax1.set_xlabel("Pixel color value ($v \in \{0, 1, \dots, 255\}$)")
        ax1.set_ylabel("Number of pixels")
        ln1 = ax1.bar(hist_data.keys(), hist_data.values(), alpha=0.6, color='r', label="Original")
        ln2 = ax1.bar(results_hist.keys(), results_hist.values(), alpha=0.6, color='b', label="EQ'd")

        ax2 = ax1.twinx()
        ln3 = ax2.bar(plottable_cmd[:,0], plottable_cmd[:,1], alpha=0.1,
                          color='g', label="cdf (Original)")
        ln4 = ax2.bar(plottable_results_cmd[:,0], plottable_results_cmd[:,1],
                          alpha=0.1, color='purple', label="cdf (EQ'd)")

        plt.legend([ln1, ln2, ln3, ln4], [ln1.get_label(), ln2.get_label(), ln3.get_label(), ln4.get_label()])
        plt.tight_layout()

        plt.savefig(os.path.join(results_dir, Path(image).stem + "_hist.svg"))
        plt.savefig(os.path.join(results_dir, Path(image).stem + "_hist.jpg"))
        plt.close()
