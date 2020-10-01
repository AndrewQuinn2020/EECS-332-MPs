#!/usr/bin/python3

import os, sys
from PIL import Image
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

script_dir = os.path.dirname(__file__)

rel_image_paths = ['small_tests/test_01.bmp',
                   'small_tests/test_02.bmp',
                   'small_tests/test_03.bmp',
                   'small_tests/test_04.bmp',
                   'small_tests/test_05.bmp',
                   'small_tests/test_06.bmp',
                   'small_tests/test_07.bmp',
                   'small_tests/test_08.bmp',
                   'small_tests/test_09.bmp',
                   'small_tests/test_10.bmp',
                   'face.bmp',
                   'gun.bmp',
                   'test.bmp']

abs_image_paths = list(map(lambda s: os.path.join(script_dir, s),
                           rel_image_paths))

def check_pxl(image, regions, x, y, new_region, eq_classes=None):
    """Returns, based on the current region map, which potential region
    the new pixel might be in. Also takes note of equivalence classes,
    if a variable is passed for that, if it happens upon any in the course
    of running."""

    def check_for_eq(report=False):
        if x > 0:
            adjacencies_differ = regions[x-1, y] != regions[x, y-1]

            if image[x, y] == image[x-1, y] and adjacencies_differ:
                if report:
                    print("Possible equivalency found: (x, y, image)")
                    print("({}, {}, {})".format(x,y-1,image[x,y-1]).rjust(12))
                    print("({}, {}, {})".format(x,y,image[x,y]).rjust(12))
                    print("({}, {}, {})".format(x-1,y,image[x-1,y]).rjust(12))
                eq_classes.append((min(regions[x, y-1], regions[x-1, y]),
                                   max(regions[x, y-1], regions[x-1, y])))
        return None

    assert image.shape == regions.shape

    if y > 0:
        if image[x, y] == image[x, y-1]:
            if eq_classes is not None:
                check_for_eq()
            return regions[x, y - 1]

    if x > 0:
        if image[x, y] == image[x - 1, y]:
            return regions[x - 1, y]

    return new_region



def set_initial_regions(image, region):
    assert image.shape == region.shape

    eqs = []

    next_region = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            region[i, j] = check_pxl(image, region, i, j, next_region, eqs)
            if region[i, j] == next_region:
                next_region += 1

    return (next_region, eqs)


def eq_reduce(eq_classes, rdepth=1):
    """Given a bunch of eq classes in the (lower, higher) format I was
    using, construct disjoint sets of the equivalent regions."""

    justify = "  " * (rdepth - 1)
    print(justify + "RECURSIVE DEPTH :: {}".format(rdepth))
    eq_classes = sorted(list(set(eq_classes)))

    if len(eq_classes) < 2:
        print(justify + "Base case.")
        return eq_classes

    else:
        head = eq_classes[0]
        tail = eq_classes[1:]
        print(justify + "Head = {}, tail = {}".format(head, tail))
        for i in range(0, len(tail)):
            if head[1] == tail[i][1]:
                print(justify + "  " + "CASE 1 :: " + "Deleting {} and inserting {}.".format(tail[i], (head[0], tail[i][0])))
                tail.append((head[0], tail[i][0]))
                del tail[i]
            if head[1] == tail[i][0]:
                print(justify + "  " + "CASE 2 :: " + "Deleting {} and inserting {}.".format(tail[i], (head[0], tail[i][1])))
                tail.append((head[0], tail[i][1]))
                del tail[i]
        print("Tail has been reduced to {}.".format(tail))
        return [head] + eq_reduce(tail, rdepth=rdepth+1)


def reduced_eqs_to_dict(reduced_eqs):
    eqs_dict = dict()
    for eq in reduced_eqs:
        eqs_dict[eq[1]] = eq[0]
    return eqs_dict


def regions_finalizer(regions, eq_dict):
    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            if regions[i, j] in eq_dict.keys():
                regions[i, j] = eq_dict.get(regions[i, j])
    return None

def file_off_the_serials(regions):
    x = regions[0, 0]

    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            if regions[i, j] > x:
                x = x + 1
                regions[i, j] = x

    return x

def regions_to_txt(regions):
    pad = len(str(file_off_the_serials(regions)))

    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            print("{}".format(regions[i, j]).rjust(pad), end="")
        print("")

    return None


if __name__ == "__main__":
    print("EECS 332 - MP#1 - Andrew Quinn solution")
    print("---------------------------------------")

    for abs_image_path in abs_image_paths:
        print("Results: {}".format(abs_image_path))
        p_orig = np.array(Image.open(abs_image_path).convert('L')).astype(int)
        p_regions = np.zeros(p_orig.shape, dtype=int)

        init_scan = set_initial_regions(p_orig, p_regions)

        eqs = reduced_eqs_to_dict(eq_reduce(init_scan[1]))
        print("\n\n\n\n\n\n\n\n\n{}\n\n\n\n\n\n\n\n\n\n".format(eqs))
        regions_finalizer(p_regions, eqs)

        for i in range(0, 10):
            file_off_the_serials(p_regions)

        regions_to_txt(p_regions)
