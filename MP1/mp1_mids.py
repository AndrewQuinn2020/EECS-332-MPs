#!/usr/bin/python3

import os
import sys
from PIL import Image
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


script_dir = os.path.dirname(__file__)
mids_dir = os.path.join(script_dir, "mid_tests")

mid_test_files = [f for f in os.listdir(mids_dir)]
mid_test_files_abs = sorted(list(map(lambda x: os.path.join(mids_dir, x),
                            mid_test_files)))


rel_image_paths = ['face.bmp',
                   'gun.bmp',
                   'test.bmp']

abs_image_paths = list(map(lambda s: os.path.join(script_dir, s),
                       rel_image_paths))


def check_pxl(image, regions, x, y, new_region, eq_classes=None, verbose=False,
              extremely_verbose=False):
    """Returns, based on the current region map, which potential region
    the new pixel might be in. Also takes note of equivalence classes,
    if a variable is passed for that, if it happens upon any in the course
    of running."""

    def check_for_eq(extremely_verbose=False):
        if y > 0:
            adjacencies_differ = regions[x-1, y] != regions[x, y-1]

            if image[x, y] == image[x, y-1] and adjacencies_differ:
                if extremely_verbose:
                    print("Possible equivalency found: (x, y, image)")
                    print("({}, {}, {})".format(x,y-1,image[x,y-1]).rjust(12))
                    print("({}, {}, {})".format(x,y,image[x,y]).rjust(12))
                    print("({}, {}, {})".format(x-1,y,image[x-1,y]).rjust(12))
                eq_classes.append((min(regions[x, y-1], regions[x-1, y]),
                                   max(regions[x, y-1], regions[x-1, y])))
        return None

    assert image.shape == regions.shape

    if x > 0:
        if image[x, y] == image[x-1, y]:
            if verbose:
                print("^^^^^^^^^^^ {} {}".format(x, y))
            if eq_classes is not None:
                check_for_eq(extremely_verbose=extremely_verbose)
            if verbose:
                print("Returning {}".format(regions[x-1, y]))
            return regions[x-1, y]

    if y > 0:
        if image[x, y] == image[x, y - 1]:
            return regions[x, y - 1]

    return new_region


def set_initial_regions(image, region, verbose=False):
    assert image.shape == region.shape

    eqs = []

    next_region = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if verbose:
                print("Region [{}, {}] is beginning at {}".format(i, j, region[i, j]))
            region[i, j] = check_pxl(image, region, i, j, next_region, eqs)
            if verbose:
                print("Region [{}, {}] is ending at {}".format(i, j, region[i, j]))
            if region[i, j] == next_region:
                next_region += 1

    return eqs


def eq_reducer(eqs, verbose=False):
    eqs = sorted(eqs)

    if len(eqs) < 2:
        return eqs
    else:
        head = eqs[0]
        tail = eqs[1:]
        ts = list(filter(lambda t: head[1] == t[0] or head[1] == t[1], tail))
        if verbose:
            print("Tail is {}".format(tail))
            print("Head is {}, filtered tail is {}".format(head, ts))
        for t in ts:
            if head == t:
                if verbose:
                    print("  Duplicate deleted.")
                tail.remove(t)
            elif head[1] == t[0]:
                if verbose:
                    print("  {} is being removed, and {} appended.".format(t, (head[0], t[1])))
                tail.remove(t)
                tail.append((head[0], t[1]))
            elif head[1] == t[1]:
                if verbose:
                    print("  {} is being removed, and {} appended.".format(t, (head[0], t[0])))
                tail.remove(t)
                tail.append((head[0], t[0]))
        return [head] + eq_reducer(tail, verbose=verbose)


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
    region_swapper = dict()
    region_code = regions[0, 0]

    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            if regions[i, j] > region_code:
                if regions[i, j] in region_swapper.keys():
                    regions[i, j] = region_swapper.get(regions[i, j])
                else:
                    region_swapper[regions[i, j]] = region_code+1
                    regions[i, j] = region_code+1
                    region_code = region_code+1
    return region_code


def regions_to_grayscale(old_regions, new_grayscale, num_regions):
        for i in range(0, old_regions.shape[0]):
            for j in range(0, old_regions.shape[1]):
                new_grayscale[i, j] = int(255 - (old_regions[i, j] * (255 / num_regions)))
        return None


if __name__ == "__main__":
    for mid in mid_test_files_abs:
        print("Regions of: ", mid)

        p_orig = np.array(Image.open(mid).convert('L')).astype(int)
        p_regions = np.zeros(p_orig.shape, dtype=int)

        eqs_orig = set_initial_regions(p_orig, p_regions)

        reduced_eq = eq_reducer(eqs_orig)

        regions_finalizer(p_regions, reduced_eqs_to_dict(reduced_eq))

        number_of_regions = file_off_the_serials(p_regions)
        print("Regions total: ", number_of_regions)
        print(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        print(p_grayscale)

        mids_gs_dir = os.path.join(script_dir, "mid_tests_gs")
        gs_loc = os.path.join(mids_gs_dir, os.path.splitext(mid)[0][-7:] + "_gs.bmp")

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'P')
        im.save(gs_loc)

    for img in abs_image_paths:
        print("Regions of: ", img)

        p_orig = np.array(Image.open(img).convert('L')).astype(int)
        p_regions = np.zeros(p_orig.shape, dtype=int)

        eqs_orig = set_initial_regions(p_orig, p_regions)

        reduced_eq = eq_reducer(eqs_orig)

        regions_finalizer(p_regions, reduced_eqs_to_dict(reduced_eq))

        number_of_regions = file_off_the_serials(p_regions)
        print("Regions total: ", number_of_regions)
        # print(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        # print(p_grayscale)
        gs_loc = img[:-4] + "_gs.bmp"
        print(gs_loc)

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'L')
        im.save(gs_loc)
