#!/usr/bin/python3

import os
import sys
from PIL import Image
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

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


def size_filter_binary(old_binary, new_binary):
    def any_neighbors_like_me(old_binary, x, y):
        c = 0
        if x > 0 and old_binary[x-1, y] == old_binary[x, y]:
            c = c + 1
        if x < old_binary.shape[0]-1 and old_binary[x+1, y] == old_binary[x, y]:
            c = c + 1
        if y > 0 and old_binary[x, y-1] == old_binary[x, y]:
            c = c + 1
        if y < old_binary.shape[1]-1 and old_binary[x, y+1] == old_binary[x, y]:
            c = c + 1
        return c > 0

    pixels_deleted = 0

    for i in range(0, old_binary.shape[0]):
        for j in range(0, old_binary.shape[1]):
            if any_neighbors_like_me(old_binary, i, j):
                new_binary[i, j] = old_binary[i, j]
            else:
                pixels_deleted = pixels_deleted + 1
                new_binary[i, j] = not old_binary[i, j]

    return pixels_deleted


def size_filter(old, new, n=1, complete=True):
    if n == 1:
        return size_filter_binary(old, new)
    else:
        def find_boundary_sizes(regions):
            boundary_sizes = {0:0}

            for i in range(0, regions.shape[0]):
                for j in range(0, regions.shape[1]):
                    if old[i, j] not in boundary_sizes.keys():
                        boundary_sizes[regions[i, j]] = 1
                    else:
                        boundary_sizes[regions[i,j]] += 1
            return boundary_sizes

        bs1 = find_boundary_sizes(old)

        if np.array_equal(old, new):
            for i in range(0, old.shape[0]):
                for j in range(0, old.shape[1]):
                    if bs1[old[i, j]] < n:
                        new[i, j] = 0
                    else:
                        new[i, j] = old[i, j]
        else:
            for i in range(0, old.shape[0]):
                for j in range(0, old.shape[1]):
                    if bs1[old[i, j]] < n:
                        new[i, j] = 0

        bs2 = find_boundary_sizes(new)
        return (bs1, bs2)



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

    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            regions[i, j] = regions[i, j] << 4

    region_code = 0

    print(regions)

    region_codebook = {0: 0}

    for i in range(0, regions.shape[0]):
        for j in range(0, regions.shape[1]):
            if regions[i, j] not in region_codebook.keys():
                region_codebook[regions[i, j]] = region_code+1
                region_code = region_code+1
                print("New region code added. {}".format(region_codebook))
            regions[i, j] = region_codebook[regions[i, j]]

    return region_code+1


def regions_to_grayscale(old_regions, new_grayscale, num_regions):
        for i in range(0, old_regions.shape[0]):
            for j in range(0, old_regions.shape[1]):
                new_grayscale[i, j] = int(255 - (old_regions[i, j] * (255 / num_regions)))
        return None


def regions_to_txt(old_regions):
        for i in range(0, old_regions.shape[0]):
            for j in range(0, old_regions.shape[1]):
                print(old_regions[i, j], end="")
            print("")
        return None


if __name__ == "__main__":

    if not os.path.exists(os.path.join(script_dir, 'mid_tests_gs')):
        os.makedirs(os.path.join(script_dir, 'mid_tests_gs'))

    for mid in mid_test_files_abs:
        print("Regions of: ", mid)

        p_orig = np.array(Image.open(mid).convert('L')).astype(int)
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
        print("EQ: {}".format(reduced_eq))
        print("Pre-finalization...")
        print(p_regions)

        print("Post-finalization...")
        regions_finalizer(p_regions, reduced_eqs_to_dict(reduced_eq))
        print(p_regions)

        number_of_regions = file_off_the_serials(p_regions)
        print("Regions total: ", number_of_regions)
        regions_to_txt(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        # print(p_grayscale)
        gs_loc = img[:-4] + "_gs.bmp"
        print(gs_loc)

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'L')
        im.save(gs_loc)

    for img in [os.path.join(script_dir, 'gun.bmp')]:
        print(img)

        p_orig = np.array(Image.open(img))
        p_filtered = np.zeros_like(p_orig)
        print(p_orig)
        print(p_filtered)

        print("Single pixels filtered from gun.bmp:", size_filter(p_orig,
                                                                  p_filtered))

        filtered_loc = os.path.join(script_dir, 'gun_size_filtered.bmp')
        im = Image.fromarray(p_filtered)
        im.save(filtered_loc)

        p_orig = np.array(Image.open(filtered_loc).convert('L')).astype(int)
        p_regions = np.zeros(p_orig.shape, dtype=int)

        eqs_orig = set_initial_regions(p_orig, p_regions)

        reduced_eq = eq_reducer(eqs_orig)
        print("EQ: {}".format(reduced_eq))
        print("Pre-finalization...")
        print(p_regions)

        print("Post-finalization...")
        regions_finalizer(p_regions, reduced_eqs_to_dict(reduced_eq))
        print(p_regions)

        number_of_regions = file_off_the_serials(p_regions)
        print("Regions total: ", number_of_regions)
        regions_to_txt(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        # print(p_grayscale)
        gs_loc = filtered_loc[:-4] + "_gs.bmp"
        print(gs_loc)

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'L')
        im.save(gs_loc)


        print(size_filter(p_regions, p_regions, n=50))
        number_of_regions = file_off_the_serials(p_regions)
        print(size_filter(p_regions, p_regions, n=50))

        print("Regions total: ", number_of_regions)
        regions_to_txt(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        # print(p_grayscale)
        filtered_loc = os.path.join(script_dir, 'gun_size_filtered_hard.bmp')

        gs_loc = filtered_loc[:-4] + "_gs.bmp"
        print(gs_loc)

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'L')
        im.save(gs_loc)


        print(size_filter(p_regions, p_regions, n=250))
        number_of_regions = file_off_the_serials(p_regions)
        print(size_filter(p_regions, p_regions, n=250))

        print("Regions total: ", number_of_regions)
        regions_to_txt(p_regions)

        p_grayscale = np.zeros(p_regions.shape, dtype=int)

        regions_to_grayscale(p_regions, p_grayscale, number_of_regions)
        # print(p_grayscale)
        filtered_loc = os.path.join(script_dir, 'gun_size_filtered_very_hard.bmp')

        gs_loc = filtered_loc[:-4] + "_gs.bmp"
        print(gs_loc)

        im = Image.fromarray(p_grayscale.astype(np.uint8), 'L')
        im.save(gs_loc)
