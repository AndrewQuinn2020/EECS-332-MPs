#!/usr/bin/python3

import os.path


script_dir = os.path.dirname(__file__)
images_dir = os.path.join(script_dir, "images")
results_dir = os.path.join(script_dir, 'results')
test_images_dir = os.path.join(script_dir, "test_images")
test_results_dir = os.path.join(script_dir, 'test_results')

subdirs = [images_dir, results_dir, test_images_dir, test_results_dir]


def reveal_images():
    """Reveal the contents of the images/ and test_images/ directory."""
    assert os.path.exists(images_dir)
    assert os.path.exists(test_images_dir)

    print("    Images in images/:")
    for image in [os.path.join(images_dir, f) for f in
                  os.listdir(images_dir)]:
        print("        " + image)

    print("    Images in test_images/:")
    for test_image in [os.path.join(test_images_dir, f) for f in
                       os.listdir(test_images_dir)]:
        print("        " + image)

    return None


def hello():
    """Just a wrapper for some initial print statements and checks."""
    print("Andrew Quinn - EECS 332 - MP#3: Histogram and Color EQ")
    print("-" * 80)

    for subdir in subdirs:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    reveal_images()


if __name__ == "__main__":
    hello()
    print("\n\n\n\n")
    print("This is the helper file. Why are you here?... Go run mp3.py!")
    print("\n\n\n\n")
