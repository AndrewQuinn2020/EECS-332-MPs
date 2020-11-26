#!/usr/bin/python3

# final_project.py

import logging
import os
import sys
from itertools import product
from functools import reduce
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

import colorlog
from PIL import Image
import numpy as np
from scipy import signal
import cv2 as cv

logger = logging.getLogger(__name__)
# Change this to get more, or fewer, error messages.
#   DEBUG = Show me everything.
#   INFO = Only the green text and up.
#   WARNING = Only warnings.
#   ERROR = Only (user coded) error messages.
#   CRITICAL = Only (user coded) critical error messages.
logger.setLevel(colorlog.colorlog.logging.DEBUG)

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter())
logger.addHandler(handler)

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=10000)
np.set_printoptions(precision=4)

# Some constants so that we know where all of our stuff is.
script_dir = os.path.dirname(os.path.abspath(__file__))
frames_dir = os.path.join(script_dir, "frames")
images_dir = os.path.join(script_dir, "images")
noisemaps_dir = os.path.join(images_dir, "noisemaps")
noisemaps_test_dir = os.path.join(noisemaps_dir, "test_adding_back_to_avg")
noisemaps_quilt_dir = os.path.join(noisemaps_dir, "noise_to_avg_quilted_over")
results_dir = os.path.join(script_dir, "results")

side_by_sides_dir = os.path.join(images_dir, "side_by_sides")
orig_vs_noise_dir = os.path.join(side_by_sides_dir, "original_vs_noisemaps")
orig_vs_noise_add_dir = os.path.join(
    side_by_sides_dir, "original_vs_noisemaps_with_avg"
)
final_comparison_dir = os.path.join(side_by_sides_dir, "final_comparison")

stitches_dir = os.path.join(images_dir, "stitches")
stitches_test_dir = os.path.join(stitches_dir, "identity_stitch")
stitches_noisemap_test_dir = os.path.join(stitches_dir, "identity_noise_stitch")
stitches_left_track_right_quilt_dir = os.path.join(
    stitches_dir, "left_track_right_quilt"
)
stiches_all_together_dir = os.path.join(stitches_dir, "all_together")

pen_tracking_dir = os.path.join(images_dir, "pen_tracked")

quilting_dir = os.path.join(script_dir, "Image-Quilting-for-Texture-Synthesis")

video_loc = os.path.join(script_dir, "Original.avi")


dirs = [
    script_dir,
    frames_dir,
    images_dir,
    results_dir,
    noisemaps_dir,
    side_by_sides_dir,
    orig_vs_noise_dir,
    stitches_dir,
    stitches_test_dir,
    noisemaps_test_dir,
    stitches_noisemap_test_dir,
    orig_vs_noise_add_dir,
    pen_tracking_dir,
    noisemaps_quilt_dir,
    stitches_left_track_right_quilt_dir,
    stiches_all_together_dir,
    final_comparison_dir,
]


def avi2stills(video_loc, dir=frames_dir, zero_pad=4):
    cap = cv.VideoCapture(video_loc)

    framecount = 0
    while cap.isOpened():
        ret, frame = cap.read()
        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        print(filename)

        try:
            cv.imwrite(os.path.join(dir, filename), frame)
        except:
            break
        framecount += 1

    cap.release()


def stills2avi(avi_name, stills_dir=frames_dir, results_dir=results_dir):
    images = list(sorted([img for img in os.listdir(stills_dir) if img[-4:] == ".png"]))
    print(images)

    frame = cv.imread(os.path.join(stills_dir, images[0]))
    height, width, layers = frame.shape

    video = cv.VideoWriter(
        os.path.join(results_dir, avi_name),
        cv.VideoWriter_fourcc(*"XVID"),
        30,
        (width, height),
    )

    # video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height)
    for image in images:
        video.write(cv.imread(os.path.join(stills_dir, image)))

    cv.destroyAllWindows()
    video.release()


def stills2avg(img_out_name, stills_dir=frames_dir, results_dir=images_dir):
    images = list(sorted([img for img in os.listdir(stills_dir) if img[-4:] == ".png"]))
    print(images)

    frame = cv.imread(os.path.join(stills_dir, images[0]))
    height, width, layers = frame.shape

    avg = np.zeros_like(frame).astype(np.float32)
    print(avg.shape)
    print(avg[0, 0, :])
    for image in images:
        avg += frame.astype(np.float32) / len(images)
        print(avg[0, 0, :])
    avg = avg.astype(int)
    print(avg[0, 0, :])
    cv.imwrite(os.path.join(results_dir, img_out_name), avg)
    return os.path.join(results_dir, img_out_name)


def stills2noisemaps(
    avg_loc, stills_dir=frames_dir, results_dir=noisemaps_dir, zero_pad=4
):
    images = list(sorted([img for img in os.listdir(stills_dir) if img[-4:] == ".png"]))
    print(images)

    avg = cv.imread(avg_loc)
    frame = cv.imread(os.path.join(stills_dir, images[0]))

    print(avg.shape)
    print(avg[0, 0, :])
    framecount = 0
    for image in images:
        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        frame = cv.imread(os.path.join(stills_dir, image))
        frame -= avg
        frame += 127
        print(frame[0, 0, :])
        cv.imwrite(os.path.join(results_dir, filename), frame)
        framecount += 1
    return None


def add_noisemaps_to_avg(
    avg_loc, stills_dir=noisemaps_dir, results_dir=noisemaps_test_dir, zero_pad=4
):
    images = list(sorted([img for img in os.listdir(stills_dir) if img[-4:] == ".png"]))
    print(images)

    avg = cv.imread(avg_loc)
    frame = cv.imread(os.path.join(stills_dir, images[0]))

    print(avg.shape)
    print(avg[0, 0, :])
    framecount = 0
    for image in images:
        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        frame = cv.imread(os.path.join(stills_dir, image))
        print(frame[0, 0, :])
        frame += avg
        frame -= 127
        print(frame[0, 0, :])
        cv.imwrite(os.path.join(results_dir, filename), frame)
        framecount += 1
    return None


def stills2sidebysides(dir_1, dir_2, out_dir, zero_pad=4):
    images_left = list(sorted([img for img in os.listdir(dir_1) if img[-4:] == ".png"]))
    images_right = list(
        sorted([img for img in os.listdir(dir_2) if img[-4:] == ".png"])
    )

    framecount = 0
    for (il, ir) in zip(images_left, images_right):
        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        frame_left = cv.imread(os.path.join(dir_1, il))
        frame_right = cv.imread(os.path.join(dir_2, ir))
        frame = np.hstack((frame_left, frame_right))
        cv.imwrite(os.path.join(out_dir, filename), frame)
        framecount += 1
    return


def stillstitch(top_still, bottom_still, cutoff=60):
    return np.vstack((top_still[:cutoff, :, :], bottom_still[cutoff:, :, :]))


def stillvstitch(left_still, right_still, cutoff=60):
    return np.hstack((left_still[:, :cutoff, :], right_still[:, cutoff:, :]))


def stills2stitched(dir_1, dir_2, out_dir, zero_pad=4, cutoff=60):
    images_top = list(sorted([img for img in os.listdir(dir_1) if img[-4:] == ".png"]))
    images_bottom = list(
        sorted([img for img in os.listdir(dir_2) if img[-4:] == ".png"])
    )

    framecount = 0
    for (it, ib) in zip(images_top, images_bottom):
        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        frame_top = cv.imread(os.path.join(dir_1, it))
        frame_bottom = cv.imread(os.path.join(dir_2, ib))
        frame = stillstitch(frame_top, frame_bottom, cutoff=cutoff)
        cv.imwrite(os.path.join(out_dir, filename), frame)
        framecount += 1
    return


def stills2vstitched(
    dir_1,
    dir_2,
    out_dir,
    cutoffs,
    zero_pad=4,
):
    images_left = list(sorted([img for img in os.listdir(dir_1) if img[-4:] == ".png"]))
    images_right = list(
        sorted([img for img in os.listdir(dir_2) if img[-4:] == ".png"])
    )

    framecount = 0
    for (it, ib) in zip(images_left, images_right):
        try:
            filename = "{}.png".format(str(framecount).zfill(zero_pad))
            frame_left = cv.imread(os.path.join(dir_1, it))
            frame_right = cv.imread(os.path.join(dir_2, ib))
            frame = stillvstitch(frame_left, frame_right, cutoff=cutoffs[framecount])
            cv.imwrite(os.path.join(out_dir, filename), frame)
            framecount += 1
        except:
            break
    return


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def avi2tracked(avi_in, start, end, results_dir=pen_tracking_dir, zero_pad=4):
    centers = []
    start_box = (start[0], start[1], end[0] - start[0], end[1] - start[1])
    tracker = cv.TrackerKCF_create()

    vs = cv.VideoCapture(avi_in)

    framecount = 0
    first_frame = True
    while True:
        frame = vs.read()[1]
        if frame is None:
            break

        filename = "{}.png".format(str(framecount).zfill(zero_pad))
        if first_frame:  # show the output frame
            print(start_box)
            tracker.init(frame, start_box)
            first_frame = False
            print(start_box)
        else:
            (success, box) = tracker.update(frame)
            start = (int(box[0]), int(box[1]))
            end = (int(box[0] + box[2]), int(box[1] + box[3]))
            print(start, end)
        frame = cv.rectangle(frame, start, end, (255, 0, 0), 2)
        cv.imwrite(os.path.join(results_dir, filename), frame)

        center = (start[0] + end[0]) // 2
        centers.append(center)
        framecount += 1

    return centers


if __name__ == "__main__":
    logger.info("Andrew Quinn - EECS 332 - Final Project")
    logger.info("-" * (88 - 11))

    for dir in dirs:
        if not os.path.exists(dir):
            logger.warning("\t\t{}\t\t doesn't exist... Creating.".format(dir))
            os.makedirs(dir)

    logger.info("Writing {} as sequence of frames to {}".format(video_loc, frames_dir))

    avi2stills(video_loc)
    stills2avi("identity.avi")

    path_to_avg = stills2avg("avg.png")

    stills2noisemaps(os.path.join(images_dir, "avg.png"))
    stills2avi("noisemaps.avi", stills_dir=noisemaps_dir)

    stills2sidebysides(frames_dir, noisemaps_dir, orig_vs_noise_dir)
    stills2avi("orig_vs_noisemaps.avi", stills_dir=orig_vs_noise_dir)
    stills2stitched(frames_dir, frames_dir, stitches_test_dir)

    add_noisemaps_to_avg(path_to_avg)
    stills2stitched(frames_dir, noisemaps_test_dir, stitches_noisemap_test_dir)
    stills2sidebysides(frames_dir, stitches_noisemap_test_dir, orig_vs_noise_add_dir)
    stills2avi("orig_vs_noisemaps_plus_avg.avi", stills_dir=orig_vs_noise_add_dir)

    # Generating image quilt.
    cv.imwrite(
        os.path.join(images_dir, "avg_quilt.png"),
        cv.imread(path_to_avg)[140:180, 60:300, :],
    )

    os.system(
        "cd {}; python main.py -i ../images/avg_quilt.png -f ../images/avg_quilt_big.png".format(
            quilting_dir
        )
    )

    avg_quilt_big = cv.imread(os.path.join(images_dir, "avg_quilt_big.png"))
    quilt_cover = avg_quilt_big[-80:, -250:, :]
    cv.imwrite(os.path.join(images_dir, "quilt_cover.png"), quilt_cover)

    avg_quilted_over = cv.imread(path_to_avg)
    avg_quilted_over[60:140, 54:304, :] = quilt_cover

    cv.imwrite(os.path.join(images_dir, "avg_quilted_over.png"), avg_quilted_over)

    centers = avi2tracked(video_loc, (262, 20), (262 + 45, 20 + 45))
    centers = moving_average(centers).astype(int)
    print(centers)
    stills2avi("pen_tracked.avi", stills_dir=pen_tracking_dir)

    add_noisemaps_to_avg(
        os.path.join(images_dir, "avg_quilted_over.png"),
        stills_dir=noisemaps_dir,
        results_dir=noisemaps_quilt_dir,
    )
    stills2avi("quilted_noisemaps.avi", stills_dir=noisemaps_quilt_dir)
    stills2vstitched(
        frames_dir,
        noisemaps_quilt_dir,
        stitches_left_track_right_quilt_dir,
        centers,
    )
    stills2avi(
        "frame_left_quilted_right.avi", stills_dir=stitches_left_track_right_quilt_dir
    )
    stills2stitched(
        pen_tracking_dir,
        stitches_left_track_right_quilt_dir,
        stiches_all_together_dir,
        cutoff=80,
    )
    stills2avi("all_together.avi", stills_dir=stiches_all_together_dir)

    stills2sidebysides(
        frames_dir, stitches_left_track_right_quilt_dir, final_comparison_dir
    )
    stills2avi("final_comparison.avi", stills_dir=final_comparison_dir)
