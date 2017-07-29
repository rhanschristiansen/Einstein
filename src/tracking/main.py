import os
import glob
import cv2
import numpy as np
import math
from video_capture import VideoCaptureWrapper
from pygments.formatters import img

from engine_constants import *
import util as util
from pedestrian_detector import PedestrianDetector
from multiple_object_tracker import MultipleObjectTracker


def make_crop_pretty(bb, img, crop_bb=None):
    method = 2
    # pad to constant aspect ratio (square for now)
    # add padding
    x1, y1, x2, y2 = bb
    centroid = util.centroid_from_bb(bb)
    h = int(bb[3] - bb[1])
    w = h
    if method == 1:
        pad_percent = 0.4
        img_size = (200, 200)

        x1 -= w * pad_percent
        y1 -= h * pad_percent
        x2 += w * pad_percent
        y2 += h * pad_percent
    elif method == 2:
        # use centroid and crop w.r.t. crop_bb
        crop_h = int(crop_bb[3] - crop_bb[1])
        crop_w = int(crop_bb[2] - crop_bb[0])
        x1 = int(centroid[0] - crop_w / 2)
        y1 = int(centroid[1] - crop_h / 2)
        x2 = int(centroid[0] + crop_w / 2)
        y2 = int(centroid[1] + crop_h / 2)

    padded_bb = [x1, y1, x2, y2]
    padded_bb = util.clamp_negative_nums(padded_bb)
    crop_img = img[padded_bb[1]:padded_bb[3], padded_bb[0]:padded_bb[2]]

    bb_sz = (int(crop_bb[3] - crop_bb[1]), int(crop_bb[2] - crop_bb[0]))
    # crop_img = cv2.resize(crop_img,img_size)
    # crop_img = cv2.resize(crop_img, bb_sz)

    # resize to constant size
    return crop_img


def get_largest_bb_from_history(history):
    heights = history[:, 4] - history[:, 2]
    widths = history[:, 3] - history[:, 1]
    height_max_ind = np.argmax(heights)
    largest_bb = history[height_max_ind, 1:5]
    return largest_bb


def get_smallest_bb_from_history(history):
    smallest_bb = None
    return smallest_bb


def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period


def smooth_out_history(history):
    period = 12
    for i in xrange(1, 5):
        history[:history.shape[0] - period + 1, i] = cumsum_sma(history[:, i], period=period)
    return history


def crop_from_histories():
    vc = VideoCaptureWrapper('/home/kyle/Downloads/TownCentreXVID.avi', start_position=0)
    # vc = VideoCaptureWrapper('/home/kyle/Videos/Cam 3 2-5IeIjazimVA.mp4', start_position=9800)
    fourcc = 0x00000021  # four character code for .mp4 (do not change)

    video_writer = cv2.VideoWriter('out.mp4', fourcc=fourcc, fps=30,
                                   frameSize=(200, 200))
    history_filenames = glob.iglob('tracks/*.npy')
    history_filenames_list = []
    # history_filenames = glob.iglob('tracks/0ab0d1d5-f75b-4b7d-9174-e785c24c5cb4.npy')

    histories = []
    for history_filename in history_filenames:
        hist = np.load(history_filename)
        histories.append(hist)
        history_filenames_list.append(history_filename)

    for i, hist in enumerate(histories):
        filename = history_filenames_list[i]
        filename, _ = os.path.splitext(os.path.basename(filename))
        filename = 'tracks_movies/' + filename + '.mp4'

        bboxes = hist[:, 1:5]
        if bboxes.shape[0] < 100:
            continue

        hist = smooth_out_history(hist)
        largest_bb_in_hist = get_largest_bb_from_history(hist)
        video_writer = cv2.VideoWriter(filename, fourcc=fourcc, fps=30,
                                       frameSize=(int(largest_bb_in_hist[2] - largest_bb_in_hist[0]),
                                                  int(largest_bb_in_hist[3] - largest_bb_in_hist[1])))
        frame_ids = hist[:, 0]
        for i, frame_id in enumerate(frame_ids):
            img = vc.read(frame_pos=frame_id)
            bb = bboxes[i]
            bb = util.bb_as_ints(bb)
            bb = util.clamp_negative_nums(bb)
            if not util.bb_has_width_height(bb):
                continue
            crop_img = make_crop_pretty(bb, img, crop_bb=largest_bb_in_hist)

            # crop_img = util.crop_img(bb, img)
            # # print bb
            video_writer.write(crop_img)
            if crop_img.shape[0] < 2:
                exit()
                # cv2.imshow('img', crop_img)
                # cv2.waitKey(30)
                # video_writer.release()


# crop_from_histories()
# exit()


def main():
    write_video = False
    write_tracks = False
    # vc = cv2.VideoCapture()
    # vc = VideoCaptureWrapper('/home/kyle/Videos/Cam 3 2-5IeIjazimVA.mp4', start_position=9800)
    vc = VideoCaptureWrapper('/home/kyle/Downloads/TownCentreXVID.avi', start_position=0)
    # fr = FaceRecGFace()
    if write_video is True:
        fourcc = 0x00000021  # four character code for .mp4 (do not change)
        video_writer = cv2.VideoWriter('out.mp4', fourcc=fourcc, fps=30,
                                       frameSize=(1280, 720))
    detector = PedestrianDetector()
    tracker = MultipleObjectTracker()
    rsz = 1
    while True:
        # img = vc.read()
        # img = vc.read()
        img = vc.read()
        if img is None:
            continue

        img = cv2.resize(img, (img.shape[1] / rsz, img.shape[0] / rsz))

        detections = detector.get_detections(img=img, frame_id=vc.frame_pos)
        PedestrianDetector.draw_detections(img,detections)
        tracker.update_tracks(detections=detections, frame_id=vc.frame_pos, save=write_tracks)
        tracker.draw_tracks(img)

        cv2.imshow('img', img)
        if write_video is True:
            video_writer.write(img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            exit(0)

if __name__ == '__main__':
    main()
