import numpy as np
import uuid

import cv2

import tracking_constants as const
from detection.detection import Detection
from kalman import KalmanFilter


class Track(object):
    def __init__(self):
        # each history entry is numpy array [frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, fvec_1...128]
        self.history = np.array([])
        self.num_misses = 0  # num of missed assignments
        self.max_misses = const.MAX_NUM_MISSES_TRACK
        self.has_match = False
        self.delete_me = False  # set for manual deletion
        self.kalman = KalmanFilter()
        color = tuple(np.random.random_integers(0, 255, size=3))
        self.drawing_color = color
        self.predicted_next_bb = None
        self.LENGTH_ESTABLISHED = 1  # number of sequential detections before we consider it a track
        self.uid = uuid.uuid4()

    def get_length(self):
        return self.history.shape[0]

    def is_singular(self):
        return True if self.history.shape[0] == self.LENGTH_ESTABLISHED else False

    def is_established(self):
        return True if self.history.shape[0] > self.LENGTH_ESTABLISHED else False

    def is_empty(self):
        return True if self.history.shape[0] == 0 else False

    def is_dead(self):
        return True if (self.num_misses >= self.max_misses or self.delete_me is True) else False

    def propagate_track(self, frame_id):
        # propagate track as if there was a detection, perhaps object is temporarily occluded or not detected
        # use predicted bb as measurement
        det = Detection.det_from_numpy_array(self.history[self.history.shape[0] - 1])
        det.bbox = self.get_predicted_next_bb()
        # TODO: adjust bbox to have same width height but only propegate x,y centroid position
        # pred_c_x, pred_c_y = util.centroid_from_bb(self.get_predicted_next_bb())
        # wid, ht = util.wid_ht_from_bb(det.bbox)
        # det.bbox = np.array([pred_c_x - wid / 2,
        #                      pred_c_y - ht / 2,
        #                      pred_c_x + wid / 2,
        #                      pred_c_y + ht / 2], dtype=np.int32)

        det.frame_id = frame_id
        self.add_to_track(det)

    def get_predicted_next_bb(self):
        # # get a prediction using the latest history as a measurement
        # measurement = np.array([self.get_latest_bb()], dtype=np.float32).T
        return self.kalman.get_predicted_bb()

    def add_to_track(self, det):
        # use detection measurement to predict and correct the kalman filter
        corrected_bb = self.kalman.correct(det.bbox)
        # use corrected bbox
        det.bbox = corrected_bb
        # increment detections number of matches (could be assigned to several tracks, need to keep track of this)
        det.num_matches += 1
        new_history = det.as_numpy_array()
        # print new_history
        if self.history.size > 0:
            self.history = np.vstack((self.history, new_history))
        else:
            self.history = new_history

    def get_latest_fvec(self):
        # get feature vector from the latest detection in the track (already computed during detection phase)
        return self.history[self.history.shape[0] - 1][5:]

    def get_latest_bb(self):
        return self.history[self.history.shape[0] - 1][1:5]

    def draw_history(self, img, draw_at_bottom=False):
        if self.history.shape[0] > 1:
            bb_latest = self.get_latest_bb()
            cv2.rectangle(img, (int(bb_latest[0]), int(bb_latest[1])), (int(bb_latest[2]), int(bb_latest[3])),
                          self.drawing_color, 2)
            # iterate through detection history
            prev_bb = self.history[0][1:5]
            for det in self.history:
                bb = det[1:5]
                # cv2.rectangle(img, tuple(bb[:2]),tuple(bb[2:]),(255,0,0),2)
                centroid = (int(bb[0] + (bb[2] - bb[0]) / 2), int(bb[1] + (bb[3] - bb[1]) / 2))
                bottom = (int(bb[0] + (bb[2] - bb[0]) / 2), int(bb[3]))
                bottom_prev = (int(prev_bb[0] + (prev_bb[2] - prev_bb[0]) / 2), int(prev_bb[3]))
                # cv2.circle(img, centroid, 5, self.drawing_color, 2)
                # cv2.circle(img, bottom, 3, self.drawing_color, 2)
                cv2.line(img, bottom_prev, bottom, self.drawing_color, 4)
                prev_bb = bb
