import numpy as np


class Detection(object):
    def __init__(self):
        self.img = None
        self.bbox = None
        self.fvec = np.array([1, 2, 3])  # init in case not needed
        self.frame_id = None
        self.num_misses = 0
        self.has_match = False
        self.num_matches = 0
        self.matches = []
        self.label = None
        self.confidence = None

    def as_numpy_array(self):
        return np.array([np.hstack((self.frame_id, self.bbox, self.fvec))])

    @staticmethod
    def det_from_numpy_array(np_array):
        d = Detection()
        d.frame_id = np_array[0]
        d.bbox = np_array[1:5]
        d.fvec = np_array[5:]

        return d
