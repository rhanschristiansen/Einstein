"""
Uses kalman filter to predict subsequent detection bounding boxes
"""
import datetime
import numpy as np

import cv2

from data_logging.data_logger import DataLogger
from detection.car_detector import CarDetector
from detection.detection import Detection
from kalman import KalmanFilter
from multiple_object_tracker import MultipleObjectTracker

# log the data to a csv file
filename = 'bb_logs.csv'
headers = ['time', 'id', 'x1', 'y1', 'x2', 'y2']
logger = DataLogger(filename=filename, headers=headers)

tracker = MultipleObjectTracker()
# detector = BallDetector()
detector = CarDetector()
kf = KalmanFilter()
# video_filename = '/home/bob/datasets/video/TrackBalls/2017-05-12-112517.webm'
video_filename = '/home/bob/datasets/video/MOVI0003.avi'
save_video_filename = '/home/bob/datasets/video/MOVI0003_tracking.avi'

record_flag = False



vc = cv2.VideoCapture()
vc.open(video_filename)

scale_factor = 0.75  # reduce frame by this size
tracked_bbox = []  # a list of past bboxes
max_num_missed_det = 5
num_missed_det = 0
current_frame_position = 0
if record_flag:
    # get first frame of video to get recording size
    _, frame = vc.read()
    # declare video writer obj
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(save_video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
while True:
    _, img = vc.read()
    if img is None:
        break
    current_frame_position += 1
    img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
    # detect objects in frame
    bboxes = detector.detect(img)
    detections = []  # our list of Detections objects
    bb = None
    current_time = datetime.datetime.now()
    if bboxes is not None and len(bboxes) > 0:
        for i, bb in enumerate(bboxes):
            det = Detection()
            det.bbox = np.array([bb[0], bb[1], bb[2], bb[3]])
            det.frame_id = current_frame_position
            detections.append(det)
            # DRAW DETECTIONS IN GREEN (B, G, R) = (0, 255, 0)
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            # log the data, recall headers = ['time', 'id', 'x1', 'y1', 'x2', 'y2']
            bb_id = i
            data = [current_time, bb_id, bb[0], bb[1], bb[2], bb[3]]
            logger.log(data=data)


    tracker.update_tracks(detections=detections, frame_id=current_frame_position)
    tracker.draw_tracks(img)
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    # if key == 27:
    #     break
    if record_flag is True:
        # rescale image back to original shape
        img = cv2.resize(img, (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor)))
        vw.write(img)
