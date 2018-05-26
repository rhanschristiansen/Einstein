from __future__ import print_function
import os
import sys
import datetime
import cv2
import numpy as np
from detection.detection import Detection
from tracking.multiple_object_tracker import MultipleObjectTracker
from detection.car_detector import CarDetector
from detection.car_detector_tf import CarDetectorTF
from data_logging.data_logger import DataLogger
from distance_prediction.distance_predictor import DistancePredictor
from lidar.lidar_sensor import LidarSensor

# caffe_root = '/home/robert/caffe-0.15.9/'
# sys.path.insert(0, caffe_root + 'python')


# hold all our settings
class Settings(object):
    def __init__(self):
        self.SIMULATION_MODE = True
        self.SIMULATION_FILE = os.path.expanduser('~/PycharmProjects/Einstein/src/simulation_data/output43.csv')
        self.INPUT_VIDEO_FILE = os.path.expanduser('~/PycharmProjects/Einstein/src/simulation_data/output43.avi')
        self.WEBCAM_ID = 0
        self.RECORD_VIDEO = True
        self.OUTPUT_VIDEO_FOLDER = './output_video'
        self.OUTPUT_VIDEO_FILENAME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.ENABLE_LOGGING = True
        self.LOGGING_DIRECTORY = './logs'


def run():
    # create access to settings object
    settings = Settings()
    # initiate logging
    if settings.ENABLE_LOGGING is True:
        if not os.path.exists(settings.LOGGING_DIRECTORY):
            os.mkdir(settings.LOGGING_DIRECTORY)
        log_filename = os.path.join(settings.LOGGING_DIRECTORY,
                                    settings.OUTPUT_VIDEO_FILENAME + '.csv')
        # [current_frame_position, current_time_string, uid, latest_bbox, lidar_distance]
        headers = ['frame#', 'time', 'uid', 'x1', 'y1', 'x2', 'y2', 'lidar_d(ft)', 'grnd_angle_d(ft)']
        data_logger = DataLogger(filename=log_filename, headers=headers)

    vc = cv2.VideoCapture()
    if settings.SIMULATION_MODE is True:
        vc.open(settings.INPUT_VIDEO_FILE)
    else:
        vc.open(settings.WEBCAM_ID)

    if settings.RECORD_VIDEO is True:
        if not os.path.exists(settings.OUTPUT_VIDEO_FOLDER):
            os.mkdir(settings.OUTPUT_VIDEO_FOLDER)
        # get first frame of video to get recording size
        _, frame = vc.read()
        # declare video writer obj
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vw = cv2.VideoWriter(os.path.join(settings.OUTPUT_VIDEO_FOLDER, settings.OUTPUT_VIDEO_FILENAME + '.avi'),
                             fourcc,
                             20.0, (frame.shape[1], frame.shape[0]))

    # detector = CarDetector()
    detector = CarDetectorTF()
    tracker = MultipleObjectTracker()
    if settings.SIMULATION_MODE is True:
        lidar = LidarSensor(mode='simulation', simulation_data_file=settings.SIMULATION_FILE)
    else:
        lidar = LidarSensor()
    distance_predictor = DistancePredictor()

    scale_factor = 0.75  # reduce frame by this size
    tracked_bbox = []  # a list of past bboxes
    current_frame_position = 0

    while True:
        # take a snapshot of the current time
        current_time = datetime.datetime.now()
        current_time_string = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # first get lidar distance
        lidar_distance = lidar.get_distance(frame_number=current_frame_position)
        # now acquire image and get detections
        _, img = vc.read()
        if img is None:
            print('null image retrieved! frame={}'.format(current_frame_position))
            break
        current_frame_position += 1
        img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
        # detect objects in frame
        bboxes = detector.detect(img)
        detections = []  # our list of Detections objects
        bb = None

        # fill in the list of detections
        if bboxes is not None and len(bboxes) > 0:
            for i, bb in enumerate(bboxes):
                det = Detection()
                det.bbox = np.array([bb[0], bb[1], bb[2], bb[3]])
                det.frame_id = current_frame_position
                detections.append(det)
                # DRAW DETECTIONS IN GREEN (B, G, R) = (0, 255, 0)
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                # log the data, recall headers = ['time', 'id', 'x1', 'y1', 'x2', 'y2']
                # bb_id = i
                # data = [current_time, bb_id, bb[0], bb[1], bb[2], bb[3]]
        tracker.update_tracks(detections=detections, frame_id=current_frame_position)
        # create a list of the current row for logging
        logging_data = []
        # get the "head" of each track
        current_tracks = tracker.get_track_heads()
        for current_track in current_tracks:
            latest_bbox = current_track.get_latest_bb()
            # get predicted distances from bounding box
            predicted_distances = distance_predictor.predict(latest_bbox)
            uid = current_track.uid
            logging_row = [current_frame_position, current_time_string, uid, latest_bbox[0], latest_bbox[1],
                           latest_bbox[2],
                           latest_bbox[3], lidar_distance, predicted_distances[0]]
            logging_data.append(logging_row)
        if settings.ENABLE_LOGGING is True:
            data_logger.log(logging_data)
        tracker.draw_tracks(img)
        # draw lidar distance
        cv2.putText(img, 'lidar_dist = ' + str(int(lidar_distance)) + 'ft', (0, 25), 1, 1.5, (255, 0, 0), 2)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if settings.RECORD_VIDEO is True:
            # rescale image back to original shape
            img = cv2.resize(img, (int(img.shape[1] / scale_factor), int(img.shape[0] / scale_factor)))
            vw.write(img)


if __name__ == '__main__':
    run()
