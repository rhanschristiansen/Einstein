"""
Uses kalman filter to predict subsequent detection bounding boxes
"""
import cv2

from ball_detector import BallDetector
from kalman import KalmanFilter

detector = BallDetector()
# detector = CarDetector()
kf = KalmanFilter()
video_filename = '/home/bob/datasets/video/TrackBalls/2017-05-12-112517.webm'

record_flag = False
if record_flag:
    # declare video writer obj
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter('/home/bob/datasets/video/TrackBalls/Tracker_balls/1.avi', fourcc, 20.0, (1920, 1080))

vc = cv2.VideoCapture()
vc.open(video_filename)
scale_factor = 0.75  # reduce frame by this size
tracked_bbox = []  # a list of past bboxes
max_num_missed_det = 5
num_missed_det = 0
while True:
    _, img = vc.read()
    if img is None:
        break
    img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    # detect ball in frame (just one for now)
    bboxes = detector.detect(img)
    bb = None
    if bboxes is not None and len(bboxes)>0:
        for bb in bboxes:
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            pass
        # assume only 1 object
        bb = bboxes[0]

    # update kalman measurement
    pred_bb = kf.get_predicted_bb()
    if bb is not None:
        # if we have a measurement, correct using measurement
        corr_bb = kf.correct(bb)
        num_missed_det = 0
    else:
        # if no measurement found, use prediction as measurement to correct
        if num_missed_det < max_num_missed_det:
            corr_bb = kf.correct(pred_bb)
            num_missed_det += 1 # count as missed detection
        else:
            corr_bb = None
    # add bbox to track
    if num_missed_det < max_num_missed_det:
        tracked_bbox.append(corr_bb)
    else:
        tracked_bbox = [] # if too many missed detections, kill track
    # draw track history
    if len(tracked_bbox) > 0:
        t_bb_prev = tracked_bbox[0]
        for t_bb in tracked_bbox:

            centroid = (int(t_bb[0] + (t_bb[2] - t_bb[0]) / 2), int(t_bb[1] + (t_bb[3] - t_bb[1]) / 2))
            centroid_prev = (int(t_bb_prev[0] + (t_bb_prev[2] - t_bb_prev[0]) / 2), int(t_bb_prev[1] + (t_bb_prev[3] - t_bb_prev[1]) / 2))
            cv2.line(img, centroid_prev, centroid, (0,255,255), 3)
            t_bb_prev = t_bb
    if corr_bb is not None:
        cv2.rectangle(img, (corr_bb[0], corr_bb[1]), (corr_bb[2], corr_bb[3]), (0, 255, 255), 4)
    cv2.putText(img, 'Measurement: {}'.format(bb), (0, 25), 1, 1.5, (0,255,0),2)
    cv2.putText(img, 'Prediction: {}'.format(pred_bb), (0, 75), 1, 1.5, (0,255,255),2)
    if record_flag:
        w_img = cv2.resize(img, (1920,1080))
        out1.write(w_img)
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # 'esc'  pressed
        break

out1.release()
#TODO: data association! Munkres algorithm etc.
