import os
import time
import pandas as pd
import cv2
from lidar.leddar_m16 import LeddarM16
from data_logging.data_logger import DataLogger

RECORD = True


# draw the detection distances on the video feed

def draw_lidar_spacing_lines(frame, lidar_spacing_px=37, start_x_left=90, m16_readings=None):
    frame = frame.copy()
    end_x_right = start_x_left + 16 * lidar_spacing_px
    for seg, x_val in enumerate(range(start_x_left, end_x_right, lidar_spacing_px)):
        cv2.line(frame, (x_val, 0), (x_val, 480), (0, 255, 0), 2)
        #         print(x_val)
        if m16_readings is not None:
            # scale m16 reading btwn 0 and 480
            y_val_m = m16_readings[seg].distance
            min_y_m = 0  # 0 meters as min
            max_y_m = 4  # 4 meters as max
            y_val_normalized = (y_val_m - min_y_m) / (max_y_m - min_y_m)
            y_val_px = frame.shape[0] - int(y_val_normalized * frame.shape[0])
            center = (x_val, y_val_px)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
    return frame


LIDAR_SPACING_PX = 38  # value found experimentally
START_X_LEFT_PX = 615 - LIDAR_SPACING_PX * 15  # value found experimentally


def get_next_trial_num(log_dir):
    max_num = 1
    for file in os.listdir(log_dir):
        if file.endswith(".avi"):
            num = int(os.path.splitext(os.path.basename(file))[0])
            if num > max_num:
                max_num = num

    return max_num + 1


this_dir = os.path.dirname(__file__)
log_dir = os.path.join(this_dir, '../Data/Mar17_2018/')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

trial_num = get_next_trial_num(log_dir)
trial_num_str = '%04d' % (trial_num)

video_fpath = os.path.join(log_dir, trial_num_str + '.avi')
log_fpath = os.path.join(log_dir, trial_num_str + '.csv')

m16_sensor = LeddarM16()
headers = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16']
data_logger = DataLogger(filename=log_fpath, headers=headers)

cap = cv2.VideoCapture()
if not cap.open(0):
    raise Exception("Error opening camera")
fps = cap.get(cv2.CAP_PROP_FPS)
_, frame = cap.read()
# declare video writer obj
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vw = cv2.VideoWriter(video_fpath, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

detections = m16_sensor.get_detections()

while True:
    success, frame = cap.read()
    if not success:
        raise Exception("Error reading video frame")

    detections = m16_sensor.get_detections()
    pretty_string = ''
    for det in detections:
        pretty_string += '{1:0.1f} '.format(det.segment, det.distance)

    log_data = [x.distance for x in detections]
    data_logger.log([log_data])
    vw.write(frame)
    # cv2.putText(frame, pretty_string, (0, frame.shape[0] / 2), 1, 1, (0, 255, 0))
    # line1_y = int(frame.shape[0] / 2)
    # line2_y = int(frame.shape[0] / 1.7)
    # cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (0, 255, 0), 2)
    # cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (0, 255, 0), 2)

    # Draw lidar distances
    frame_draw = draw_lidar_spacing_lines(frame=frame,
                                          start_x_left=START_X_LEFT_PX,
                                          lidar_spacing_px=LIDAR_SPACING_PX,
                                          m16_readings=detections)

    cv2.imshow('frame', cv2.resize(frame_draw, (1280, 960)))
    ch = cv2.waitKey(10)
    if ch & 0xFF == ord('q'):
        break
