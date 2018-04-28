"""Open video and lidar readings to review data"""
from __future__ import print_function
import os
import cv2
import pandas as pd
import datetime
LIDAR_SPACING_PX = 38  # value found experimentally
START_X_LEFT_PX = 615 - LIDAR_SPACING_PX * 15  # value found experimentally


# def draw_lidar_spacing_lines(frame, lidar_spacing_px=37, start_x_left=90, m16_readings=None):
#     frame = frame.copy()
#     end_x_right = start_x_left + 16 * lidar_spacing_px
#     for seg, x_val in enumerate(range(start_x_left, end_x_right, lidar_spacing_px)):
#         cv2.line(frame, (x_val, 0), (x_val, 480), (0, 255, 0), 2)
#         #         print(x_val)
#         if m16_readings is not None:
#             # scale m16 reading btwn 0 and 480
#             y_val_m = m16_readings[seg].distance
#             min_y_m = 0  # 0 meters as min
#             max_y_m =15  # meters as max
#             y_val_normalized = (y_val_m - min_y_m) / (max_y_m - min_y_m)
#             y_val_px = frame.shape[0] - int(y_val_normalized * frame.shape[0])
#             center = (x_val, y_val_px)
#             cv2.circle(frame, center, 5, (255, 0, 0), -1)
#             cv2.putText(frame, '{:.2f}'.format(y_val_m), center, 1, 1, (0,0,255), 2)
#     return frame
M_TO_FT = 3.28084
def draw_lidar_spacing_lines(frame, frame_num, lidar_spacing_px=37, start_x_left=90, m16_detections=None):
    end_x_right = start_x_left + 16 * lidar_spacing_px
    for seg, x_val in enumerate(range(start_x_left, end_x_right, lidar_spacing_px)):
        cv2.line(frame, (x_val, 0), (x_val, 480), (0, 255, 0), 2)
        #         print(x_val)
        if m16_detections is not None:
            # scale m16 reading btwn 0 and 480
            y_val_m = m16_detections['z{}'.format(seg + 1)][frame_num]
            y_val_ft = y_val_m * M_TO_FT
            min_y_m = 0  # 0 meters as min
            max_y_m = 30  # meters as max
            y_val_normalized = (y_val_m - min_y_m) / (max_y_m - min_y_m)
            y_val_px = frame.shape[0] - int(y_val_normalized * frame.shape[0])
            center = (x_val, y_val_px)
            cv2.circle(frame, center, 5, (255, 0, 0), -1)
            cv2.putText(frame, '{:.2f}'.format(y_val_ft), center, 1, 1, (0, 0, 255), 2)
        #             print(x_val)
    return frame


PWD = os.path.dirname(__file__)
DATA_DIR = os.path.join(PWD, '/home/robert/PycharmProjects/Einstein/Data/{}'.format(datetime.datetime.today().strftime('%Y-%m-%d')))
RUN_NUMBER = '0006'
m16_detections = pd.read_csv('{}/{}.csv'.format(DATA_DIR, RUN_NUMBER))
video_feed = cv2.VideoCapture()
video_feed.open('{}/{}.avi'.format(DATA_DIR, RUN_NUMBER))
SKIP_SECONDS = 20
video_feed.set(cv2.CAP_PROP_POS_FRAMES, SKIP_SECONDS * 30)
print(m16_detections.head())

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
date = os.path.basename(DATA_DIR)
vw = cv2.VideoWriter(os.path.join('/home/robert/PycharmProjects/Einstein/Data/Processed', '{}_{}.avi'.format(date,RUN_NUMBER)), fourcc, 20.0, (640, 480))

while True:
    success, frame = video_feed.read()
    if not success:
        print('no frame')
        break
    frame_num = video_feed.get(cv2.CAP_PROP_POS_FRAMES)
    try:
        frame_draw = draw_lidar_spacing_lines(frame, frame_num, m16_detections=m16_detections)
    except:
        print('Leave last frame')
        break

    # write video
    vw.write(frame_draw)
    # draw vertical line in center of image
    cv2.line(frame_draw, (int(frame_draw.shape[1]/2), 0), (int(frame_draw.shape[1]/2), int(frame_draw.shape[0])), (255,0,255), 1)
    cv2.line(frame_draw, (0, int(frame_draw.shape[0]/2)), (int(frame_draw.shape[1]), int(frame_draw.shape[0]/2)), (255,0,255), 1)
    # show and display
    cv2.imshow('draw_frame', cv2.resize(frame_draw, (1280, 720)))
    cv2.waitKey(30)

vw.release()
