"""
color based ball detector
"""

import cv2
import numpy as np


class Ball(object):
    def __init__(self):
        self.bbox = [0, 0, 0, 0]  # [x1,y1,x2,y2]
        self.hsv_filter = [0, 0, 0, 255, 255, 255]  # h_min,s_min,v_min,h_max,s_max,v_max


class HSVfilter(object):
    def __init__(self):
        pass

    @staticmethod
    def apply(img, hsv_filter):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_binary = cv2.inRange(img_hsv, tuple(hsv_filter[:3]), tuple(hsv_filter[3:]))
        return img_binary


class BallDetector(object):
    def __init__(self):
        self.ball = Ball()
        self.ball.hsv_filter = [0, 158, 239, 10, 255, 255]

    def detect(self, img):
        img_binary = HSVfilter.apply(img, self.ball.hsv_filter)
        cv2.imshow('binary',img_binary)
        im2, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                bbox_xywh = cv2.boundingRect(cnt)
                bbox = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
                bboxes.append(bbox)

        return bboxes


# for video '/home/robert/datasets/video/TrackBalls/2017-05-12-112517.webm'
# we found red ball to have filter: [2, 175, 255, 16, 255, 255]

def try_me():
    video_filename = '/home/bob/datasets/video/TrackBalls/2017-05-12-112517.webm'
    vc = cv2.VideoCapture()
    vc.open(video_filename)
    det = BallDetector()
    while True:
        _, img = vc.read()
        if img is None:
            break
        img = cv2.resize(img, (1280, 720))

        #################################################### demo what hsv frame looks like for Bob
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow('img_hsv', img_hsv)
        ################################################### comment me out to get rid of me (ctrl + '/')
        bboxes = det.detect(img)
        for bb in bboxes:
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    try_me()