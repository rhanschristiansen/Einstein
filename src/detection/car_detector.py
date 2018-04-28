import os
import glob

import caffe
import cv2
import numpy as np
from bobs_utils import img_utils


class CarDetector(object):
    # constructor
    def __init__(self):
        # load the caffe network
        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.proto_path = './models/20170304-111715-5c04_epoch_100.0/deploy.prototxt'
        self.model_path = './models/20170304-111715-5c04_epoch_100.0/snapshot_iter_26600.caffemodel'
        self.net = caffe.Net(self.proto_path, self.model_path, caffe.TEST)
        # load some of the net data shape info
        self.net_batch, self.net_ch, self.net_height, self.net_width = self.net.blobs['data'].data[...].shape

    def detect(self, img):
        # TODO: get car_bboxes
        # resize image to fit into the network data layer
        im_resized, scaleFactor = img_utils.resize_pad_image(img, (self.net_width, self.net_height))
        im2 = im_resized.transpose((2, 0, 1))
        self.net.blobs['data'].data[...] = im2
        # run a forward pass on neural network
        self.net.forward()
        # get the results from the bbox-list layer
        net_output = [o for o in self.net.blobs['bbox-list'].data[0] if o.any()]

        # TODO: resize the bounding boxes to reflect the size of the original input image
        confidence = []
        bboxes_car = []
        for i in xrange(len(net_output)):
            # get bbox from first 4 elements of net_output at index i
            bbox = net_output[i][:4]
            confidence.append(net_output[i][4])
            # resize the bounding box
            bbox = np.array([bbox[0] * scaleFactor[0], bbox[1] * scaleFactor[1], bbox[2] * scaleFactor[0],
                             bbox[3] * scaleFactor[1]], dtype=int)
            # append bounding box to bboxes_car for output to the user
            bboxes_car.append(bbox)

        return bboxes_car


def main():
    # instantiate our car detector
    car_detector = CarDetector()
    video_file = '/home/robert/datasets/video/MOVI0003.avi'  # path to input video file
    # init the video capture
    vc = cv2.VideoCapture(video_file)
    while True:
        # load image from capture device
        grab, img = vc.read()
        if not grab:
            break
        # get bounding box output from our car detector class
        car_bboxes = car_detector.detect_cars(img)
        #iterate through bounding boxes to display on image
        for bbox in car_bboxes:
            img_utils.draw_bbox(img, bbox)
        # show image
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
