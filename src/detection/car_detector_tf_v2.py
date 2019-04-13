from __future__ import absolute_import
"""
Car detector using tensorflow models
TODO: move into a class
"""
import os
import tarfile
import numpy as np
import cv2
import tensorflow as tf
from .yolov3 import yolov3


class CarDetectorTFV2(object):
    def __init__(self):
        tf.reset_default_graph()
        self.session = tf.Session()
        self.batch_size = 1
        self.max_output_size = 10
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.5
        self.model = yolov3.Yolo_v3(max_output_size=self.max_output_size,
                                    iou_threshold=self.iou_threshold,
                                    confidence_threshold=self.confidence_threshold)
        self.class_names = self.model.class_names
        self.model_size = self.model.model_size
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.model_size[0], self.model_size[1], 3])
        self.run_inference = self.model(self.inputs, training=False)
        self.model_vars = tf.global_variables(scope='yolo_v3_model')
        self.assign_ops = yolov3.load_weights(self.model_vars, self.model.weights_path)
        self.session.run(self.assign_ops)

    def detect(self, img, return_class_scores=False):
        """
        Given input image, return detections
        """
        detection_boxes, detection_classes, detection_scores = [], [], []
        img_net = cv2.resize(img, (self.model_size[0], self.model_size[1]))
        batch = np.array([img_net])
        detection_result = self.session.run(self.run_inference, feed_dict={self.inputs: batch})
        det = detection_result[0]
        resize_factor = (img.shape[1] / self.model_size[0], img.shape[0] / self.model_size[1])
        for cls in range(len(self.class_names)):
            boxes = det[cls]
            if np.size(boxes) != 0:
                for box in boxes:
                    xy, confidence = box[:4], box[4]
                    xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                    x1, y1, x2, y2 = xy[0], xy[1], xy[2], xy[3]
                    bbox = [x1, y1, x2, y2]
                    detection_boxes.append(bbox)
                    detection_classes.append(self.class_names[cls])
                    detection_scores.append(confidence)
        if return_class_scores:
            return detection_boxes, detection_classes, detection_scores
        else:
            return detection_boxes


if __name__ == '__main__':
    detector = CarDetectorTFV2()
    video_filename = './yolov3/videos/0002.avi'
    vc = cv2.VideoCapture()
    vc.open(video_filename)
    # skip to frames of interest
    SKIP_FRAMES = 400
    CONFIDENCE_THRESHOLD = 0.80
    print("skipping first {} frames...".format(SKIP_FRAMES))
    for _ in range(SKIP_FRAMES):
        vc.read()
    print('done.')
    while True:
        _, img = vc.read()
        if img is None:
            break
        detections = detector.detect(img=img, return_class_scores=True)
        bboxes, class_names, confidences = detections
        for bbox, class_name, confidence in zip(bboxes, class_names, confidences):
            if class_name in ['car', 'truck', 'bus'] and confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
                text = '{} {:.1f}%'.format(class_name,
                                           confidence * 100)
                cv2.putText(img, text, (x1, y1-5), 1, 1.5, color, 2)
        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key & 0xFF == 27 or key == ord('q'):
            break
    vc.release()
    cv2.destroyAllWindows()
