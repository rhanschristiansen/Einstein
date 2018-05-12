"""
Car detector using tensorflow models
TODO: move into a class
"""
import os
import urllib
import urllib2
import tarfile
import numpy as np
import tensorflow as tf
import cv2


class CarDetectorTF(object):
    def __init__(self):
        self.detection_graph = self.load_model()

    def load_model(self):
        model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
        model_dir = os.getcwd()
        model_path = os.path.join(model_dir, model_name)
        checkpoint_path = os.path.join(model_path, 'frozen_inference_graph.pb')
        if not os.path.exists(checkpoint_path):
            model_download_url = 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(model_name)
            self.download_extract_model(model_download_url=model_download_url)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def download_extract_model(self, model_download_url):
        fname = os.path.basename(model_download_url)
        print('downloading {}...'.format(fname))
        f = urllib2.urlopen(model_download_url)
        data = f.read()
        with open(fname, "wb") as code:
            code.write(data)
        print('extracting {}'.format(fname))
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()

    def init_session(self):
        """WORK IN PROGRESS"""
        self.context_manager = self.detection_graph.as_default()
        self.session = tf.Session()

    def run_inference_for_video(self, video_file):
        vc = cv2.VideoCapture()
        if not vc.open(video_file):
            raise Exception('error opening {}'.format(video_file))
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                while True:
                    _, img = vc.read()
                    detections = sess.run(tensor_dict,
                                          feed_dict={image_tensor: np.expand_dims(img, 0)})
                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    detections['num_detections'] = int(detections['num_detections'][0])
                    detections['detection_classes'] = detections[
                        'detection_classes'][0].astype(np.uint8)
                    detections['detection_boxes'] = detections['detection_boxes'][0]
                    detections['detection_scores'] = detections['detection_scores'][0]
                    if 'detection_masks' in detections:
                        detections['detection_masks'] = detections['detection_masks'][0]

                    num_detections = detections['num_detections']
                    detection_boxes = detections['detection_boxes'][:num_detections]
                    detection_scores = detections['detection_scores'][:num_detections]
                    detection_classes = detections['detection_classes'][:num_detections]
                    img_width = img.shape[1]
                    img_height = img.shape[0]
                    img_draw = img.copy()
                    for det, cls, score in zip(detection_boxes, detection_classes, detection_scores):
                        y1, x1, y2, x2 = det
                        x1 *= img_width
                        x2 *= img_width
                        y1 *= img_height
                        y2 *= img_height
                        cv2.putText(img_draw, '{}'.format(cls), (int(x1), int(y1)), 2, 1, (0, 255, 0))
                        cv2.putText(img_draw, '{}'.format(score), (int(x1) + 10, int(y1)), 1, 1, (0, 255, 255))
                        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    cv2.imshow('img', img_draw)
                    ch = cv2.waitKey(10)
                    if ch & 0xFF == ord('q') or ch & 0xFF == 27:
                        break



if __name__ == '__main__':
    video_file = os.path.expanduser('~/PycharmProjects/Einstein/Data/2018-05-05/0015.avi')
    detector = CarDetectorTF()
    detector.run_inference_for_video(video_file=video_file)
