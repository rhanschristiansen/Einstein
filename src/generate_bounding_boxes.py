from __future__ import print_function
"""
Use a object detector to generate bounding box data for a given video sequence
The output will be:
1) a .csv file containing bounding boxes from object detector [frame_num,x1,y1,x2,y2]
2) a video file with bounding boxes drawn to each frame 
"""
import cv2
from tqdm import tqdm
from data_logging.data_logger import DataLogger
from detection.car_detector_tf import CarDetectorTF
data_date = '2018-05-05'
run_number = '0015'
src_video_file = '/home/robert/PycharmProjects/Einstein/Data/{}/{}.avi'.format(data_date, run_number)
src_data_file = '/home/robert/PycharmProjects/Einstein/Data/{}/{}.csv'.format(data_date, run_number)

dst_video_file = '/home/robert/PycharmProjects/Einstein/Data/Processed/{}/{}.avi'.format(data_date, run_number)
dst_data_file = '/home/robert/PycharmProjects/Einstein/Data/Processed/{}/{}.csv'.format(data_date, run_number)

headers = ['frame_num', 'bboxes']
data_logger = DataLogger(filename=dst_data_file, headers=headers)
vc = cv2.VideoCapture()
vc.open(src_video_file)
num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
detector = CarDetectorTF()

for frame_num in tqdm(range(num_frames)):
    _, img = vc.read()
    if img is None:
        break
    bboxes = detector.detect(img=img)
    data_logger.log_bboxes(frame_num=frame_num, bboxes=bboxes)
print('done')
# for each frame
# detect bounding boxes
# write bounding boxes to csv
# write frame to output video
# done
