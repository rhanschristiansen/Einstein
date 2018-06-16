"""
A simple data logger
"""
import os
class DataLogger(object):
    def __init__(self, filename, headers, overwrite=True):
        self.filename = filename
        if overwrite is True:
            if os.path.exists(self.filename):
                os.remove(self.filename)
        self.headers = headers
        with open(self.filename, 'w') as f:
            self.__create_header(f)

    def __create_header(self, f):
        for header in self.headers:
            f.write(str(header) + ',')
        f.write('\n')

    def log(self, data):

        with open(self.filename, 'a') as f:
            for row in data:
                if len(row) != len(self.headers):
                    raise Exception(
                        'ERROR, len(row) != len(self.headers). {} vs. {}'.format(len(data), len(self.headers)))
                for item in row:
                    f.write(str(item) + ',')
                f.write('\n')

    def log_bboxes(self, frame_num, bboxes):
        """
        Log the bbox detections for each frame.
        There will be two columns: frame_num, bboxes.
        
        Notes:
        bboxes will be in the following format:
        each element of bb is space delimited, and each bb is underscore _ delimited
        eg. '391 251 453 293_380 254 405 273_434 255 539 327'
        
        To parse this to list from csv, use:
        
            def parse_bbox(bb_str):
                all_bboxes = []
                for bb_s in bb_str.split('_'):
                    vals = bb_s.split(' ')
                    bbox_list = []
                    for v in vals:
                        bbox_list.append(int(v))
                    all_bboxes.append(bbox_list)
                return all_bboxes

        """

        def _list_2d_to_str(list_2d):
            """
            Convert from [[1,2,3],[4,5,6]] to '1 2 3_4 5 6'
            """
            output_str = ''
            for idx2d, l in enumerate(list_2d):
                for idx, val in enumerate(l):
                    output_str += str(val)
                    if idx < len(l) - 1:
                        output_str += ' '

                if idx2d < len(list_2d) - 1:
                    output_str += '_'
            return output_str
        with open(self.filename, 'a') as f:
            bbox_str = _list_2d_to_str(bboxes)
            f.write(str(frame_num) + ',' + bbox_str + '\n')


"""
a sample implementation of the data logger
if in a new file, we will just need to import the class:
    from data_logging.data_logger import DataLogger
"""
if __name__ == '__main__':
    import datetime
    import time
    filename = 'testing123.csv'
    headers = ['time', 'x1', 'y1', 'x2', 'y2', 'lidar_dist1']
    logger = DataLogger(filename=filename, headers=headers)
    bb = [0, 0, 0, 0]
    lidar_dist1 = 0
    # simulate N program loops with fake data to test the data logger
    N = 10
    for i in xrange(N):
        time.sleep(1)
        t = datetime.datetime.now()
        for i in xrange(len(bb)):
            bb[i] += 1
        lidar_dist1 += 2
        # Note: data needs to be of same length as headers list (so each data corresponds to each header)
        data = [t, bb[0], bb[1], bb[2], bb[3], lidar_dist1]
        logger.log(data)



