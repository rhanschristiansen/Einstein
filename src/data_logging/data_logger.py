"""
A simple data logger
"""
class DataLogger(object):
    def __init__(self, filename, headers):
        self.filename = filename
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



