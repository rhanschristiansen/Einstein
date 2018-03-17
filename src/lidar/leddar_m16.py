import os
import ctypes


class LdDetection(ctypes.Structure):
    """C type struct to interface with LeddarC"""
    _fields_ = [('mDistance', ctypes.c_float),
                ('mAmplitude', ctypes.c_float),
                ('mSegment', ctypes.c_ushort),
                ('mFlags', ctypes.c_ushort)]

    @classmethod
    def array(cls, num_elements):
        """Returns array of struct. In C it would look like: `LdDetection my_array[num_elements]`"""
        elems = (cls * num_elements)()
        struct_array = ctypes.cast(elems, ctypes.POINTER(cls))
        return struct_array


class Detection(object):
    """pythonic interface to LdDetection"""

    def __init__(self):
        self.distance = None
        self.amplitude = None
        self.segment = None
        self.flags = None


class LeddarM16(object):
    def __init__(self):
        # load C dll
        try:
            self.clib = ctypes.cdll.LoadLibrary('/usr/local/lib/libLeddarC.so')
        except Exception as e:
            raise Exception('Unable to load Leddar libs. Be sure they are installed at /usr/local/lib'
                            'and /usr/local/lib is added to LD_LIBRARY_PATH environment variable')

        self.chandle = self.clib.LeddarCreate()
        # init struct array of 16 detections
        self.num_detections = 16
        self.c_detections = LdDetection.array(num_elements=self.num_detections)
        self.connect()

    def connect(self):
        print "Connecting to M16 sensor..."
        # os.system('sudo chmod a+rw,o+rw /dev/bus/usb -R')
        connection = self.clib.LeddarConnect(self.chandle,
                                             ctypes.c_char_p("USB"),
                                             ctypes.c_char_p("AJ21005"))
        if connection != 0:
            raise Exception("Unable to connect to M16 USB. Be sure it is plugged in. \n"
                            "Also, be sure we have read/write permissions to the USB device. "
                            "Try: \"sudo chmod a+rw,o+rw /dev/bus/usb -R\"")
        connected = self.clib.LeddarGetConnected(self.chandle)
        detection_count = self.clib.LeddarGetDetectionCount(self.chandle)
        if connection == 0 and connected == 0 and detection_count == self.num_detections:
            print "Successfully connected to M16 sensor."
        print "Starting data transfer..."
        lddl_detections = 2
        transfer_success = self.clib.LeddarStartDataTransfer(self.chandle, lddl_detections)
        if transfer_success == 0:
            print "successfully started data transfer."
        else:
            raise Exception("ERROR STARTING DATA TRANSFER")

    def get_detections(self):
        if self.clib.LeddarIsNewDataAvailable(self.chandle):
            self.clib.LeddarGetDetections(self.chandle, self.c_detections, self.num_detections)
        detections = []
        for i in range(self.num_detections):
            detection = Detection()
            detection.distance = self.c_detections[i].mDistance
            detection.amplitude = self.c_detections[i].mAmplitude
            detection.segment = self.c_detections[i].mSegment
            detection.flags = self.c_detections[i].mFlags
            detections.append(detection)
        return detections


def try_sensor():
    import time
    m16_sensor = LeddarM16()
    while True:
        detections = m16_sensor.get_detections()
        pretty_string = ''
        for det in detections:
            pretty_string += '{0}:{1:0.2f} '.format(det.segment, det.distance)
        print pretty_string
        print

        time.sleep(0.1)


def try_sensor_and_camera():
    import time
    import cv2
    cap = cv2.VideoCapture()
    if not cap.open(0):
        raise Exception("Error opening camera")

    m16_sensor = LeddarM16()
    while True:
        success, frame = cap.read()
        if not success:
            raise Exception("Error reading video frame")

        detections = m16_sensor.get_detections()
        pretty_string = ''
        for det in detections:
            pretty_string += '{1:0.1f} '.format(det.segment, det.distance)
        print pretty_string
        print
        cv2.putText(frame, pretty_string, (0, frame.shape[0]/2), 1, 1, (0, 255, 0))
        time.sleep(0.1)
        cv2.imshow('frame', frame)
        cv2.waitKey(30)


if __name__ == '__main__':
    try_sensor_and_camera()
