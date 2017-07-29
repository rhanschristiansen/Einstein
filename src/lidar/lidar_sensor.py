import sys
import minimalmodbus


class LidarSensor(object):
    def __init__(self):
        self.mode = 'lidar'
        try:
            self.instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)  # port name, slave address (in decimal)
            self.instrument.serial.baudrate = 115200  # Baud
            self.instrument.serial.bytesize = 8
            self.instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
            self.instrument.serial.stopbits = 1
            self.instrument.serial.timeout = 0.05  # seconds
            self.instrument.mode = minimalmodbus.MODE_RTU  # rtu or ascii mode
            # instrument.mode = minimalmodbus.MODE_ASCII   # rtu or ascii mode
            # instrument.debug = True
        except Exception as e:
            print 'ERROR OPENING MODBUS CONNECTION: {}\nWill continue using simulation data...'.format(sys.exc_info())
            self.mode = 'simulation'

        self.DISTANCE_1_REGISTER = 24
        self.CM_TO_FT = 0.0328084

    def get_distance(self):
        distance = None
        if self.mode == 'lidar':
            try:
                distance = self.instrument.read_register(self.DISTANCE_1_REGISTER, 1, functioncode=4) * self.CM_TO_FT
            except IOError:
                # print("Failed to read from instrument at address {}".format(DISTANCE_1_REGISTER))
                pass
            except ValueError:
                # print("Failed to read from instrument at address {}".format(DISTANCE_1_REGISTER))
                pass
            return distance
        elif self.mode == 'simulation':
            return 12345


def test():
    lidar_sensor = LidarSensor()
    while True:
        dist = lidar_sensor.get_distance()
        if dist is not None:
            print dist

# test()
