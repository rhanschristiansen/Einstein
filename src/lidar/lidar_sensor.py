import sys
import minimalmodbus
import numpy as np
import pandas as pd


class LidarSensor(object):
    def __init__(self, mode='lidar', simulation_data_file=None):
        try:
            self.distance = 0
            self.mode = mode
            self.instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)  # port name, slave address (in decimal)
            self.instrument.serial.baudrate = 115200  # Baud
            self.instrument.serial.bytesize = 8
            self.instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
            self.instrument.serial.stopbits = 1
            self.instrument.serial.timeout = 0.05  # seconds
            self.instrument.mode = minimalmodbus.MODE_RTU  # rtu or ascii mode
            # instrument.mode = minimalmodbus.MODE_ASCII   # rtu or ascii mode
            # instrument.debug = True
            self.simulation_data = None
        except Exception as e:
            print 'ERROR OPENING MODBUS CONNECTION: {}\nWill continue using simulation data...'.format(sys.exc_info())
            self.mode = 'simulation'
            if simulation_data_file is not None:
                self.simulation_data = pd.read_csv(simulation_data_file)

        self.DISTANCE_1_REGISTER = 24
        self.CM_TO_FT = 0.0328084

    def get_distance(self, frame_number=None):
        distance = None
        if self.mode == 'lidar':
            try:
                self.distance = self.instrument.read_register(self.DISTANCE_1_REGISTER, 1,
                                                              functioncode=4) * self.CM_TO_FT
            except IOError:
                # print("Failed to read from instrument at address {}".format(DISTANCE_1_REGISTER))
                pass
            except ValueError:
                # print("Failed to read from instrument at address {}".format(DISTANCE_1_REGISTER))
                pass
            return self.distance
        elif self.mode == 'simulation':
            self.distance = self.get_simulation_data(frame_number)
            return self.distance

    def get_simulation_data(self, frame_number):

        if self.simulation_data is None:
            return 12345
        else:
            # check whether frame number is out of bounds (no more sim data to be processed)
            if frame_number > self.simulation_data['Frame'].values.max():
                return -111
            # get the corresponding lidar distance for the video frame
            matching_dist = self.simulation_data.loc[
                self.simulation_data['Frame'] == frame_number, 'Distance Lidar'].values

            if len(matching_dist) == 0:
                return self.distance  # return previous value #TODO: revisit this
                # raise Exception('no data for frame {}!'.format(frame_number))
            return matching_dist[0]


def test():
    lidar_sensor = LidarSensor(mode='simulation',
                               simulation_data_file='/home/bob/PycharmProjects/Einstein/src/simulation_data/output43.csv')
    frame_number = 7
    while True:
        dist = lidar_sensor.get_distance(frame_number=frame_number)
        if dist == -111:
            print 'simulation data finished.'
            exit()
        if dist is not None:
            print dist
        frame_number += 1


if __name__ == '__main__':
    test()
