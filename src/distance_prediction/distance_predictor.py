class DistancePredictor(object):
    def __init__(self):
        self.f = 1100.0  # focal length
        self.d_z = 5.0  # distance from ground to camera
        self.distance = lambda v: self.f * self.d_z / v

    def predict(self, bbox):
        predictions = []
        x1, y2, x2, y2 = bbox
        predictions.append(self._ground_angle_prediction(y2))
        return predictions

    def _ground_angle_prediction(self, y2):
        return self.distance(y2)
