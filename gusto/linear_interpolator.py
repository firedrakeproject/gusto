import numpy as np

class linearinterpolator(object):

    def __init__(self, data_points, function_points):
        self.data_points = data_points
        self.function_points = function_points
         
    def interpolate(self, points):
        return np.interp(points, self.data_points, self.function_points)