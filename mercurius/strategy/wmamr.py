from mercurius.strategy.pamr import pamr
from mercurius.strategy import tools
import numpy as np


class wmamr(pamr):
    """ Weighted Moving Average Passive Aggressive Algorithm for Online Portfolio Selection.
    It is just a combination of OLMAR and PAMR, where we use mean of past returns to predict
    next day's return.

    Reference:
        Li Gao, Weiguo Zhang
        Weighted Moving Averag Passive Aggressive Algorithm for Online Portfolio Selection, 2013.
        http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6643896
    """

    def __init__(self, window=5):
        """
        :param w: Windows length for moving average.
        """
        super(wmamr, self).__init__()

        if window < 1:
            raise ValueError('window parameter must be >=1')
        self.window = window


    def get_b(self, x, last_b):
        self.record_history(x)
        xx = np.mean(self.history[-self.window:,], axis=0)
        # calculate return prediction
        b = self.update(last_b, xx, self.eps, self.C)

        return b

