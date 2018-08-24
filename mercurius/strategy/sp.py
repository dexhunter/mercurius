from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class sp(expert):
    '''Switch Portfolio'''
    def __init__(self, gamma=0.25):
        super(sp, self).__init__()
        self.gamma = gamma

    def get_b(self, data, last_b):
        T, N = data.shape
        b = last_b * (1-self.gamma-self.gamma/(N-1)) + self.gamma/(N-1)
        b = b / np.sum(b)
        return b

if __name__ == "__main__":
    tools.run(sp())
