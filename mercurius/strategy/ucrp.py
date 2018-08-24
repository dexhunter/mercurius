from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class ucrp(expert):
    """uniform constant rebalanced portfolio"""
    def __init__(self):
        super(ucrp, self).__init__()

    def get_b(self, data, last_b):
        n, m = data.shape
        return np.ones((m,1))/m

if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO)
    #data = tools.load_mat('djia')
    #test = ucrp()
    #test.trade(data)
    tools.run(ucrp())
