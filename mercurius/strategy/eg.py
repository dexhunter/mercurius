# -*- coding: utf-8 -*-
from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class eg(expert):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
        http://www.cis.upenn.edu/~mkearns/finread/helmbold98line.pdf
    """

    def __init__(self, eta=0.05, b=None):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(eg, self).__init__()
        self.eta = eta
        self.b = b

    def get_b(self, data, last_b):
        T, N = data.shape
        if self.b is None:
            self.b = np.ones((N,1)) / N
        last_b = self.b # the weight there is not rebalanced!
        self.b[:,0] = last_b[:,0] * np.exp(self.eta * data[T-1,:].T / (data[T-1,:].dot(last_b)))
        self.b = self.b/np.sum(self.b)
        return self.b

if __name__ == "__main__":
    tools.run(eg())
