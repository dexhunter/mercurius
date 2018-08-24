from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class olmar(expert):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    def __init__(self, window=5, eps=10, b=None):
        """
        :param window: Lookback window.
        :param eps(epsilon): Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """
        super(olmar, self).__init__()
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps
        self.b = b

    def get_b(self, data, last_b):
        T, N = data.shape
        if self.b is None:
            self.b = np.ones(N) / N
        if T < self.window + 1 :
            data_phi=data[T-1,:]
        else:
            data_phi = np.zeros((1,N))
            tmp_x = np.ones((1,N))
            for i in range(self.window):
                data_phi += 1. /  tmp_x
                tmp_x *= data[T-i-1,:]
            data_phi *=  np.float32(1./self.window)

        ell = np.maximum(0, self.eps - data_phi.dot(last_b))
        #print data_phi
        x_bar = np.mean(data, dtype=np.float32)
        dd = data_phi - x_bar
        denominator = np.linalg.norm(dd) ** 2
        if denominator != 0:
            lam = ell /  denominator
        else:
            lam = 0
        self.b = self.b.ravel() + lam * dd
        self.b = tools.simplex_proj2(self.b.ravel())
        self.b = self.b[:,None]
        return self.b

if __name__ == "__main__":
    tools.run(olmar(), min_period=2)
