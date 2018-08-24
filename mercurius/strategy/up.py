from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class up(expert):
    """
    Universal Portfolio

    T. M. Cover
    """
    def __init__(self):
        super(up, self).__init__()


    def get_b(self, data, last_b):
        del0 = 4e-3
        delta = 5e-3
        M = int(1e1)
        S = int(0.5e1)

        N = data.shape[1]

        r = np.ones((N,1)) / N
        b = np.ones(r.shape)

        allM = np.zeros((N,M))

        for m in range(M):
            b = r
            for i in range(S):
                bnew = np.copy(b)
                j = np.random.randint(low=1, high=N)
                a = np.random.randint(-1,1)
                if a == 0:
                    a = 1
                bnew[j] = b[j] + (a*delta)
                if bnew[j] >= del0 and bnew[N-1] >= del0:
                    x = self.findQ(b, data, del0, delta)
                    y = self.findQ(bnew, data, del0, delta)
                    pr = np.minimum(y/x, 1)
                    temp = np.random.uniform(0,1)
                    if temp < pr:
                        b = bnew

            allM[:,m] = b.conj().T

        return np.sum(allM, axis=1) / M

    def findQ(self, b, data, del0, delta):
        N = data.shape[1]
        P = np.prod(data.dot(b))
        Q = np.dot(P, np.minimum(1, np.exp((b[N-1]-(2*del0))/((N-1)*delta))))
        return Q

if __name__ == '__main__':
    tools.run(up())
