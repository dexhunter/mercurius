from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np

class m0(expert):
    def __init__(self, beta=0.5, C=None):
        super(m0, self).__init__()
        self.beta = beta
        self.C = C

    def get_b(self, data, last_b):
        m = data.shape[1]
        if self.C is None:
            self.C = np.zeros((m,1))
        b = (self.C+self.beta)/(m*self.beta+np.ones((1,m)).dot(self.C))
        max_ind = np.argmax(data[-1,:])
        self.C[max_ind] += 1
        return b

if __name__ == "__main__":
    tools.run(m0())
