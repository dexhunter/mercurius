from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np

class olmar2(expert):
    '''Moving average reversion strategy for on-line portfolio selection

    Reference:
        Bin Li, Steven C.H. Hoi, Doyen Sahoo, Zhi-Yong Liu

    '''

    def __init__(self,  eps=10, alpha=0.5):
        '''init
        :param eps: mean reversion threshold
        :param alpha: trade off parameter for moving average
        '''
        super(olmar2, self).__init__()
        self.eps = eps
        self.alpha = alpha


    def get_b(self, data, last_b):
        T, N = data.shape
        data_phi = np.ones((1,N))
        data_phi = self.alpha + (1-self.alpha)*data_phi/data[T-1,:]


        ell = max(0, self.eps - data_phi.dot(last_b))

        x_bar = data_phi.mean()
        denominator = np.linalg.norm(data_phi - x_bar)**2

        if denominator == 0:
            lam = 0
        else:
            lam = ell / denominator

        data_phi = np.squeeze(data_phi)
        b = last_b.ravel() +  lam * (data_phi - x_bar)
        b = tools.simplex_proj(b.ravel())
        return b[:,None]

if __name__ == "__main__":
    tools.run(olmar2())
