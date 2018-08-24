from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


class ons(expert):
    """
    Online newton step algorithm.

    Reference:
        A.Agarwal, E.Hazan, S.Kale, R.E.Schapire.
        Algorithms for Portfolio Management based on the Newton Method, 2006.
        http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_AgarwalHKS06.pdf
    """
    def __init__(self, delta=0.125, beta=1., eta=0., A=None, b=None):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super(ons, self).__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.A = A
        self.b = b

    def init_portfolio(self, X):
        m = X.shape[1]
        self.A = np.mat(np.eye(m))
        self.bt = np.mat(np.zeros(m)).T


    def get_b(self, data, last_b):
        '''
        :param x: input matrix
        :param last_b: last portfolio
        '''
        T, N = data.shape
        if self.b is None:
            self.b = np.ones(N) / N

        last_b = self.b # don't rebalance last_b

        if self.A is None:
            self.init_portfolio(data)

        # calculate gradient
        grad = np.mat(data[-1,:] / np.dot(last_b,data[-1,:])).T
        # update A
        self.A += grad * grad.T
        # update b
        self.bt += (1 + 1./self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.bt, self.A)
        self.b = pp * (1 - self.eta) + np.ones(N)/N * self.eta
        return self.b


    def projection_in_norm(self, x, M):
        """ Projection of x to simplex indiced by matrix M. Uses quadratic programming.
        """
        m = M.shape[0]

        P = matrix(2*M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m,1)))
        A = matrix(np.ones((1,m)))
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])

if __name__ == "__main__":
    tools.run(ons())
