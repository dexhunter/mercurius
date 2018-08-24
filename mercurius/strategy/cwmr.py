from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import pandas as pd
import scipy.stats
from numpy.linalg import inv
from numpy import diag, sqrt, log, trace
import logging


class cwmr(expert):
    """ Confidence weighted mean reversion.

    Reference:
        B. Li, S. C. H. Hoi, P.L. Zhao, and V. Gopalkrishnan.
        Confidence weighted mean reversion strategy for online portfolio selection, 2013.
        http://jmlr.org/proceedings/papers/v15/li11b/li11b.pdf
    """
    def __init__(self, eps=-0.5, confidence=0.95, b=None):
        """
        :param eps: Mean reversion threshold (expected return on current day must be lower
                    than this threshold). Recommended value is -0.5.
        :param confidence: Confidence parameter for profitable mean reversion portfolio.
                    Recommended value is 0.95.
        """
        super(cwmr, self).__init__()
        # input check
        if not (0 <= confidence <= 1):
            raise ValueError('confidence must be from interval [0,1]')

        self.eps = eps
        self.theta = scipy.stats.norm.ppf(confidence)
        self.b = b


    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        m = X.shape[1]
        self.sigma = np.matrix(np.eye(m) / m**2)


    def get_b(self, data, last_b):
        # initialize
        x = data[-1,:]
        x = np.reshape(x, (1, x.size))
        if self.b is None:
            self.b = np.ones(x.size) / x.size
        last_b = self.b
        last_b = np.reshape(last_b, (1, last_b.size))
        self.init_step(x)
        m = len(x)
        mu = np.matrix(last_b).T
        sigma = self.sigma
        theta = self.theta
        eps = self.eps
        x = np.matrix(x).T    # matrices are easier to manipulate

        # 4. Calculate the following variables
        M = mu.T * x
        V = x.T * sigma * x
        x_upper = sum(diag(sigma) * x) / trace(sigma)

        # 5. Update the portfolio distribution
        mu, sigma = self.update(x, x_upper, mu, sigma, M, V, theta, eps)

        # 6. Normalize mu and sigma
        mu = tools.simplex_proj2(mu)
        sigma = sigma / (m**2 * trace(sigma))
        """
        sigma(sigma < 1e-4*eye(m)) = 1e-4;
        """
        self.sigma = sigma
        return mu

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        # lambda from equation 7
        foo = (V - x_upper * x.T * np.sum(sigma, axis=1)) / M**2 + V * theta**2 / 2.
        a = foo**2 - V**2 * theta**4 / 4
        b = 2 * (eps - log(M)) * foo
        c = (eps - log(M))**2 - V * theta**2

        a,b,c = a[0,0], b[0,0], c[0,0]

        lam = max(0,
                  (-b + sqrt(b**2 - 4 * a * c)) / (2. * a),
                  (-b - sqrt(b**2 - 4 * a * c)) / (2. * a))
        # bound it due to numerical problems
        lam = min(lam, 1E+7)

        # update mu and sigma
        U_sqroot = 0.5 * (-lam * theta * V + sqrt(lam**2 * theta**2 * V**2 + 4*V))
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + theta * lam / U_sqroot * diag(x)**2)
        """
        tmp_sigma = inv(inv(sigma) + theta*lam/U_sqroot*diag(xt)^2);
        % Don't update sigma if results are badly scaled.
        if all(~isnan(tmp_sigma(:)) & ~isinf(tmp_sigma(:)))
            sigma = tmp_sigma;
        end
        """
        return mu, sigma

    def trade(self, data, tc=0, min_period=1):
        '''Meta Trader
        :param data:
        :param tc: transaction cost, default 0
        :param min_period: minimum period to start trading, default 1
        '''
        logging.info('------------------------------------------\n')
        logging.info('Parameters [tc: %f].\n' % tc)
        logging.info('day\t Daily Return\t Total Return\n')
        logging.info('------------------------------------------\n')

        n, m = data.shape

        cum_ret = 1 #cumulative return aka total return
        cumprod_ret = np.ones((n,1), np.float)
        daily_ret = np.ones((n, 1), np.float) #daily return

        b = np.ones((m,1), np.float) / m # initialize b
        re_b = np.zeros((m,1), np.float) # initialize rebalanced b
        daily_portfolio = np.zeros((n,m))

        for t in range(n):
            if t>= min_period:
                b = self.get_b(data[:t,:], re_b)

            # double check (Normalize the constraint)
            b = b / np.sum(b)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret *= daily_ret[t]
            cumprod_ret[t] = cum_ret
            re_b = b * np.reshape(data[t,:], (1,m)) / daily_ret[t]

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

        self.pDiff = daily_ret

if __name__ == '__main__':
    tools.run(cwmr())
