from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging
from scipy.optimize import minimize

class bcrp(expert):
    """ Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed with hindsight. It is often used as benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        super(bcrp, self).__init__()
        self.b = b

    def weights(self, data):
        """ Find weights which maximize return on X in hindsight! """
        return opt_weights(data)

    def get_b(self, data, last_b):
        pass

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

        b = self.weights(data)
        for t in range(n):

            # double check (Normalize the constraint)
            b = b / np.sum(b)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret *= daily_ret[t]
            cumprod_ret[t] = cum_ret

            re_b = b * data[t,:][:,None] / daily_ret[t]

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))
        self.pDiff = daily_ret
        self.last_b = b

def opt_weights(X, max_leverage=1):
    x_0 = max_leverage * np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.prod(X.dot(b))
    cons = ({'type': 'eq', 'fun': lambda b: max_leverage-np.sum(b)},)
    bnds = [(0., max_leverage)]*len(x_0)
    res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp', options={'ftol': 1e-07})
    return res.x


if __name__ == '__main__':
    tools.run(bcrp())
