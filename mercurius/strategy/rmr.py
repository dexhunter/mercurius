from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class rmr(expert):
    ''' Robust Median Reversion

    Reference:


    '''
    def __init__(self, eps=5, W=5, b=None):
        '''
        :param eps: the parameter control the reversion threshold
        :pram W: the length of window
        '''
        super(rmr, self).__init__()
        self.eps = eps
        self.W = W
        self.b = b

    def get_b(self, data, last_b):
        pass

    def update(self, data_close, data, t1,  last_b, eps, W):
        #print data.shape
        T, N = data.shape
        if self.b is None:
            self.b = np.ones(N) / N

        last_b = self.b
        if t1 < W+2:
            x_t1 = data[t1-2, :]
        else:
            x_t1 = tools.l1_median_VaZh(data_close[(t1-W-1):(t1-2),:]) / data_close[(t1-2),:]

        if np.linalg.norm(x_t1 - x_t1.mean())**2 == 0:
            tao = 0
        else:
            tao = np.minimum(0, (x_t1.dot(last_b)-eps) / np.linalg.norm(x_t1 - x_t1.mean())**2)
        self.b -= tao * (x_t1 - np.mean(x_t1) * np.ones(x_t1.shape))
        self.b = tools.simplex_proj(self.b)
        return self.b

    def get_close(self, data):
        return np.cumprod(data,axis=0)

    def trade(self, data, tc=0, min_period=1):
        '''Meta Trader
mr_var:param data:
        :param tc: transaction cost, default 0
        :param min_period: minimum period to start trading, default 1
        '''
        logging.info('------------------------------------------\n')
        logging.info('Parameters [tc: %f].\n' % tc)
        logging.info('day\t Daily Return\t Total Return\n')
        logging.info('------------------------------------------\n')

        n, m = data.shape

        data_close = self.get_close(data)

        cum_ret = 1 #cumulative return aka total return
        cumprod_ret = np.ones((n,1), np.float)
        daily_ret = np.ones((n, 1), np.float) #daily return

        b = np.ones((m,1), np.float) / m # initialize b
        re_b = np.zeros((m,1), np.float) # initialize rebalanced b
        daily_portfolio = np.zeros((n,m))

        for t in range(n):

            b = self.update(data_close, data, t+1, self.b, self.eps, self.W)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret *= daily_ret[t]
            cumprod_ret[t] = cum_ret
            re_b = b * data[t,:][:,None] / daily_ret[t]

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

        self.pDiff = daily_ret

if __name__ == "__main__":
    tools.run(rmr())
