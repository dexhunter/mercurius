from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class best(expert):
    '''Benchmark '''
    def __init__(self, b=None):
        super(best, self).__init__()
        self.b = b

    def weights(self, data):
       n, m = data.shape
       tmp_cumprod_ret = np.cumprod(data, axis=0)
       best_ind = np.argmax(tmp_cumprod_ret[-1,:])
       weights = np.zeros((m,1))
       weights[best_ind] = 1
       return weights

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

if __name__ == '__main__':
    tools.run(best())
