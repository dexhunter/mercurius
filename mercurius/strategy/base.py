import numpy as np
from time import time
from sys import exit
import logging
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

class expert(object):
    '''Meta class for trading algorithms
        according to run in OLPS

    '''
    def __init__(self, history=None, last_b=None, pDiff=None):
        '''init.
        :param hisotry: record history of relative price vector if True
        :param last_b: last portfolio weight
        :difference of portfolio: p(t)/p(t-1)
        '''
        self.history = history
        self.last_b = last_b
        self.pDiff = pDiff

    def get_b(self, data, last_b):
        '''get next portfolio weight vector.
        :param data: input matrix without the lastest price
        '''
        raise NotImplementedError('subclass must implement this!')

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
            #print(b)
            #print(data[t,:])
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            #print(daily_ret[t])
            cum_ret *= daily_ret[t]
            #print(cum_ret)
            if cum_ret == 0 or cum_ret > 10:
                print('too small or too large, quitting...')
                exit()
            cumprod_ret[t] = cum_ret
            re_b = b * data[t,:][:,None] / daily_ret[t]

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

        self.pDiff = daily_ret
        self.last_b = np.squeeze(b)

    def finish(self, re=True, plot=False):
        """
        :param re: if return the result
        :param plot: if plot the result
        """
        if re:
            result = {}
            result['portfolio'] = np.cumprod(self.pDiff)
            result['portfolio_diff'] = self.pDiff
            result['last_b'] = self.last_b
            return result

        if plot:
            name = self.__class__.__name__
            plt.plot(result['portfolio'], label=name)
            plt.title("Portfolio")
            plt.legend(name, fancybox=True)
            plt.xlabel('period')
            plt.ylabel("portfolio")
            plt.show()
            plt.savefig('{name}.png'.format(name=name))

