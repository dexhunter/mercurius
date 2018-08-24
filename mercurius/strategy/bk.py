from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class bk(expert):
    '''
    anti-correlation olps
    '''
    def __init__(self, K=5, L=10, c=1, exp_w=None):
        super(bk, self).__init__()
        self.K = K
        self.L = L
        self.c = c
        self.exp_ret = np.ones((K,L+1))

    def get_b(self, data, last_b):
        n, m = data.shape

        if self.exp_w is None:
            self.exp_w = np.ones((K*(L+1),m)) / m

        self.exp_w[self.K*self.L,:] = self.update(data, 0, 0, self.c)
        for k in np.arange(self.K):
            for l in np.arange(self.L):
                self.exp_w[(k-1)*self.L+l,:] = self.update(data, k, l, self.c)

        p = 1./(self.K*self.L)
        numerator = p * self.exp_ret[0,self.L] * self.exp_w[self.K*self.L,:]
        denominator = p * self.exp_ret[0, self.L]

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                numerator += p*self.exp_ret[k, l] * self.exp_w[(k-1)*self.L+l,:]
                denominator += p*self.exp_ret[k,l]

        weight = numerator.T / denominator

        return weight

    def update(self, data, k, l, c):
        T, N = data.shape
        m = 0
        histdata = np.zeros((T,N))

        if T <= k+1:
            return np.ones((1,N)) / N

        if k==0 and l==0:
            histdata = data[:T,:]
            m = T
        else:
            for i in np.arange(k+1,T):
                data2 = data[i-k:i-1,:] - data[T-k+1:T,:]

                if (np.sqrt(np.trace(data2.dot(data2.T)))) <= c/l:
                    m += 1
                    histdata[m,:] = data[i,:]

        if m == 0:
            return np.ones((1,N)) / N

        b = tools.opt_weights(histdata)
        return b

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

        self.exp_ret = np.ones((self.K, self.L+1))
        self.exp_w = np.ones((self.K*(self.L+1), m)) / m

        for t in range(n):
            if t>= min_period:
                b = self.get_b(data[:t,:], re_b)

            # double check (Normalize the constraint)
            b = b / np.sum(b)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret = cum_ret * daily_ret[t]
            cumprod_ret[t] = cum_ret

            re_b = b * data[t,:][:,None] / daily_ret[t]

            self.exp_ret[0, self.L] *= np.dot(data[t,:],self.exp_w[self.K*self.L,:])

            for k in np.arange(self.K):
                for l in np.arange(self.L):
                    self.exp_ret[k,l] *= np.dot(data[t,:],self.exp_w[(k-1)*self.L+l-1,:])

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

if __name__ == "__main__":
    tools.run(bk())
