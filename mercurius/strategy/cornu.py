from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class cornu(expert):
    '''
    Correlation driven non parametric Uniform
    '''
    def __init__(self, K=5, L=1, c=0.1, exp_w=None):
        '''
        :param K: maximum window size
        :param L: splits into L parts, in each K
        '''
        super(cornu, self).__init__()
        self.K = K
        self.L = L
        self.c = c
        self.exp_ret = np.ones((K,L))
        self.exp_w = exp_w


    def get_b(self, data, last_b):

        for k in np.arange(self.K):
            for l in np.arange(self.L):
                self.exp_w[(k-1)*self.L+l,:] = self.update(data, k, self.c)


        numerator = 0
        denominator = 0

        p = 1./(self.K*self.L)
        for k in np.arange(self.K):
            for l in np.arange(self.L):
                numerator += p * self.exp_ret[k,l] * self.exp_w[(k-1)*self.L+l,:]
                denominator += p * self.exp_ret[k,l]

        b = np.divide(numerator.T , denominator )
        return b

    def update(self, data, w, c):
        '''
        :param w: window sze
        :param c: correlation coefficient threshold
        '''
        T, N = data.shape
        m = 0
        histdata = np.zeros((T,N))

        if T <= w+1:
            '''use uniform portfolio weight vector'''
            return np.ones(N) / N

        if w==0:
            histdata = data[:T,:]
            m = T
        else:
            for i in np.arange(w, T):
                d1 = data[i-w:i-1,:].ravel()
                d2 = data[T-w+1:T,:].ravel()
                datacorr = np.corrcoef(d1,d2)[1,0]
                if datacorr >= c:
                    m += 1
                    histdata[m,:] = data[i,:] #minus one to avoid out of bounds issue

        if m==0:
            return np.ones(N) / N

        b = tools.opt_weights(histdata[:m])
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

        self.exp_ret = np.ones((self.K,self.L))
        self.exp_w = np.ones((self.K*self.L,m)) / m

        for t in range(n):
            if t>= min_period:
                b = self.get_b(data[:t-1,:], re_b)

            # double check (Normalize the constraint)
            b = b / np.sum(b)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret = cum_ret * daily_ret[t]
            cumprod_ret[t] = cum_ret

            re_b = b * data[t,:][:,None] / daily_ret[t]

            self.exp_ret[:,0] *= np.dot(data[t,:],self.exp_w.T)

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

        self.pDiff = daily_ret

if __name__ == "__main__":
    tools.run(cornu())
