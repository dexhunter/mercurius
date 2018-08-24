import numpy as np
import pandas as pd
#from nntrader.tdagent.algorithms.olmar import OLMAR
from mercurius.strategy.olmar import olmar
import tools
import logging
import portfolioopt as pfopt

class rmr2(olmar):
    def __init__(self, window=5, eps=10, tau=1e-3):
        super(rmr2, self).__init__(window, eps)
        self.tau = tau

    def get_b(self, data, last_b):
        x = data
        x_mean = np.mean(x)
        lam = max(0., (self.eps - np.dot(last_b, x)) / norm(x - x_mean) **2)
        lam = min(1e+5, lam)
        b = last_b + lam * (x - x_mean)
        b = np.log(1. + np.exp(b))
        b = b / b.sum()
        b = pfopt.truncate_weights(b)
        return b

    def predict_rmr(self, x, history):
        y = history.mean()
        y_last = None
        while y_last is None or norm(y-y_last)/norm(y_last)>self.tau:
            y_last=y
            d=norm(history-y)
            y = history.div(d, axis=0).sum() / (1./d).sum()
        return y/x

    def get_close(self, data):
        return np.cumprod(data, axis=0)

    def trade(self, data, tc=0, min_period=1):
        n, m = data.shape
        history = self.get_close(data)
        x = history[-1]

        cum_ret = 1
        cumprod_ret = np.ones((n, 1), np.float)
        daily_ret = np.ones((n, 1), np.float)

        b = np.ones((m,1), np.float) / m
        re_b = np.zeros((m,1), np.float)
        daily_portfolio = np.zeros((n, m))

        for t in range(n):
            b = self.update(x, history)
            daily_portfolio[t,:] = b.reshape((1,m))
            daily_ret[t] = np.dot(data[t,:],b)*(1-tc/2*np.sum(np.absolute(b-re_b)))
            cum_ret *= daily_ret[t]
            cumprod_ret[t] = cum_ret
            re_b = b * data[t,:][:,None] / daily_ret[t]

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

        self.pDiff = daily_ret

def norm(x):
    if len(x.shape) == 1:
        axis=0
    else:
        axis=1
    return np.sqrt((x**2).sum(axis=axis))

if __name__ == "__main__":
    tools.run(rmr2())
