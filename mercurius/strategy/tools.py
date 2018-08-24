import scipy.io as sio
import os
import sys
import logging
from time import time
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, euclidean

def run(algo, data=None, tc=0, min_period=1, plot=False):
    '''Run an algorithm on data with transaction fee
    :param algo: algorithm
    :param data: input data (relative price)
    :param tc: transaction fee
    '''
    logging.basicConfig(level=logging.INFO)
    try:
        if data is None:
            #data = random_portfolio()
            data = load_mat('djia')
        st = time() #start time
        algo.trade(data, tc=tc, min_period=min_period)
        logging.info("The trading time is %s" % (time()-st))

    finally:
        algo.finish(plot=plot)

def plot_result(result):
    pass

def load_mat(name):
    """ Load .mat from /data directory. """
    mod = sys.modules[__name__]
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', name + '.mat')
    fi = sio.loadmat(filename)
    return fi['data']

def random_portfolio():
    pass

def simplex_proj(v, s=1):
    """Compute the Euclidean projection on a positive simplex
    :param v: n-dimensional vector to project
    :param s: radius of the simplex

    Reference: J. Duchi
    """
    assert s > 0, "Radius s must be positive (%d <= 0)" % s

    n, = v.shape
    #check if already on the simplex
    if v.sum() == s and np.alltrue(v>=0):
        return v

    # get the array of cumulative sums of a sorted copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of >0 components of the optimal solution
    rho = np.nonzero(u*np.arange(1,n+1)>(cssv-s))[0][-1]
    #compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho]-s) / (rho + 1.)
    w = (v-theta).clip(min=0)
    return w

def simplex_proj2(y):
    m = len(y)
    bget = False

    s = sorted(y, reverse=True)
    tmpsum = 0.

    for i in range(m-1):
        tmpsum += s[i]
        tmax = (tmpsum - 1) / (i + 1)
        if tmax >= s[i+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] - 1) / m
    return np.maximum(0, y-tmax)

def l1_median_VaZh(X, eps=1e-5):
    '''calculate the L1_median of X with the l1median_VaZh method'''
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r==0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def corn_expert(self, data, w, c):
    '''
    :param w: window sze
    :param c: correlation coefficient threshold
    '''
    T, N = data.shape
    m = 0
    histdata = np.zeros((T,N))

    if T <= w+1:
        return np.ones(N) / N

    if w==0:
        histdata = data[:T,:]
        m = T
    else:
        for i in np.arange(w, T):
            d1 = data[i-w:i,:]
            d2 = data[T-w:T,:]
            datacorr = np.corrcoef(d1,d2)[0,1]

            if datacorr >= c:
                m += 1
                histdata[m,:] = data[i-1,:] #minus one to avoid out of bounds issue

    if m==0:
        return np.ones(N) / N

    #sqp according to OLPS implementation
    x_0 = np.ones((1,N)) / N
    objective = lambda b: -np.prod(np.dot(histdata, b))
    cons = ({'type': 'eq', 'fun': lambda b: 1-np.sum(b, axis=0)},)
    bnds = [(0.,1)]*N
    while True:
        res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp')
        eps = 1e-7
        if (res.x < 0-eps).any() or (res.x > 1+eps).any():
            data += np.random.randn(1)[0] * 1e-5
            logging.debug('Optimal portfolio weight vector not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution does not exist, use uniform pwv')
                res.x = np.ones(N) / N
            else:
                logging.warning('Converged but not successfully.')
            break

    return res.x

def opt_weights(X, max_leverage=1):
    x_0 = max_leverage * np.ones(X.shape[1]) / np.float(X.shape[1])
    objective = lambda b: -np.prod(X.dot(b))
    cons = ({'type': 'eq', 'fun': lambda b: max_leverage-np.sum(b)},)
    bnds = [(0., max_leverage)]*len(x_0)
    while True:
        res = minimize(objective, x_0, bounds=bnds, constraints=cons, method='slsqp', options={'ftol':1e-07})
        eps = 1e-7
        if (res.x < 0-eps).any() or (res.x > max_leverage+eps).any():
            X = X + np.random.randn(1)[0] * 1e-5
            logging.debug('Optimal weights not found, trying again')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                logging.warning('Solution not found')
                res.x = np.ones(X.shape[1]) / X.shape[1]
            else:
                logging.warning("converged but not successfully")
            break

    return res.x


if __name__ == '__main__':
    data = load_mat('djia')
    n, m = data.shape
    print(data.shape)
    print(data[:n,:].shape)
