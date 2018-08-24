from mercurius.strategy import expert
import numpy as np
import logging
import tools

class anticort(expert):
    '''test
    anti-correlation olps
    '''
    def __init__(self, window=30):
        super(anticort, self).__init__()
        self.window = window

    def get_b(self, data, last_b):
        for k in np.arange(1,self.window):
            self.exp_w[k-1,:] = self.update(data, self.exp_w[k-1,:].T, k)

        numerator = 0
        denominator = 0

        for k in np.arange(1,self.window):
            numerator += self.exp_ret[k-1, 0] * self.exp_w[k-1,:]
            denominator += self.exp_ret[k-1,0]

        weight = numerator.T / denominator

        return weight

    def update(self, data,last_b, w):
        T, N = data.shape

        b = last_b
        if T >= 2*w:
            LX1 = np.log(data[T-2*w+1:T-w,:]) #(k,N)
            LX2 = np.log(data[T-w+1:T,:])

            mu1 = np.mean(LX1, axis=0)
            mu2 = np.mean(LX2, axis=0)

            n_LX1 = np.subtract(LX1, mu1)
            n_LX2 = np.subtract(LX2, mu2)

            std1 = np.std(n_LX1, axis=1)
            std2 = np.std(n_LX2, axis=1)
            std12 = np.dot(std1.T, std2) # sigma

            mu_matrix = np.ones((mu2.shape[0],mu2.shape[0]), dtype = bool)

            for i in range(N):
                for j in range(N):
                    if mu2[i] > mu2[j]:
                        mu_matrix[i,j] = True
                    else:
                        mu_matrix[i,j] = False

            mCv = np.zeros((N,N))
            mCorr = np.zeros((N,N))
            mCov = np.dot(n_LX1.T, n_LX2) / np.float64(w-1)

            mCorr = np.where(std12!=0, np.divide(mCov,std12), std12) #set

            s12 =  np.multiply(mCorr>0,mu_matrix)
            claim = np.zeros((N,N))
            claim[s12] += mCorr[s12]


            bool_claim = np.absolute(claim) > 0

            diag_mCorr = np.diag(mCorr)
            cor1 = np.maximum(0, np.tile(-diag_mCorr, (1,N))).reshape((N,N))
            cor2 = np.maximum(0, np.tile(-diag_mCorr.T, (N,1)))
            claim[s12] += (cor1[s12] + cor2[s12])

            transfer = np.zeros((N,N))

            sum_claim = np.tile(np.sum(claim, axis=1), (N,1))

            s1 = np.absolute(sum_claim) > 0
            w_b = np.tile(b, (N,1))
            transfer[s1] = w_b[s1] * claim[s1] / sum_claim[s1]

            transfer = np.where(np.isnan(transfer), 0, transfer)
            transfer_ij = transfer.T - transfer
            #print np.sum(transfer_ij)
            b -= np.sum(transfer_ij)
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

        self.exp_ret = np.ones((self.window-1,1))
        self.exp_w = np.ones((self.window-1,m)) / m

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

            for k in np.arange(1,self.window):
                self.exp_ret[k-1,0] = np.dot(self.exp_ret[k-1,0] * data[k,:],self.exp_w[k-1,:].T)
            self.exp_ret[:,0] /= np.sum(self.exp_ret[:,0])

            logging.info('%d\t%f\t%f\n' % (t+1, daily_ret[t], cumprod_ret[t]))

        logging.info('tc=%f, Final Return: %.2f\n' % (tc, cum_ret))

if __name__ == "__main__":
    tools.run(anticort())
