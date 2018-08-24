from mercurius.strategy.base import expert
from mercurius.strategy import tools
import numpy as np
import logging

class anticor(expert):
    '''
    anti-correlation olps
    '''
    def __init__(self, window=30):
        super(anticor, self).__init__()
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
            sig1 = np.std(LX1, axis=0, ddof=1)
            mu2 = np.mean(LX2, axis=0)
            sig2 = np.std(LX2, axis=0, ddof=1)
            sigma = np.dot(sig1.T,sig2)

            mu_matrix = np.ones((mu2.shape[0],mu2.shape[0]), dtype = bool)
            for i in range(0, mu2.shape[0]):
                for j in range(0, mu2.shape[0]):
                    if mu2[i] > mu2[j]:
                        mu_matrix[i,j] = True
                    else:
                        mu_matrix[i,j] = False

            mCov = (1.0/np.float64(w-1)) * np.dot(np.transpose(np.subtract(LX1,mu1)),np.subtract(LX2,mu2))

            mCorr = np.where(sigma!=0, np.divide(mCov,sigma), sigma) #set

#Multiply the correlation matrix by the boolean matrix comparing mu2[i] to mu2[j] and by the boolean matrix where correlation matrix is greater than zero

            claim = mCorr * np.multiply(mCorr>0,mu_matrix)

#The boolean claim matrix will be used to obtain only the entries that meet the criteria that mu2[i] > mu2[j] and mCorr is > 0 for the i_corr and j_corr matrices

            bool_claim = claim > 0

#If stock i is negatively correlated with itself we want to add that correlation to all instances of i.  To do this, we multiply a matrix of ones by the diagonal of the correlation matrix row wise.

            i_corr = np.ones((mu1.shape[0],mu2.shape[0])) * np.diagonal(mCorr)[:,None]

#Since our condition is when the correlation is negative, we zero out any positive values, also we want to multiply by the bool_claim matrix to obtain valid entries only

            i_corr = np.where(i_corr > 0,0,i_corr)
            i_corr = i_corr * bool_claim

#Subtracting out these negative correlations is essentially the same as adding them to the claims matrix

            claim -= i_corr

#We repeat the same process for stock j except this time we will multiply the diagonal of the correlation matrix column wise

            j_corr = np.ones((mu1.shape[0],mu2.shape[0])) * np.diagonal(mCorr)[None,:]

#Since our condition is when the correlation is negative, we zero out any positive values again multiplying by the bool_claim matrix

            j_corr = np.where(j_corr > 0,0,j_corr)
            j_corr = j_corr * bool_claim

#Once again subtract these to obtain our final claims matrix
            claim -= j_corr

            sum_claim = np.sum(claim, axis=1) #(N,)
#Then divide each element of the claims matrix by the sum of it's corresponding row
            transfer = claim / sum_claim[:,None] #(N,N)
#Multiply the original weights to get the transfer matrix row wise
            transfer = np.multiply(transfer,b[:,None])
#Replace the nan with zeros in case the divide encountered any
            transfer = np.where(np.isnan(transfer),0,transfer)
#We don't transfer any stock to itself, so we zero out the diagonals
            #np.fill_diagonal(transfer,0) #does not influence result at all
#Create the new portfolio weight adjustments, by adding the j direction weights or the sum by columns and subtracting the i direction weights or the sum by rows
            adjustment = np.sum(transfer, axis=0) - np.sum(transfer,axis=1) #transfer_ij
            #print adjustment
            b += adjustment
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

        self.pDiff = daily_ret

if __name__ == "__main__":
    tools.run(anticor())
