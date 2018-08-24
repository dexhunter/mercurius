from strategy import expert
import tools
import numpy as np


class pamr_old(expert):
    """
    OLPS implementation, exactly same reuslt as olps
    Passive aggressive mean reversion strategy for portfolio selection.
    There are three variants with different parameters, see original article
    for details.

    Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/PAMR_ML_final.pdf
    """
    def __init__(self, eps=0.5, C=500, variant=0, b=None):
        """
        :param eps: Control parameter for variant 0. Must be >=0, recommended value is
                    between 0.5 and 1.
        :param C: Control parameter for variant 1 and 2. Recommended value is 500.
        :param variant: Variants 0, 1, 2 are available.
        """
        super(pamr_old, self).__init__()
        self.eps = eps
        self.C = C
        self.variant = variant
        self.b = b

    def get_b(self, data, last_b):
        if self.b is None:
            self.b = np.ones(data.shape[1]) / data.shape[1]

        # calculate return prediction
        b = self.update(self.b, data, self.eps, self.C)
        self.b = b
        return self.b


    def update(self, b, data, eps, C):
        """ Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous weights. """
        T, N = data.shape
        x = data[-1,:]
        x_mean = np.mean(x)
        le = np.maximum(0., np.dot(b, x) - eps)
        denominator = np.square(np.linalg.norm(x - x_mean))

        if self.variant == 0:
            tau = le / denominator
        elif self.variant == 1:
            tau = np.minimum(C, le / denominator)
        elif self.variant == 2:
            tau = le / (denominator + 0.5 / C)

        # limit lambda to avoid numerical problems
        tau = np.minimum(100000, tau)

        # update portfolio
        b = b - tau * (data[T-1,:] - x_mean)
        # project it onto simplex
        return tools.simplex_proj(b.ravel())

if __name__ == "__main__":
    tools.run(pamr_old())
