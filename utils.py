import numpy as np
from sklearn import metrics

def mmd_rbf(X, Y, gamma=1.0, w = None):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    if (w is None):
        output = XX.mean() + YY.mean() - 2 * XY.mean()
    else:
        output = XX.mean() + np.sum(YY*np.outer(w, w)) - 2*np.sum(XY*np.outer(w, np.ones(X.shape[0])/X.shape[0]))
    return output