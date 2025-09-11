import numpy as np
from sklearn import metrics
from scipy import linalg, stats


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


def diagnostics(x, Niter, prob_val, true_sample, xx, yy, w = None):## only 1d for now
    diag = np.zeros((Niter, 6))
    dx = xx[1]-xx[0]
    if (w is None):
        for j in range(Niter):
            # mean
            diag[j, 0] = np.mean(x[j, :])
            # variance
            diag[j, 1] = np.var(x[j, :])
            # prob
            diag[j, 2] = np.mean(x[j, :]>prob_val)
            # w1
            diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten())
            # kl
            kde = stats.gaussian_kde(x[j, :])
            diag[j, 4] = dx*np.sum(kde(xx)*(np.log(kde(xx))-np.log(yy)))
            # mmd
            diag[j, 5] = mmd_rbf(true_sample.reshape(-1, 1), x[j, :].T)
    else:
        for j in range(Niter):
            # mean
            diag[j, 0] = np.sum(x[j, :]*w[j, :].T)
            # variance
            diag[j, 1] = np.sum(x[j, :]**2*w[j, :].T) - diag[j, 0]**2
            # prob
            diag[j, 2] = np.sum((x[j, :]>prob_val)*w[j, :].T)
            # w1
            diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten(), u_weights = w[j, :])
            # kl
            kde = stats.gaussian_kde(x[j, :], weights = w[j, :])
            diag[j, 4] = dx*np.sum(kde(xx)*(np.log(kde(xx))-np.log(yy)))
            # mmd
            diag[j, 5] = mmd_rbf(true_sample.reshape(-1, 1), x[j, :].T, w = w[j, :])
    return diag
    
    
def diagnosticsHD(x, Niter, prob_val, true_sample, true_mean, true_cov, xx, yy, dx, w = None):## only 1d for now
    d = x.shape[1]
    diag = np.zeros((Niter, 6))
    if (w is None):
        for j in range(Niter):
            # mean
            diag[j, 0] = np.sum((np.mean(x[j, :], axis = 1) - true_mean)**2)
            # variance
            diag[j, 1] = np.sum((np.cov(x[j, :]) - true_cov)**2)
            # prob
#             diag[j, 2] = np.mean(x[j, :]>prob_val, axis = 0)
#             tmp = 0
#             for i in range(d):
#                 # w1
#                 tmp += stats.wasserstein_distance(x[j, i, :].flatten(), true_sample[i, :].flatten())
#             diag[j, 3] = tmp/d
            # kl
            kde = stats.gaussian_kde(x[j, :])
            kde_eval = kde(xx)
            diag[j, 4] = dx**d*np.sum(kde_eval*(np.log(kde_eval)-np.log(yy)))
            # mmd
            diag[j, 5] = mmd_rbf(true_sample, x[j, :].T)
    else:
        for j in range(Niter):
            # mean
            diag[j, 0] = np.sum((np.sum(x[j, :]*w[j, :].T, axis = 1) - true_mean)**2)
#             # variance
#             diag[j, 1] = np.sum(x[j, :]**2*w[j, :].T) - diag[j, 0]**2
#             # prob
#             diag[j, 2] = np.sum((x[j, :]>prob_val)*w[j, :].T)
#             # w1
#             diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten(), u_weights = w[j, :])
            # kl
            kde = stats.gaussian_kde(x[j, :], weights = w[j, :])
            kde_eval = kde(xx)
            diag[j, 4] = dx**d*np.sum(kde_eval*(np.log(kde_eval)-np.log(yy)))
            # mmd
            diag[j, 5] = mmd_rbf(true_sample, x[j, :].T, w = w[j, :])
    return diag