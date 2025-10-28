import numpy as np
from sklearn import metrics
from scipy import linalg, stats
import numpy as np
from scipy.spatial.distance import cdist

def imq_kernel_matrix(X, c=1.0, beta=0.5):
    """
    Compute the IMQ kernel matrix and squared pairwise distances.

    k(x, x') = (c^2 + ||x - x'||^2)^(-beta)
    """
    sq_dists = cdist(X, X, 'sqeuclidean')  # (n, n)
    K = (c**2 + sq_dists) ** -beta
    return K, sq_dists

def kernel_stein_discrepancy_imq_weighted(samples, scores, weights=None, c=1.0, beta=0.5):
    """
    Compute Kernel Stein Discrepancy (KSD) using IMQ kernel with optional weights.

    Parameters:
        samples : (n, d) numpy array
            Samples from q(x)
        score_fn : function
            Computes ∇ log p(x) for each sample x
        weights : (n,) array or None
            Optional weights for samples. If None, use uniform weights.
        c : float
            IMQ kernel scale parameter
        beta : float
            IMQ kernel exponent parameter (0 < beta < 1)

    Returns:
        float : Weighted KSD value
    """
    n, d = samples.shape

    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)  # normalize

#     scores = np.array([score_fn(x) for x in samples])  # (n, d)
    K, sq_dists = imq_kernel_matrix(samples, c, beta)
    denom = (c**2 + sq_dists)

    # Pairwise diffs
    X_i = samples[:, np.newaxis, :]  # (n, 1, d)
    X_j = samples[np.newaxis, :, :]  # (1, n, d)
    diff = X_i - X_j  # (n, n, d)

    # Stein kernel terms
    term1 = np.dot(scores, scores.T) * K
    grad_k_i = -2 * beta * diff / denom[:, :, np.newaxis] * K[:, :, np.newaxis]
    grad_k_j = -grad_k_i  # symmetric in IMQ
    term2 = np.einsum("ik,ijk->ij", scores, grad_k_i)
    term3 = np.einsum("jk,ijk->ij", scores, grad_k_j)
    term4 = (
        2 * beta * K * ((2 * beta + 1) * sq_dists / denom**2 - d / denom)
    )

    H = term1 + term2 + term3 + term4  # (n, n)

    # Weighted sum: sum_i sum_j w_i w_j H_ij
    W = np.outer(weights, weights)  # (n, n)
    ksd_squared = np.sum(W * H)
    return np.sqrt(ksd_squared)

def mmd_rbf(X, Y, gamma=1, w = None):
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
        output = XX.mean() + np.sum(YY*np.outer(w, w)) - 2*np.sum(XY*np.outer(w, np.ones(X.shape[0])/X.shape[0]).T)
    return output


# def diagnostics(x, Niter, prob_val, true_sample, xx, yy, w = None):## only 1d for now
#     diag = np.zeros((Niter, 6))
#     dx = xx[1]-xx[0]
#     if (w is None):
#         for j in range(Niter):
#             # mean
#             diag[j, 0] = np.mean(x[j, :])
#             # variance
#             diag[j, 1] = np.var(x[j, :])
#             # prob
#             diag[j, 2] = np.mean(x[j, :]>prob_val)
#             # w1
#             diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten())
#             # kl
#             kde = stats.gaussian_kde(x[j, :])
#             diag[j, 4] = dx*np.sum(kde(xx)*(np.log(kde(xx))-np.log(yy)))
#             # mmd
#             diag[j, 5] = mmd_rbf(true_sample.reshape(-1, 1), x[j, :].T)
#     else:
#         for j in range(Niter):
#             # mean
#             diag[j, 0] = np.sum(x[j, :]*w[j, :].T)
#             # variance
#             diag[j, 1] = np.sum(x[j, :]**2*w[j, :].T) - diag[j, 0]**2
#             # prob
#             diag[j, 2] = np.sum((x[j, :]>prob_val)*w[j, :].T)
#             # w1
#             diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten(), u_weights = w[j, :])
#             # kl
#             kde = stats.gaussian_kde(x[j, :], weights = w[j, :])
#             diag[j, 4] = dx*np.sum(kde(xx)*(np.log(kde(xx))-np.log(yy)))
#             # mmd
#             diag[j, 5] = mmd_rbf(true_sample.reshape(-1, 1), x[j, :].T, w = w[j, :])
#     return diag
    
    
# def diagnosticsHD(x, Niter, prob_val, true_sample, true_mean, true_cov, xx, yy, dx, w = None):## only 1d for now
#     d = x.shape[1]
#     diag = np.zeros((Niter, 6))
#     if (w is None):
#         for j in range(Niter):
#             # mean
#             diag[j, 0] = np.sum((np.mean(x[j, :], axis = 1) - true_mean)**2)
#             # variance
#             diag[j, 1] = np.sum((np.cov(x[j, :]) - true_cov)**2)
#             # prob
# #             diag[j, 2] = np.mean(x[j, :]>prob_val, axis = 0)
# #             tmp = 0
# #             for i in range(d):
# #                 # w1
# #                 tmp += stats.wasserstein_distance(x[j, i, :].flatten(), true_sample[i, :].flatten())
# #             diag[j, 3] = tmp/d
#             # kl
#             kde = stats.gaussian_kde(x[j, :])
#             kde_eval = kde(xx)
#             diag[j, 4] = dx**d*np.sum(kde_eval*(np.log(kde_eval)-np.log(yy)))
#             # mmd
#             diag[j, 5] = mmd_rbf(true_sample, x[j, :].T)
#     else:
#         for j in range(Niter):
#             # mean
#             diag[j, 0] = np.sum((np.sum(x[j, :]*w[j, :].T, axis = 1) - true_mean)**2)
# #             # variance
# #             diag[j, 1] = np.sum(x[j, :]**2*w[j, :].T) - diag[j, 0]**2
# #             # prob
# #             diag[j, 2] = np.sum((x[j, :]>prob_val)*w[j, :].T)
# #             # w1
# #             diag[j, 3] = stats.wasserstein_distance(x[j, :].flatten(), true_sample.flatten(), u_weights = w[j, :])
#             # kl
#             kde = stats.gaussian_kde(x[j, :], weights = w[j, :])
#             kde_eval = kde(xx)
#             diag[j, 4] = dx**d*np.sum(kde_eval*(np.log(kde_eval)-np.log(yy)))
#             # mmd
#             diag[j, 5] = mmd_rbf(true_sample, x[j, :].T, w = w[j, :])
#     return diag