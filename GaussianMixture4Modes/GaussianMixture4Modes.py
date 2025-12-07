import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy import linalg, stats, spatial, optimize
from scipy.special import logsumexp
from particles import resampling as rs
from particles import smc_samplers as ssp
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from scipy.spatial.distance import cdist

### Gradient and other utilities
def gradient_4modes(x, ms, Sigmas, Sigmas_inv, weights):
    gradient = np.zeros(x.shape)
    denominator = np.zeros((weights.size, x.shape[1]))
    for j in range(weights.size):
        denominator[j, :] = weights[j]*multivariate_normal.pdf(x.T, ms[j,:], Sigmas[j,:,:])
        gradient += -denominator[j, :]*np.matmul(Sigmas_inv[j,:,:],(x.T-ms[j,:]).T)
    return gradient/np.sum(denominator, axis = 0)
def logpi_4modes(x, ms, Sigmas, weights):
    logpi = np.zeros((weights.size, x.shape[1]))
    for j in range(weights.size):
        logpi[j, :] = weights[j]*multivariate_normal.pdf(x.T, ms[j,:], Sigmas[j,:,:])
    return np.log(np.sum(logpi, axis = 0))


def BDL_kernelisedPDE(gamma, Niter, ms, Sigmas, Sigmas_inv, weights, N, h, X0):
    d = ms[0,:].size
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    for i in range(1, Niter):
        gradient = gradient_4modes(X[i-1, :, :], ms, Sigmas, Sigmas_inv, weights)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        beta = -logpi_4modes(X[i, :, :], ms, Sigmas, weights)
#         kernel_rate = metrics.pairwise.rbf_kernel(X[i, :, :].T, X[i, :, :].T, 1/h)*(2*h*np.pi)**(-d/2)
#         beta = beta + np.log(np.mean(kernel_rate, axis = 0))
        distSq = -(1.0 / (2 * h))*cdist(X[i, :, :].T, X[i, :, :].T, metric='sqeuclidean')
        kernel_rate = logsumexp(distSq, axis=1)-np.log(N)-0.5*d*np.log(2*h*np.pi)
        beta = beta + kernel_rate
        beta = beta - np.mean(beta)
        kill = (beta >= 0) * (np.random.uniform(size = N) < 1-np.exp(-beta*gamma))
        duplicate = (beta < 0) * (np.random.uniform(size = N) < 1-np.exp(beta*gamma))
        Xnew = np.concatenate((X[i, :, duplicate], X[i, :, duplicate]))
        Nres = Xnew.shape[0]
        if(Nres > N):
            tokeep = np.random.randint(Nres, size = N)
            X[i, :, :] = Xnew[tokeep, :].T
        else:
            toduplicate = np.random.randint(N, size = N-Nres)
            X[i, :, :] = np.concatenate((Xnew, X[i, :, toduplicate])).T
    return X
def BDL_kernelisedKL(gamma, Niter, ms, Sigmas, Sigmas_inv, weights, N, h, X0):
    d = ms[0,:].size
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    for i in range(1, Niter):
        gradient = gradient_4modes(X[i-1, :, :], ms, Sigmas, Sigmas_inv, weights)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        beta = -logpi_4modes(X[i, :, :], ms, Sigmas, weights)
        kernel_rate = metrics.pairwise.rbf_kernel(X[i, :, :].T, X[i, :, :].T, 1/h)*(2*h*np.pi)**(-d/2)
        beta = beta + np.log(np.mean(kernel_rate, axis = 0))
        beta = beta - np.mean(beta) - 1 + np.sum(kernel_rate/np.sum(kernel_rate, axis = 0), axis = 1)
        kill = (beta >= 0) * (np.random.uniform(size = N) < 1-np.exp(-beta*gamma))
        duplicate = (beta < 0) * (np.random.uniform(size = N) < 1-np.exp(beta*gamma))
        Xnew = np.concatenate((X[i, :, duplicate], X[i, :, duplicate]))
        Nres = Xnew.shape[0]
        if(Nres > N):
            tokeep = np.random.randint(Nres, size = N)
            X[i, :, :] = Xnew[tokeep, :].T
        else:
            toduplicate = np.random.randint(N, size = N-Nres)
            X[i, :, :] = np.concatenate((Xnew, X[i, :, toduplicate])).T
    return X

def SMC_WFR(gamma, Niter, ms, Sigmas, Sigmas_inv, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    W = np.zeros((Niter, N))
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        gradient_step = X[n-1, :, :] + gamma*gradient_4modes(X[n-1, :, :], ms, Sigmas, Sigmas_inv, weights)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        distSq = -(1.0 / (4 * gamma))*cdist(X[n, :, :].T, gradient_step.T, metric='sqeuclidean')
        weight_denominator = logsumexp(distSq, axis=1)
        logW = (1-np.exp(-gamma))*(logpi_4modes(X[n, :, :], ms, Sigmas, weights)-weight_denominator)
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W
