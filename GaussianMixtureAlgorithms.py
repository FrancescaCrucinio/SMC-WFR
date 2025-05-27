import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy import linalg, stats, spatial
from scipy.special import logsumexp
from particles import resampling as rs

### Gradient and other utilities
def gradient_mixture(x, ms, Sigmas, weights):
    v = x.view(float)
    N = x.shape[0]
    v.shape = (N, -1)
    gradient = np.zeros(v.shape)
    denominator = np.zeros((weights.size, v.shape[1]))
    for j in range(weights.size):
        denominator[j, :] = weights[j]*multivariate_normal.pdf(v.T, ms[j,:], Sigmas[j,:,:])
        gradient += -denominator[j, :]*np.matmul(linalg.inv(Sigmas[j,:,:]),(v.T-ms[j,:]).T)
    return gradient/np.sum(denominator, axis = 0)
def logpi_mixture(x, ms, Sigmas, weights):
    v = x.view(float)
    N = x.shape[0]
    v.shape = (N, -1)
    logpi = np.zeros((weights.size, v.shape[1]))
    for j in range(weights.size):
        logpi[j, :] = weights[j]*multivariate_normal.pdf(v.T, ms[j,:], Sigmas[j,:,:])
    return np.log(np.sum(logpi, axis = 0))

### ULA

def ParallelULA(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter+1, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter+1):
        gradient = gradient_mixture(X[i-1, :, :], ms, Sigmas, weights)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X
def TemperedULA(gamma, Niter, ms, Sigmas, weights, X0, lseq):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter+1, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter+1):
        l = lseq[i-1]
        gradient = l*gradient_mixture(X[i-1, :, :], ms, Sigmas, weights) - (1-l)*X[i-1, :, :]
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X

### WFR

def WFR(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter+1, d, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0.T
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        # reweight
        gaussian_convolution = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, np.zeros(d), 2*gamma*np.eye(d)).reshape(N, N)
        weight_denominator = np.mean(gaussian_convolution, axis = 1)
        logW = (1-np.exp(-gamma))*(logpi_mixture(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def TemperedWFR(gamma, Niter, ms, Sigmas, weights, X0, lseq, delta):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter+1, d, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0.T
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        l = lseq[n-1]
        gradient_step = X[n-1, :, :] + gamma*(l*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights) - (1-l)*X[n-1, :, :])
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        # reweight
        gaussian_convolution = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, np.zeros(d), 2*gamma*np.eye(d)).reshape(N, N)
        weight_denominator = np.mean(gaussian_convolution, axis = 1)
        logW = delta*(logpi_mixture(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W