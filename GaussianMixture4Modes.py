import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg, stats
from particles import resampling as rs

def gradient_4modes(x, ms, Sigmas, weights):
    v = x.view(float)
    N = x.shape[0]
    v.shape = (N, -1)
    gradient = np.zeros(v.shape)
    denominator = np.zeros((weights.size, v.shape[1]))
    for j in range(weights.size):
        denominator[j, :] = weights[j]*multivariate_normal.pdf(v.T, ms[j,:], Sigmas[j,:,:])
        gradient += -denominator[j, :]*np.matmul(linalg.inv(Sigmas[j,:,:]),(v.T-ms[j,:]).T)
    return gradient/np.sum(denominator, axis = 0)
def logpi_4modes(x, ms, Sigmas, weights):
    v = x.view(float)
    N = x.shape[0]
    v.shape = (N, -1)
    logpi = np.zeros((weights.size, v.shape[1]))
    for j in range(weights.size):
        logpi[j, :] = weights[j]*multivariate_normal.pdf(v.T, ms[j,:], Sigmas[j,:,:])
    return np.log(np.sum(logpi, axis = 0))
def ULA(gamma, Niter, ms, Sigmas, weights):
    d = ms[0,:].size
    X = np.zeros((Niter, d))
    X[0, :] = np.random.multivariate_normal(ms[0,:], 0.3*np.eye(d))
    for i in range(1, Niter):
        gradient = gradient_4modes(X[i-1, :], ms, Sigmas, weights).flatten()
        X[i, :] = X[i-1, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d))
    return X
def BDL_kernelisedPDE(gamma, Niter, ms, Sigmas, weights, N, h, X0):
    d = ms[0,:].size
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    for i in range(1, Niter):
        gradient = gradient_4modes(X[i-1, :, :], ms, Sigmas, weights)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        beta = -logpi_4modes(X[i, :, :], ms, Sigmas, weights)
        kernel_rate = multivariate_normal.pdf(np.kron(X[i, :, :].T, np.ones((N, 1))) - np.tile(X[i, :, :], N).T, np.zeros(d), h*np.eye(d)).reshape(N, N)
        beta = beta + np.log(np.mean(kernel_rate, axis = 0))
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
def BDL_kernelisedKL(gamma, Niter, ms, Sigmas, weights, N, h, X0):
    d = ms[0,:].size
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    for i in range(1, Niter):
        gradient = gradient_4modes(X[i-1, :, :], ms, Sigmas, weights)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        beta = -logpi_4modes(X[i, :, :], ms, Sigmas, weights)
        kernel_rate = multivariate_normal.pdf(np.kron(X[i, :, :].T, np.ones((N, 1))) - np.tile(X[i, :, :], N).T, np.zeros(d), h*np.eye(d)).reshape(N, N)
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

def SMC_WFR(gamma, Niter, ms, Sigmas, weights, X0):
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
        gradient_step = X[n-1, :, :] + gamma*gradient_4modes(X[n-1, :, :], ms, Sigmas, weights)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        kde_matrix = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, mean = np.zeros(d), cov = 2*gamma*np.eye(d)).reshape(N, N)
        weight_denominator = np.mean(kde_matrix, axis = 1)
        logW = gamma*(logpi_4modes(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W