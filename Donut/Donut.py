import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg, stats
from particles import resampling as rs
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from scipy.spatial.distance import cdist

def gradient_donut(X, ystar, sigma):
    precomputed_norm = linalg.norm(X, axis = 0)
    gradient = -2*(X/precomputed_norm)*(precomputed_norm - ystar)/(sigma**2)
    return gradient

def logpi_donut(X, ystar, sigma):
    precomputed_norm = linalg.norm(X, axis = 0)
    return -(precomputed_norm - ystar)**2/(sigma**2)


def ULA(gamma, Niter, ystar, sigma, X0):
    d = X0.size
    X = np.zeros((Niter, d))
    X[0, :] = X0
    for i in range(1, Niter):
        precomputed_norm = linalg.norm(X[i-1, :])
        gradient = (X[i-1, :]/precomputed_norm)*(precomputed_norm - ystar)/(sigma**2)
        X[i, :] = X[i-1, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d))
    return X
def ParallelULA(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter):
        gradient = gradient_donut(X[i-1, :, :], ystar, sigma)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X
def ParallelMALA(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = gradient_donut(X[i-1, :, :], ystar, sigma)
        prop = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X[i, :, :], accepted[i, :] = mala_accept_reject(prop, X[i-1, :, :], gamma, ystar, sigma)
    return X, accepted

def mala_accept_reject(prop, v, gamma, ystar, sigma):
    d = prop.shape[0]
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_donut(prop, ystar, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_donut(v, ystar, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_donut(prop, ystar, sigma) - logpi_donut(v, ystar, sigma) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted

def SMC_WFR(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
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
        gradient_step = X[n-1, :, :] + gamma*gradient_donut(X[n-1, :, :], ystar, sigma)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
#         kde_matrix = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, mean = np.zeros(d), cov = 2*gamma*np.eye(d)).reshape(N, N)
        kde_matrix =  metrics.pairwise.rbf_kernel(X[n, :, :].T, gradient_step.T, 1/(4*gamma))
        weight_denominator = np.mean(kde_matrix, axis = 1)
        logW = gamma*(logpi_donut(X[n, :, :], ystar, sigma)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_ULA(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
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
        gradient_step = X[n-1, :, :] + gamma*gradient_donut(X[n-1, :, :], ystar, sigma)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_donut(X[n, :, :], ystar, sigma) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_MALA(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    W = np.zeros((Niter, N))
    W[0, :] = np.ones(N)/N
    accepted = np.zeros((Niter, N))
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        gradient_step = X[n-1, :, :] + gamma*gradient_donut(X[n-1, :, :], ystar, sigma)
        prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        X[n, :, :], accepted[n, :] = mala_accept_reject(prop, X[n-1, :, :], gamma, ystar, sigma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_donut(X[n-1, :, :], ystar, sigma) +0.5*np.sum(X[n-1, :, :]**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_donut(X[n, :, :], ystar, sigma) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, accepted

### FR
def SMC_FR(gamma, Niter, ystar, sigma, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    W = np.zeros((Niter, N))
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter):
        t = gamma*n
        tpast = gamma*(n-1)
        l = 1-np.exp(-t)
        lpast = 1-np.exp(-tpast)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        prop = rwm_proposal(X[n-1, :, :].T, W[n-1, :]).T
        X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], l, ystar, sigma)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_donut(X[n, :, :], ystar, sigma))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W


def view_2d_array(theta):
    """Returns a view to record array theta which behaves
    like a (N,d) float array.
    """
    v = theta.view(float)
    N = theta.shape[0]
    v.shape = (N, -1)
    # raise an error if v cannot be reshaped without creating a copy
    return v
def rwm_proposal(v, W):
    arr = view_2d_array(v)
    N, d = arr.shape
    m, cov = rs.wmean_and_cov(W, arr)
    scale = 2.38 / np.sqrt(d)
    L = scale * linalg.cholesky(cov, lower=True)
    arr_prop = arr + stats.norm.rvs(size=arr.shape) @ L.T
    return arr_prop
def rwm_accept_reject(prop, v, l, ystar, sigma):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(logpi_donut(prop, ystar, sigma)-logpi_donut(v, ystar, sigma))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
