import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg, stats
from particles import resampling as rs
from particles import smc_samplers as ssp
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from scipy.spatial.distance import cdist

### utils
def gradient_banana(x, sigma):
#     return np.array([-(-400*(x[1, :]-x[0, :]**2)*x[0, :]-2*(1-x[0, :]))/20, -10*(x[1, :]-x[0, :]**2)])
    return np.array([-(-2*sigma**2*(x[1, :]-x[0, :]**2)*x[0, :]-(1-x[0, :])/sigma**2), -sigma**2*(x[1, :]-x[0, :]**2)])
def logpi_banana(x, sigma):
    return -sigma**2*(x[1, :]-x[0, :]**2)**2/2-(1-x[0, :])**2/(2*sigma**2)


### W
def ParallelULA(gamma, Niter, X0, sigma):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter):
        gradient = gradient_banana(X[i-1, :, :], sigma)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X
def ParallelMALA(gamma, Niter, X0, sigma):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = gradient_banana(X[i-1, :, :], sigma)
        prop = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X[i, :, :], accepted[i, :] = mala_accept_reject(prop, X[i-1, :, :], gamma, sigma)
    return X, accepted

def mala_accept_reject(prop, v, gamma, sigma):
    d = prop.shape[0]
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_banana(prop, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_banana(v, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_banana(prop, sigma) - logpi_banana(v, sigma) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted

### WFR
def SMC_WFR(gamma, Niter, X0, sigma):
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
        gradient_step = X[n-1, :, :] + gamma*gradient_banana(X[n-1, :, :], sigma)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
#         H = 2*gamma*np.eye(d) #-- original derivation
#         H = (4/(N*(d+2)))**(2/(d+4))*np.diag(np.var(X[n, :, :], axis = 1)) #-- KDE theory
#         squared_distances = pdist(X[n, :, :].T)
#         pairwise_squared_distances = squareform(squared_distances)**2
#         H = np.median(pairwise_squared_distances)/(2*np.log(N))*np.eye(d) # -- median euristic
#         kde_matrix = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, mean = np.zeros(d), cov = 2*gamma*np.eye(d)).reshape(N, N)
        kde_matrix =  metrics.pairwise.rbf_kernel(X[n, :, :].T, gradient_step.T, 1/(2*gamma))
        weight_denominator = np.mean(kde_matrix, axis = 1)
        logW = (1-np.exp(-gamma))*(logpi_banana(X[n, :, :], sigma)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_ULA(gamma, Niter, X0, sigma):
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
        gradient_step = X[n-1, :, :] + gamma*gradient_banana(X[n-1, :, :], sigma)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_banana(X[n, :, :], sigma) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_MALA(gamma, Niter, X0, sigma):
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
        gradient_step = X[n-1, :, :] + gamma*gradient_banana(X[n-1, :, :], sigma)
        prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        X[n, :, :], accepted[n, :] = mala_accept_reject(prop, X[n-1, :, :], gamma, sigma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_banana(X[n-1, :, :], sigma) +0.5*np.sum(X[n-1, :, :]**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_banana(X[n, :, :], sigma) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, accepted

### FR
def SMC_FR(gamma, Niter, X0, sigma):
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
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        prop = rwm_proposal(X[n-1, :, :].T, W[n-1, :]).T
        X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], l, sigma)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_banana(X[n, :, :], sigma))
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
def rwm_accept_reject(prop, v, l, sigma):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(logpi_banana(prop, sigma)-logpi_banana(v, sigma))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
