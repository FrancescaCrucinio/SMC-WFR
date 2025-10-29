import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg, stats
from particles import resampling as rs
from sklearn import metrics
from scipy.spatial.distance import cdist

### utils
def gradient_funnel(x):
    d = x.shape[0]
    return np.vstack([-x[0, :]/9-(d-1)/2+np.sum(x[1:, :]**2/(2*np.exp(x[0, :])), axis = 0), -x[1:, :]/np.exp(x[0, :])])
def logpi_funnel(x):
    d = x.shape[0]
    return -x[0, :]**2/18 - x[0, :]*(d-1)/2 - np.sum(x[1:, :]**2/(2*np.exp(x[0, :])), axis = 0)

### W
def ParallelULA(gamma, Niter, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter):
        gradient = gradient_funnel(X[i-1, :, :])
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X
def ParallelMALA(gamma, Niter, X0):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = gradient_funnel(X[i-1, :, :])
        prop = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X[i, :, :], accepted[i, :] = mala_accept_reject(prop, X[i-1, :, :], gamma)
    return X, accepted

def mala_accept_reject(prop, v, gamma):
    d = prop.shape[0]
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_funnel(prop))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_funnel(v))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_funnel(prop) - logpi_funnel(v) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted

### WFR
def SMC_WFR(gamma, Niter, X0, nmcmc):
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
        if(nmcmc > 1):
            Xmcmc = np.zeros((nmcmc, d, N))
            Xmcmc[0, :] = X[n-1, :, :]
            for j in range(1, nmcmc):
                gradient_step = Xmcmc[j-1, :] + gamma*gradient_funnel(Xmcmc[j-1, :, :])
                Xmcmc[j, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] + gamma*gradient_funnel(X[n-1, :, :])
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        #         kde_matrix = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, mean = np.zeros(d), cov = 2*gamma*np.eye(d)).reshape(N, N)
        kde_matrix = metrics.pairwise.rbf_kernel(X[n, :, :].T, gradient_step.T, 1/(2*gamma))
        weight_denominator = np.mean(kde_matrix, axis = 1)
        logW = (1-np.exp(-gamma))*(logpi_funnel(X[n, :, :])-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_ULA(gamma, Niter, X0, nmcmc):
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
        if(nmcmc > 1):
            Xmcmc = np.zeros((nmcmc, d, N))
            Xmcmc[0, :] = X[n-1, :, :]
            for j in range(1, nmcmc):
                gradient_step = Xmcmc[j-1, :] + gamma*gradient_funnel(Xmcmc[j-1, :, :])
                Xmcmc[j, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] + gamma*gradient_funnel(X[n-1, :, :])
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_funnel(X[n, :, :]) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_MALA(gamma, Niter, X0, nmcmc):
    d = X0.shape[1]
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0,:,:] = X0.T
    W = np.zeros((Niter, N))
    W[0, :] = np.ones(N)/N
    if(nmcmc > 1):
        accepted = np.zeros((Niter, N, nmcmc-1))
    else:
        accepted = np.zeros((Niter, N))
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = np.zeros((nmcmc, d, N))
            Xmcmc[0, :] = X[n-1, :, :]
            for j in range(1, nmcmc):
                gradient_step = Xmcmc[j-1, :] + gamma*gradient_funnel(Xmcmc[j-1, :, :])
                prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
                Xmcmc[j, :], accepted[n, :, j-1] = mala_accept_reject(prop, Xmcmc[j-1, :], gamma)
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] + gamma*gradient_funnel(X[n-1, :, :])
            prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :, :], accepted[n, :] = mala_accept_reject(prop, X[n-1, :, :], gamma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_funnel(X[n-1, :, :]) +0.5*np.sum(X[n-1, :, :]**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_funnel(X[n, :, :]) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, accepted

### FR
def SMC_FR(gamma, Niter, X0, nmcmc):
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
        if(nmcmc > 1):
            Xmcmc = np.zeros((nmcmc, d, N))
            Xmcmc[0, :] = X[n-1, :, :]
            for j in range(1, nmcmc):
                prop = rwm_proposal(Xmcmc[j-1, :].T, W[n-1, :]).T
                Xmcmc[j, :] = rwm_accept_reject(prop, Xmcmc[j-1, :], l)
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            prop = rwm_proposal(X[n-1, :, :].T, W[n-1, :]).T
            X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_funnel(X[n, :, :]))
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
def rwm_accept_reject(prop, v, l):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(logpi_funnel(prop)-logpi_funnel(v))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
