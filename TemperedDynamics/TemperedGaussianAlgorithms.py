import numpy as np
from scipy.stats import norm
from scipy import linalg, stats, spatial
from particles import resampling as rs
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from scipy.spatial.distance import cdist

def FR_reweigthing(mu, sigma, X, gamma, delta, gradient_step):
    weight_denominator = np.mean(norm.pdf(X - gradient_step.reshape(-1,1), loc = 0, scale = np.sqrt(2*gamma)), axis = 0)
    logW = delta*(-0.5*(X-mu)**2/sigma-np.log(weight_denominator))
    return rs.exp_and_normalise(logW)

def W_move(mu, sigma, X, gamma, noise):
    gradient_step = X - gamma*(X - mu)/sigma
    Xnew = gradient_step + np.sqrt(2*gamma)*noise
    return Xnew, gradient_step

def tempered_W_move(mu, sigma, X, gamma, noise, l):
    gradient_step = X - gamma*(l*(X - mu)/sigma + (1-l)*X)
    Xnew = gradient_step + np.sqrt(2*gamma)*noise
    return Xnew, gradient_step

def SMC_reweigthing(mu, sigma, X, delta):
    logW = delta*(-0.5*(X-mu)**2/sigma+0.5*X**2)
    return rs.exp_and_normalise(logW)

def SMC_WFR(gamma, Niter, mu, sigma, mu0, sigma0, X0):
    N = X0.size
    X = np.zeros((Niter, N))
    W = np.zeros((Niter, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n, :] = X[n-1, ancestors]
        # MCMC move
        noise = np.random.normal(size = N)
        X[n, :], gradient_step = W_move(mu, sigma, X[n, :], gamma, noise)
        # reweight
        W[n, :] = FR_reweigthing(mu, sigma, X[n, :], gamma, 1-np.exp(-gamma), gradient_step)
    return X, W

def SMC_TWFR(gamma, Niter, mu, sigma, mu0, sigma0, X0, lseq, delta):
    N = X0.size
    X = np.zeros((Niter+1, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
        l = lseq[n-1]
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n, :] = X[n-1, ancestors]
        # MCMC move
        noise = np.random.normal(size = N)
        X[n, :], gradient_step = tempered_W_move(mu, sigma, X[n, :], gamma, noise, l)
        # reweight
        W[n, :] = FR_reweigthing(mu, sigma, X[n, :], gamma, delta, gradient_step)
    return X, W

def SMC_UnitFR(Niter, mu, sigma, mu0, sigma0, X0, gamma):
    N = X0.size
    X = np.zeros((Niter, N))
    W = np.zeros((Niter, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter):
        t = gamma*n
        tpast = gamma*(n-1)
        l = 1-np.exp(-t)
        lpast = 1-np.exp(-tpast)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :] = X[n-1, ancestors]
        # MCMC move
        prop = rwm_proposal(X[n-1, :], W[n-1, :])
        X[n, :] = rwm_accept_reject(prop, X[n-1, :], mu, sigma, mu0, sigma0, l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*(X[n, :]-mu0)**2/sigma0 - 0.5*(X[n, :]-mu)**2/sigma)
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
    return arr_prop.flatten()
def rwm_accept_reject(prop, v, mu, sigma, mu0, sigma0, l):
    log_acceptance = (1-l)*0.5*((v-mu0)**2 - (prop-mu0)**2)/sigma0 + 0.5*l*((v-mu)**2-(prop-mu)**2)/sigma
    accepted = np.log(np.random.uniform(size = v.size)) <= log_acceptance
    output = np.copy(v)
    output[accepted] = prop[accepted]
    return output