import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg, stats
from particles import resampling as rs
from particles import smc_samplers as ssp
from scipy.special import logsumexp
from sklearn import metrics
from scipy.spatial.distance import cdist

### utils
def gradient_banana(x, sigma):
#     return np.array([-(-400*(x[1, :]-x[0, :]**2)*x[0, :]-2*(1-x[0, :]))/20, -10*(x[1, :]-x[0, :]**2)])
    return np.array([-(-2*sigma**2*(x[1, :]-x[0, :]**2)*x[0, :]-(1-x[0, :])/sigma**2), -sigma**2*(x[1, :]-x[0, :]**2)])
def logpi_banana(x, sigma):
    return -sigma**2*(x[1, :]-x[0, :]**2)**2/2-(1-x[0, :])**2/(2*sigma**2)


### W
def ParallelULA(gamma, Niter, X0, sigma, true_sample, true_cov):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    for i in range(1, Niter):
        gradient = gradient_banana(X, sigma)
        X = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        mse_cov[i] = np.mean((np.cov(X) - true_cov)**2)
        for j in range(d):
            w1[i, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    return X, w1, mse_cov
def ParallelMALA(gamma, Niter, X0, sigma, true_sample, true_cov):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X) - true_cov)**2)
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = gradient_banana(X, sigma)
        prop = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X, accepted[i, :] = mala_accept_reject(prop, X, gamma, sigma)
        mse_cov[i] = np.mean((np.cov(X) - true_cov)**2)
        for j in range(d):
            w1[i, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    return X, accepted, w1, mse_cov

def mala_accept_reject(prop, v, gamma, sigma):
    d = prop.shape[0]
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_banana(prop, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_banana(v, sigma))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_banana(prop, sigma) - logpi_banana(v, sigma) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted

### WFR
def SMC_WFR(gamma, Niter, X0, sigma, true_sample, true_cov, nmcmc):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = X
            for j in range(1, nmcmc):
                gradient_step = Xmcmc + gamma*gradient_banana(Xmcmc, sigma)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_banana(X, sigma)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        distSq = -(1.0 / (4 * gamma))*cdist(X.T, gradient_step.T, metric='sqeuclidean')
        weight_denominator = logsumexp(distSq, axis=1)
        logW = (1-np.exp(-gamma))*(logpi_banana(X, sigma)-weight_denominator)
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, w1, mse_cov

def SMC_ULA(gamma, Niter, X0, sigma, true_sample, true_cov, nmcmc):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = X
            for j in range(1, nmcmc):
                gradient_step = Xmcmc + gamma*gradient_banana(Xmcmc, sigma)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_banana(X, sigma)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_banana(X, sigma) +0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, w1, mse_cov

def SMC_MALA(gamma, Niter, X0, sigma, true_sample, true_cov, nmcmc):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    if(nmcmc > 1):
        accepted = np.zeros((Niter, N, nmcmc-1))
    else:
        accepted = np.zeros((Niter, N))
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = X
            for j in range(1, nmcmc):
                gradient_step = Xmcmc + gamma*gradient_banana(Xmcmc, sigma)
                prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
                Xmcmc, accepted[n, :, j-1] = mala_accept_reject(prop, Xmcmc, gamma, sigma)
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_banana(X, sigma)
            prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X, accepted[n, :] = mala_accept_reject(prop, X, gamma, sigma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_banana(X, sigma) +0.5*np.sum(X**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_banana(X, sigma) +0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, accepted, w1, mse_cov

### FR
def SMC_FR(gamma, Niter, X0, sigma, true_sample, true_cov, nmcmc):
    d = X0.shape[1]
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    for n in range(1, Niter):
        t = gamma*n
        tpast = gamma*(n-1)
        l = 1-np.exp(-t)
        lpast = 1-np.exp(-tpast)
        Xold = np.copy(X)
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = Xold
            for j in range(1, nmcmc):
                prop = rwm_proposal(Xmcmc.T, W).T
                Xmcmc = rwm_accept_reject(prop, Xmcmc, l, sigma)
            X = Xmcmc
        else:
            prop = rwm_proposal(Xold.T, W).T
            X = rwm_accept_reject(prop, Xold, l, sigma)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X**2, axis = 0) + logpi_banana(X, sigma))
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, w1, mse_cov


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
