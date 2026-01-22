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

def gradient_mixture(x, log_w, mus, invcovs, logdets):
    """
    x:        (d, N)
    log_w:    (K,)
    mus:      (K,d)
    invcovs:  (K,d,d)
    logdets:  (K,)
    returns:  (d, N)
    """
    d, N = x.shape
    K = mus.shape[0]

    # diff: (N,K,d)
    diff = x.T[:,None,:] - mus[None,:,:]

    quad = np.einsum('nkd,kdj,nkj->nk', diff, invcovs, diff)

    logNk = -0.5 * (d*np.log(2*np.pi) + logdets + quad)

    a = log_w + logNk
    a -= np.max(a, axis=1, keepdims=True)
    r = np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)   # (N,K)

    comp = -np.einsum('kij,nkj->nki', invcovs, diff)           # (N,K,d)

    score = np.einsum('nk,nkd->nd', r, comp)                  # (N,d)

    return score.T  

def logpi_mixture(x, log_w, mus, invcovs, logdets):
    """
    x:        (d, N)
    log_w:    (K,)
    mus:      (K,d)
    invcovs:  (K,d,d)
    returns:  (N,) unnormalized log p(x_n)
    """
    d, N = x.shape
    K = mus.shape[0]

    # diff: (N,K,d)
    diff = x.T[:,None,:] - mus[None,:,:]

    # quadratic form only
    quad = np.einsum('nkd,kdj,nkj->nk', diff, invcovs, diff)  # (N,K)

    # unnormalized log N_k(x) = -0.5 (x-mu)^T Σ⁻¹ (x-mu)
    logNk = -0.5 * quad-0.5*d*np.log(2*np.pi)-0.5*logdets

    # log-sum-exp with mixture weights
    a = log_w + logNk
    amax = np.max(a, axis=1, keepdims=True)
    logp = amax[:,0] + np.log(np.sum(np.exp(a - amax), axis=1))

    return logp

### ULA

def ParallelULA(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov):
    log_w = np.log(weights)
    d = ms[0,:].size
    N = X0.shape[0]
    X = X0.T
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    for i in range(1, Niter):
        gradient = gradient_mixture(X, log_w, ms, Sigmas_inv, logdets)
        X = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        mse_cov[i] = np.mean((np.cov(X) - true_cov)**2)
        for j in range(d):
            w1[i, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    return X, w1, mse_cov

def ParallelMALA(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov):
    log_w = np.log(weights)
    d = ms[0,:].size
    N = X0.shape[0]
    X = X0.T
    accepted = np.zeros((Niter, N))
    w1 = np.zeros((Niter, d))
    mse_cov = np.zeros(Niter)
    mse_cov[0] = np.mean((np.cov(X) - true_cov)**2)
    for j in range(d):
        w1[0, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    for i in range(1, Niter):
        gradient = gradient_mixture(X, log_w, ms, Sigmas_inv, logdets)
        prop = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X, accepted[i, :] = mala_accept_reject(prop, X, log_w, ms, Sigmas_inv, logdets, gamma)
        mse_cov[i] = np.mean((np.cov(X) - true_cov)**2)
        for j in range(d):
            w1[i, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:])
    return X, accepted, w1, mse_cov
def mala_accept_reject(prop, v, log_w, ms, Sigmas_inv, logdets, gamma):
    d = ms[0,:].size
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_mixture(prop, log_w, ms, Sigmas_inv, logdets))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_mixture(v, log_w, ms, Sigmas_inv, logdets))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_mixture(prop, log_w, ms, Sigmas_inv, logdets) - logpi_mixture(v, log_w, ms, Sigmas_inv, logdets) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted


### WFR

def SMC_WFR(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov, nmcmc):
    log_w = np.log(weights)
    d = ms[0,:].size
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
                gradient_step = Xmcmc + gamma*gradient_mixture(Xmcmc, log_w, ms, Sigmas_inv, logdets)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_mixture(X, log_w, ms, Sigmas_inv, logdets)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        distSq = -(1.0 / (4 * gamma))*cdist(X.T, gradient_step.T, metric='sqeuclidean')
        weight_denominator = logsumexp(distSq, axis=1)
        logW = (1-np.exp(-gamma))*(logpi_mixture(X, log_w, ms, Sigmas_inv, logdets) - weight_denominator)
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, w1, mse_cov

def SMC_ULA(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov, nmcmc):
    log_w = np.log(weights)
    d = ms[0,:].size
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
                gradient_step = Xmcmc + gamma*gradient_mixture(Xmcmc, log_w, ms, Sigmas_inv, logdets)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_mixture(X, log_w, ms, Sigmas_inv, logdets)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_mixture(X, log_w, ms, Sigmas_inv, logdets) +0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, w1, mse_cov

def SMC_MALA(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov, nmcmc):
    log_w = np.log(weights)
    d = ms[0,:].size
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
                gradient_step = Xmcmc + gamma*gradient_mixture(Xmcmc, log_w, ms, Sigmas_inv, logdets)
                prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
                Xmcmc, accepted[n, :, j-1] = mala_accept_reject(prop, Xmcmc, log_w, ms, Sigmas_inv, logdets, gamma)
            X = Xmcmc
        else:
            gradient_step = X + gamma*gradient_mixture(X, log_w, ms, Sigmas_inv, logdets)
            prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X, accepted[n, :] = mala_accept_reject(prop, X, log_w, ms, Sigmas_inv, logdets, gamma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_mixture(X, log_w, ms, Sigmas_inv, logdets) +0.5*np.sum(X**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_mixture(X, log_w, ms, Sigmas_inv, logdets) +0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        mse_cov[n] = np.mean((np.cov(X, aweights = W, bias = True) - true_cov)**2)
        for j in range(d):
            w1[n, j] = stats.wasserstein_distance(X[j,:], true_sample[j,:], u_weights = W)
    return X, W, accepted, w1, mse_cov

### FR
def SMC_UnitFR(gamma, Niter, ms, Sigmas, Sigmas_inv, logdets, weights, X0, true_sample, true_cov, nmcmc):
    log_w = np.log(weights)
    d = ms[0,:].size
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
            Xmcmc = np.copy(Xold)
            for j in range(1, nmcmc):
                prop = rwm_proposal(Xmcmc.T, W).T
                Xmcmc = rwm_accept_reject(prop, Xmcmc, ms, Sigmas_inv, log_w, l)
            X = Xmcmc
        else:
            prop = rwm_proposal(Xold.T, W).T
            X = rwm_accept_reject(prop, Xold, ms, Sigmas_inv, log_w, logdets, l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X**2, axis = 0) + logpi_mixture(X, log_w, ms, Sigmas_inv, logdets))
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
def rwm_accept_reject(prop, v, ms, Sigmas_inv, log_w, logdets, l):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(logpi_mixture(prop, log_w, ms, Sigmas_inv, logdets)-logpi_mixture(v, log_w, ms, Sigmas_inv, logdets))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
