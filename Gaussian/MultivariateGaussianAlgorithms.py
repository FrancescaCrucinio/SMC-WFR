import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy import linalg, stats, spatial, optimize
from scipy.special import logsumexp
from scipy.linalg import expm
from particles import resampling as rs
from particles import smc_samplers as ssp
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from scipy.spatial.distance import cdist, mahalanobis
from scipy.linalg import cho_factor, cho_solve

def logpi_MultiGaussian(x, ms, Sigmas_inv):
    
    # Compute (x - m)
    xm = x - ms

    quad = np.sum(xm @ Sigmas_inv * xm, axis=1)
    
    return -0.5*quad


## Exact flows

def GaussmultiD_WFRexact(m0, C0, mpi, Cpi, t, dm):

    Cpiinv = np.linalg.inv(Cpi)
    Gam = Cpiinv + 0.5 * np.eye(dm)

    G = (
        expm(Gam * t)
        @ (np.linalg.inv(C0 - Cpi) + np.linalg.inv(2 * np.eye(dm) + Cpi))
        @ expm(Gam * t)
        - np.linalg.inv(Cpi + 2 * np.eye(dm))
    )

    Cwfr = Cpi + np.linalg.inv(G)

    mwfr = mpi + (Cwfr - Cpi) @ expm(t * Cpiinv) @ np.linalg.inv(C0 - Cpi) @ (m0 - mpi)

    return mwfr, Cwfr

def GaussmultiD_Wass(m0, C0, mpi, Cpi, t):

    Cpi_inv = np.linalg.inv(Cpi)
    Mt = expm(-t * Cpi_inv)

    dm = m0.shape[0]
    I = np.eye(dm)

    m = Mt @ m0 + (I - Mt) @ mpi
    C = Cpi + Mt @ (C0 - Cpi) @ Mt

    return m, C

def GaussmultiD_FR(m0, C0, mpi, Cpi, t):
    
    C0_inv = np.linalg.inv(C0)
    Cpi_inv = np.linalg.inv(Cpi)

    Cinv = Cpi_inv + np.exp(-t) * (C0_inv - Cpi_inv)
    C = np.linalg.inv(Cinv)

    m = mpi + np.exp(-t) * C @ C0_inv @ (m0 - mpi)

    return m, C
### ULA

def ParallelULA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0):
    d = ms.size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter):
        gradient = -np.matmul(Sigmas_inv, (X[i-1, :, :].T-ms).T)
        X[i, :, :] = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
    return X

def ParallelMALA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0):
    d = ms.size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = -np.matmul(Sigmas_inv, (X[i-1, :, :].T-ms).T)
        prop = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X[i, :, :], accepted[i, :] = mala_accept_reject(prop, X[i-1, :, :], ms, Sigmas, Sigmas_inv, gamma)
    return X, accepted
def mala_accept_reject(prop, v, ms, Sigmas, Sigmas_inv, gamma):
    d = ms.size
    gradient_prop = -np.matmul(Sigmas_inv, (prop.T-ms).T)
    gradient_v = -np.matmul(Sigmas_inv, (v.T-ms).T)
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_prop)).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_v)).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = multivariate_normal.logpdf(prop.T, ms, Sigmas) - multivariate_normal.logpdf(v.T, ms, Sigmas) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted

### WFR
def SMC_WFR(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    W = np.zeros((Niter, N))
    X[0, :] = X0.T
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
                gradient_step = Xmcmc[j-1, :] - gamma*np.matmul(Sigmas_inv, (Xmcmc[j-1, :].T-ms).T)
                Xmcmc[j, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] - gamma*np.matmul(Sigmas_inv, (X[n-1, :, :].T-ms).T)
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        distSq = -(1.0 / (4 * gamma))*cdist(X[n, :, :].T, gradient_step.T, metric='sqeuclidean')
# #         Hinv = (4/(N*(d+2)))**(-2/(d+4))*np.diag(1/np.var(gradient_step, axis = 1))
#         distSq = -cdist(X[n, :, :].T, gradient_step.T, metric='mahalanobis', VI=Hinv)**2
        weight_denominator = logsumexp(distSq, axis=1)
        logW = (1-np.exp(-gamma))*(logpi_MultiGaussian(X[n, :, :].T, ms, Sigmas_inv)-weight_denominator)
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_ULA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
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
                gradient_step = Xmcmc[j-1, :] - gamma*np.matmul(Sigmas_inv, (Xmcmc[j-1, :].T-ms).T)
                Xmcmc[j, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] - gamma*np.matmul(Sigmas_inv, (X[n-1, :, :].T-ms).T)
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_MultiGaussian(X[n, :, :].T, ms, Sigmas_inv) + 0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_MALA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
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
                gradient_step = Xmcmc[j-1, :] - gamma*np.matmul(Sigmas_inv, (Xmcmc[j-1, :].T-ms).T)
                prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
                Xmcmc[j, :], accepted[n, :, j-1] = mala_accept_reject(prop, Xmcmc[j-1, :], ms, Sigmas, Sigmas_inv, gamma)
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            gradient_step = X[n-1, :, :] - gamma*np.matmul(Sigmas_inv, (X[n-1, :, :].T-ms).T)
            prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X[n, :, :], accepted[n, :] = mala_accept_reject(prop, X[n-1, :, :], ms, Sigmas, Sigmas_inv, gamma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_MultiGaussian(X[n-1, :, :].T, ms, Sigmas_inv) +0.5*np.sum(X[n-1, :, :]**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_MultiGaussian(X[n, :, :].T, ms, Sigmas_inv) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, accepted

### FR
def SMC_UnitFR(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
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
        if(nmcmc > 1):
            Xmcmc = np.zeros((nmcmc, d, N))
            Xmcmc[0, :] = X[n-1, :, :]
            for j in range(1, nmcmc):
                prop = rwm_proposal(Xmcmc[j-1, :].T, W[n-1, :]).T
                Xmcmc[j, :] = rwm_accept_reject(prop, Xmcmc[j-1, :], ms, Sigmas, l)
            X[n, :] = Xmcmc[nmcmc-1, :]
        else:
            prop = rwm_proposal(X[n-1, :, :].T, W[n-1, :]).T
            X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], ms, Sigmas, l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_MultiGaussian(X[n, :, :].T, ms, Sigmas_inv))
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
def rwm_accept_reject(prop, v, ms, Sigmas, l):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(multivariate_normal.logpdf(prop.T, ms, Sigmas)-multivariate_normal.logpdf(v.T, ms, Sigmas))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
