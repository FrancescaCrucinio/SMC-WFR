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

def KL(mu0, Sigma0, mu1, Sigma1):
    """
    Computes KL(N0 || N1) for multivariate Gaussians.
    
    Parameters
    ----------
    mu0 : (k,) array_like
        Mean vector of distribution 0.
    Sigma0 : (k,k) array_like
        Covariance matrix of distribution 0.
    mu1 : (k,) array_like
        Mean vector of distribution 1.
    Sigma1 : (k,k) array_like
        Covariance matrix of distribution 1.
    
    Returns
    -------
    float
        KL divergence KL(N0 || N1).
    """
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    Sigma0 = np.asarray(Sigma0)
    Sigma1 = np.asarray(Sigma1)
    
    k = mu0.shape[0]

    # Compute inverse and determinants
    invSigma1 = np.linalg.inv(Sigma1)
    detSigma0 = np.linalg.det(Sigma0)
    detSigma1 = np.linalg.det(Sigma1)

    # Mahalanobis term
    diff = mu1 - mu0
    mahal = diff.T @ invSigma1 @ diff

    # Trace term
    trace_term = np.trace(invSigma1 @ Sigma0)

    # KL divergence
    kl = 0.5 * (trace_term + mahal - k + np.log(detSigma1 / detSigma0))

    return kl

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

def W_move(ms, Sigmas_inv, X, gamma, noise):
    gradient = -np.matmul(Sigmas_inv, (X.T-ms).T)
    gradient_step = X + gamma*gradient
    Xnew = gradient_step + np.sqrt(2*gamma)*noise
    return Xnew, gradient_step

def FR_reweigthing(ms, Sigmas_inv, X, gamma, delta, gradient_step):
    distSq = -(1.0 / (4 * gamma))*cdist(X.T, gradient_step.T, metric='sqeuclidean')
    weight_denominator = logsumexp(distSq, axis=1)
    logW = delta*(logpi_MultiGaussian(X.T, ms, Sigmas_inv)-weight_denominator)
    return rs.exp_and_normalise(logW)

### ULA

def ParallelULA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0):
    d = ms.size
    N = X0.shape[0]
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
    X = X0.T
    for i in range(1, Niter):
        gradient = -np.matmul(Sigmas_inv, (X.T-ms).T)
        X = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        m = np.mean(X, axis = 1)
        C = np.cov(X)
        kl[i] = KL(m, C, ms, Sigmas)
    return X, kl

def ParallelMALA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0):
    d = ms.size
    N = X0.shape[0]
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
    X = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = -np.matmul(Sigmas_inv, (X.T-ms).T)
        prop = X + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X, accepted[i, :] = mala_accept_reject(prop, X, ms, Sigmas, Sigmas_inv, gamma)
        m = np.mean(X, axis = 1)
        C = np.cov(X)
        kl[i] = KL(m, C, ms, Sigmas)
    return X, accepted, kl

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
    X = X0.T
    W = np.ones(N)/N
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = X
            for j in range(1, nmcmc):
                gradient_step = Xmcmc - gamma*np.matmul(Sigmas_inv, (Xmcmc.T-ms).T)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X - gamma*np.matmul(Sigmas_inv, (X.T-ms).T)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        distSq = -(1.0 / (4 * gamma))*cdist(X.T, gradient_step.T, metric='sqeuclidean')
#         Hinv = (4/(N*(d+2)))**(-2/(d+4))*np.diag(1/np.var(gradient_step, axis = 1))
#         distSq = -cdist(X[n, :, :].T, gradient_step.T, metric='mahalanobis', VI=Hinv)**2
        weight_denominator = logsumexp(distSq, axis=1)
        logW = (1-np.exp(-gamma))*(logpi_MultiGaussian(X.T, ms, Sigmas_inv)-weight_denominator)
        W = rs.exp_and_normalise(logW)
        m = np.sum(X*W, axis = 1)
        C = np.cov(X, aweights = W, bias = True)
        kl[n] = KL(m, C, ms, Sigmas)
    return X, W, kl

def SMC_ULA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
    for n in range(1, Niter):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W)
            X = X[:, ancestors]
        # MCMC move
        if(nmcmc > 1):
            Xmcmc = X
            for j in range(1, nmcmc):
                gradient_step = Xmcmc - gamma*np.matmul(Sigmas_inv, (Xmcmc.T-ms).T)
                Xmcmc = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X = Xmcmc
        else:
            gradient_step = X - gamma*np.matmul(Sigmas_inv, (X.T-ms).T)
            X = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_MultiGaussian(X.T, ms, Sigmas_inv) + 0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        m = np.sum(X*W, axis = 1)
        C = np.cov(X, aweights = W, bias = True)
        kl[n] = KL(m, C, ms, Sigmas)
    return X, W, kl


def SMC_MALA(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
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
                gradient_step = Xmcmc - gamma*np.matmul(Sigmas_inv, (Xmcmc.T-ms).T)
                prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
                Xmcmc, accepted[n, :, j-1] = mala_accept_reject(prop, Xmcmc, ms, Sigmas, Sigmas_inv, gamma)
            X = Xmcmc
        else:
            gradient_step = X - gamma*np.matmul(Sigmas_inv, (X.T-ms).T)
            prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
            X, accepted[n, :] = mala_accept_reject(prop, X, ms, Sigmas, Sigmas_inv, gamma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_MultiGaussian(X.T, ms, Sigmas_inv) +0.5*np.sum(X**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_MultiGaussian(X.T, ms, Sigmas_inv) +0.5*np.sum(X**2, axis = 0))
        W = rs.exp_and_normalise(logW)
        m = np.sum(X*W, axis = 1)
        C = np.cov(X, aweights = W, bias = True)
        kl[n] = KL(m, C, ms, Sigmas)
    return X, W, accepted, kl

### FR
def SMC_UnitFR(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
    N = X0.shape[0]
    X = X0.T
    W = np.ones(N)/N
    kl = np.zeros(Niter)
    kl[0] = KL(np.zeros(d), np.eye(d), ms, Sigmas)
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
                Xmcmc = rwm_accept_reject(prop, Xmcmc, ms, Sigmas, l)
            X = Xmcmc
        else:
            prop = rwm_proposal(Xold.T, W).T
        X = rwm_accept_reject(prop, Xold, ms, Sigmas, l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X**2, axis = 0) + logpi_MultiGaussian(X.T, ms, Sigmas_inv))
        W = rs.exp_and_normalise(logW)
        m = np.sum(X*W, axis = 1)
        C = np.cov(X, aweights = W, bias = True)
        kl[n] = KL(m, C, ms, Sigmas)
    return X, W, kl


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


### WFR Exact weights
def SMC_WFR_exactW(gamma, Niter, ms, Sigmas, Sigmas_inv, X0, nmcmc):
    d = ms.size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    W = np.zeros((Niter, N))
    X[0, :] = X0.T
    W[0, :] = np.ones(N)/N
    m_seq = np.zeros((Niter, d))
    C_seq = np.zeros((Niter, d, d))
    C_seq[0, :] = np.eye(d)
    for n in range(1, Niter):
        # exact WFR splitting
        m_tmp, C_tmp = GaussmultiD_Wass(m_seq[n-1, :], C_seq[n-1, :], ms, Sigmas, gamma)
        m_seq[n, :], C_seq[n, :] = GaussmultiD_FR(m_tmp, C_tmp, ms, Sigmas, gamma)
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
        logW = (1-np.exp(-gamma))*(logpi_MultiGaussian(X[n, :, :].T, ms, Sigmas_inv)-logpi_MultiGaussian(X[n, :, :].T, m_tmp, linalg.inv(C_tmp)))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, m_seq, C_seq
