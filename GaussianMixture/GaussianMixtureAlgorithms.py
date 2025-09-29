import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy import linalg, stats, spatial, optimize
from scipy.special import logsumexp
from particles import resampling as rs
from particles import smc_samplers as ssp
from scipy.spatial.distance import pdist, squareform


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
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    for i in range(1, Niter):
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

def ParallelMALA(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    X[0, :, :] = X0.T
    accepted = np.zeros((Niter, N))
    for i in range(1, Niter):
        gradient = gradient_mixture(X[i-1, :, :], ms, Sigmas, weights)
        prop = X[i-1, :, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        X[i, :, :], accepted[i, :] = mala_accept_reject(prop, X[i-1, :, :], ms, Sigmas, weights, gamma)
    return X, accepted
def mala_accept_reject(prop, v, ms, Sigmas, weights, gamma):
    d = ms[0,:].size
    log_proposal = multivariate_normal.logpdf((v-(prop+gamma*gradient_mixture(prop, ms, Sigmas, weights))).T, np.zeros(d), 2*gamma*np.eye(d))-multivariate_normal.logpdf((prop - (v+gamma*gradient_mixture(v, ms, Sigmas, weights))).T, np.zeros(d), 2*gamma*np.eye(d))
    log_acceptance = logpi_mixture(prop, ms, Sigmas, weights) - logpi_mixture(v, ms, Sigmas, weights) + log_proposal
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output, accepted
### WFR

def SMC_WFR(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
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
        gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
        # reweight
#         H = 2*gamma*np.eye(d) #-- original derivation
        H = (4/(N*(d+2)))**(2/(d+4))*np.diag(np.var(X[n, :, :], axis = 1)) #-- KDE theory
#         squared_distances = pdist(X[n, :, :].T)
#         pairwise_squared_distances = squareform(squared_distances)**2
#         H = np.median(pairwise_squared_distances)/(2*np.log(N))*np.eye(d) # median euristic
#         print(H)
        gaussian_convolution = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, np.zeros(d), H).reshape(N, N)
        weight_denominator = np.mean(gaussian_convolution, axis = 1)
        logW = (1-np.exp(-gamma))*(logpi_mixture(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_WFR_adaptive(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    W = np.zeros((Niter, N))
    delta = 0
    n = -1
    deltas = [delta]
    while ((delta < 1) & (n < Niter)):
        n = n+1
        if (n == 0):
            X[n, :] = X0.T
            W[n, :] = np.ones(N)/N
        else:
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
            # MCMC move
            gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
            # reweight
            gaussian_convolution = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, np.zeros(d), 2*gamma*np.eye(d)).reshape(N, N)
            weight_denominator = np.mean(gaussian_convolution, axis = 1)
            logW = logpi_mixture(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator)
            delta = ssp.next_annealing_epn(0, 0.95, logW)
            logW = delta*logW
            deltas.append(delta)
            if (delta >= 1): delta = 1
            W[n, :] = rs.exp_and_normalise(logW)
    return X, W, deltas

def SMC_WFR_adaptive_v2(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    W = np.zeros((Niter, N))
    delta = 0
    n = -1
    deltas = [delta]
    while ((delta < 1) & (n < Niter)):
        n = n+1
        if (n == 0):
            X[n, :] = X0.T
            W[n, :] = np.ones(N)/N
        else:
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :, :] = X[n-1, :, ancestors].T
            # MCMC move
            gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
            X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.multivariate_normal(np.zeros(d), np.eye(d), size = N).T
            # reweight
            gaussian_convolution = multivariate_normal.pdf(np.kron(X[n, :, :].T, np.ones((N, 1))) - np.tile(gradient_step, N).T, np.zeros(d), 2*gamma*np.eye(d)).reshape(N, N)
            weight_denominator = np.mean(gaussian_convolution, axis = 1)
            logW = logpi_mixture(X[n, :, :], ms, Sigmas, weights)-np.log(weight_denominator)
            delta = next_annealing_epn_kl(delta, 0.01, logW)
            logW = delta*logW
            deltas.append(delta)
            if (delta >= 1): delta = 1
            W[n, :] = rs.exp_and_normalise(logW)
    return X, W, deltas



def next_annealing_epn_kl(epn, beta, lw):
    """Find next annealing exponent by solving ESS(exp(lw)) = alpha * N.

    Parameters
    ----------
    epn: float
        current exponent
    alpha: float in (0, 1)
        defines the ESS target
    lw:  numpy array of shape (N,)
        log-weights
    """
    N = lw.shape[0]
    

#     def f(e):
#         ess = rs.essl(e * lw) if e > 0.0 else N  # avoid 0 x inf issue when e==0
#         return ess - alpha * N
    def f(e):
        klapprox = -e*np.mean(lw)+np.log(np.mean(np.exp(e*lw)))
        return klapprox-beta

    if (f(1.0)*f(0) < 0.0):
        return optimize.brentq(f, 0, 1.0)
    else:
        return 1.0

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

def SMC_ULA(gamma, Niter, ms, Sigmas, weights, X0):
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
        gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
        X[n, :, :] = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        logW = delta*(logpi_mixture(X[n, :, :], ms, Sigmas, weights) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_MALA(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
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
        gradient_step = X[n-1, :, :] + gamma*gradient_mixture(X[n-1, :, :], ms, Sigmas, weights)
        prop = gradient_step + np.sqrt(2*gamma)*np.random.normal(size = (d, N))
        X[n, :, :], accepted[n, :] = mala_accept_reject(prop, X[n-1, :, :], ms, Sigmas, weights, gamma)
        # reweight
        delta = np.exp(-(n-1)*gamma)
        logW = delta*(logpi_mixture(X[n-1, :, :], ms, Sigmas, weights) +0.5*np.sum(X[n-1, :, :]**2, axis = 0)) - delta*np.exp(-gamma)*(logpi_mixture(X[n, :, :], ms, Sigmas, weights) +0.5*np.sum(X[n, :, :]**2, axis = 0))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, accepted

### FR
def SMC_UnitFR(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
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
        X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], ms, Sigmas, weights, l)
        # reweight
        delta = l - lpast
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_mixture(X[n, :, :], ms, Sigmas, weights))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W

def SMC_UnitFR_adaptive(gamma, Niter, ms, Sigmas, weights, X0):
    d = ms[0,:].size
    N = X0.shape[0]
    X = np.zeros((Niter, d, N))
    W = np.zeros((Niter, N))
    l = 0
    n = -1
    lambdas = [l]
    while ((l < 1) & (n < Niter)):
        n = n+1
        if (n == 0):
            X[n, :] = X0.T
            W[n, :] = np.ones(N)/N
        else:
            # MCMC move
            prop = rwm_proposal(X[n-1, :, :].T, W[n-1, :]).T
            X[n, :] = rwm_accept_reject(prop, X[n-1, :, :], ms, Sigmas, weights, l)
        # reweight
        logW = 0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_mixture(X[n, :, :], ms, Sigmas, weights)
        new_l = ssp.next_annealing_epn(l, 0.5, logW)
        delta = new_l - l
        l = new_l
        lambdas.append(l)
        if (l >= 1): l = 1
        logW = delta*(0.5*np.sum(X[n, :, :]**2, axis = 0) + logpi_mixture(X[n, :, :], ms, Sigmas, weights))
        W[n, :] = rs.exp_and_normalise(logW)
    return X, W, lambdas


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
def rwm_accept_reject(prop, v, ms, Sigmas, weights, l):
    log_acceptance = (1-l)*0.5*np.sum(v**2 - prop**2, axis = 0) + l*(logpi_mixture(prop, ms, Sigmas, weights)-logpi_mixture(v, ms, Sigmas, weights))
    accepted = np.log(np.random.uniform(size = v.shape[1])) <= log_acceptance
    output = np.copy(v)
    output[:, accepted] = prop[:, accepted]
    return output
