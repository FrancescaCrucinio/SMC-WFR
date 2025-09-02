import numpy as np
from scipy.stats import norm
from scipy import linalg, stats, spatial
from particles import resampling as rs


def FR_reweigthing(mu, sigma, X, gamma, delta, gradient_step):
    weight_denominator = np.mean(norm.pdf(X - gradient_step.reshape(-1,1), loc = 0, scale = np.sqrt(2*gamma)), axis = 1)
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

def SMC_ULA_reweighting(mu, sigma, X, Xold, gradient_step, delta, gamma):
    gradient_step_reverse = X - gamma*(X - mu)/sigma
    proposal_ratio = (X - gradient_step)**2 - (Xold - gradient_step_reverse)**2
#     
#     
#     norm.pdf(Xold - gradient_step_reverse, loc = 0, scale = np.sqrt(2*gamma))/norm.pdf(X - gradient_step, loc = 0, scale = np.sqrt(2*gamma))
#     print(np.exp(proposal_ratio)**(1/(2*gamma)))
    logW = delta*(-0.5*(X-mu)**2/sigma+0.5*X**2) + proposal_ratio
    return rs.exp_and_normalise(logW)

def SMC_WFR(gamma, Niter, mu, sigma, mu0, sigma0, X0):
    N = X0.size
    X = np.zeros((Niter+1, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
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

def SMC_ULA(gamma, Niter, mu, sigma, mu0, sigma0, X0):
    N = X0.size
    X = np.zeros((Niter+1, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :] = X[n-1, ancestors]
        # MCMC move
        noise = np.random.normal(size = N)
#         Xold = np.copy(X[n, :])
        X[n, :], gradient_step = W_move(mu, sigma, X[n-1, :], gamma, noise)
        # reweight
        delta = (1-np.exp(-gamma))*np.exp(-(n-1)*gamma)
        W[n, :] = SMC_ULA_reweighting(mu, sigma, X[n, :], X[n-1, :], gradient_step, delta, gamma)
#         W[n, :] = np.ones(N)/N
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

def SMC_UnitFR(Niter, mu, sigma, mu0, sigma0, X0):
    N = X0.size
    X = np.zeros((Niter+1, N))
    W = np.zeros((Niter+1, N))
    X[0, :] = X0
    W[0, :] = np.ones(N)/N
    for n in range(1, Niter+1):
        l = n/Niter
        if (n > 1):
            # resample
            ancestors = rs.resampling('stratified', W[n-1, :])
            X[n-1, :] = X[n-1, ancestors]
        # MCMC move
        prop = rwm_proposal(X[n-1, :], W[n-1, :])
        X[n, :] = rwm_accept_reject(prop, X[n-1, :], mu, sigma, mu0, sigma0, (n-1)/Niter)
        # reweight
        delta = n/Niter - (n-1)/Niter
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

### Birth Death Langevin
def BDL_kernelisedPDE_gaussian1D(gamma, Niter, mu, sigma, N, h, X0):
    X = np.zeros((Niter, N))
    X[0,:] = X0
    for i in range(1, Niter):
        gradient = -(X[i-1, :] - mu)/sigma
        X[i, :] = X[i-1, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.normal(size = N)
        beta = -norm.logpdf(X[i, :], loc = mu, scale = np.sqrt(sigma))
        kernel_rate = norm.pdf(X[i, :] - X[i, :].reshape(-1,1), loc = 0, scale = h)
        beta = beta + np.log(np.mean(kernel_rate, axis = 0))
        beta = beta - np.mean(beta)
        kill = (beta >= 0) * (np.random.uniform(size = N) < 1-np.exp(-beta*gamma))
        duplicate = (beta < 0) * (np.random.uniform(size = N) < 1-np.exp(beta*gamma))
        Xnew = np.concatenate((X[i, duplicate], X[i, duplicate]))
        Nres = Xnew.shape[0]
        if(Nres > N):
            tokeep = np.random.randint(Nres, size = N)
            X[i, :] = Xnew[tokeep]
        else:
            toduplicate = np.random.randint(N, size = N-Nres)
            X[i, :] = np.concatenate((Xnew, X[i, toduplicate]))
    return X
def BDL_kernelisedKL_gaussian1D(gamma, Niter, mu, sigma, N, h, X0):
    X = np.zeros((Niter, N))
    X[0,:] = X0
    for i in range(1, Niter):
        gradient = -(X[i-1, :] - mu)/sigma
        X[i, :] = X[i-1, :] + gamma*gradient + np.sqrt(2*gamma)*np.random.normal(size = N)
        beta = -norm.logpdf(X[i, :], loc = mu, scale = np.sqrt(sigma))
        kernel_rate = norm.pdf(X[i, :] - X[i, :].reshape(-1,1), loc = 0, scale = h)
        beta = beta + np.log(np.mean(kernel_rate, axis = 0))
        beta = beta - np.mean(beta) - 1 + np.sum(kernel_rate/np.sum(kernel_rate, axis = 0), axis = 1)
        kill = (beta >= 0) * (np.random.uniform(size = N) < 1-np.exp(-beta*gamma))
        duplicate = (beta < 0) * (np.random.uniform(size = N) < 1-np.exp(beta*gamma))
        Xnew = np.concatenate((X[i, duplicate], X[i, duplicate]))
        Nres = Xnew.shape[0]
        if(Nres > N):
            tokeep = np.random.randint(Nres, size = N)
            X[i, :] = Xnew[tokeep]
        else:
            toduplicate = np.random.randint(N, size = N-Nres)
            X[i, :] = np.concatenate((Xnew, X[i, toduplicate]))
    return X