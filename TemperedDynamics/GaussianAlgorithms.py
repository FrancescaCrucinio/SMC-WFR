import numpy as np
from scipy.stats import norm
from scipy import linalg, stats, spatial
from particles import resampling as rs


def FR_reweigthing(mu, sigma, X, gamma, delta, gradient_step, W):
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