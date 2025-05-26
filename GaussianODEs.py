import numpy as np

def KL(mu, sigma, mu_target, sigma_target):
    return 0.5*np.log(sigma_target/sigma)+(sigma+(mu-mu_target)**2)/(2*sigma_target) - 0.5

### Infinite time

def wasserstein_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    for i in range(tsteps):
        mean_eq[i+1] = mean_eq[i] - gamma*(mean_eq[i]-mu)/sigma
        var_eq[i+1] = var_eq[i] + gamma*(-2*var_eq[i]/sigma+2)
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo

def tempered_wasserstein_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma, lseq):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    for i in range(tsteps):
        l = lseq[i]
        mean_eq[i+1] = mean_eq[i] - gamma*((1-l)*(mean_eq[i]-mu0)/sigma0 + l*(mean_eq[i]-mu)/sigma)
        var_eq[i+1] = var_eq[i] + 2*gamma*(1 - (1-l)*var_eq[i]/sigma0 - l*var_eq[i]/sigma)
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo

def fisherrao_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    A = 1/sigma - 1/sigma0  
    for i in range(tsteps):
        mean_eq[i+1] = mean_eq[i] - gamma*(mean_eq[i]-mu)*var_eq[i]/sigma
        var_eq[i+1] = var_eq[i] + gamma*(-var_eq[i]**2/sigma+var_eq[i])
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo

def tempered_fisherrao_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma, lseq):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    A = 1/sigma - 1/sigma0  
    for i in range(tsteps):
        l = lseq[i]
        mean_eq[i+1] = mean_eq[i] - gamma*(mean_eq[i]-mu)*var_eq[i]*(l/sigma+(1-l)/sigma0)
        var_eq[i+1] = var_eq[i] + gamma*(-var_eq[i]**2*(l/sigma+(1-l)/sigma0)+var_eq[i])
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo

def wfr_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    for i in range(tsteps):
        mean_eq[i+1] = mean_eq[i] - gamma*(mean_eq[i]-mu)*(var_eq[i]+1)/sigma
        var_eq[i+1] = var_eq[i] + gamma*(-var_eq[i]**2/sigma+var_eq[i]-2*var_eq[i]/sigma+2)
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo

def tempered_wfr_infinite_time(mu0, mu, sigma0, sigma, tsteps, gamma, lseq):
    mean_eq = np.zeros(tsteps+1)
    var_eq = np.zeros(tsteps+1)
    kl_evo = np.zeros(tsteps+1)
    mean_eq[0] = mu0
    var_eq[0] = sigma0
    kl_evo[0] = KL(mu0, sigma0, mu, sigma)
    for i in range(tsteps):
        l = lseq[i]
        mean_eq[i+1] = mean_eq[i] - gamma*(mean_eq[i]-mu)*var_eq[i]*(l/sigma+(1-l)/sigma0) - gamma*((1-l)*(mean_eq[i]-mu0)/sigma0 + l*(mean_eq[i]-mu)/sigma)
        var_eq[i+1] = var_eq[i] + gamma*(-var_eq[i]**2*(l/sigma+(1-l)/sigma0)+var_eq[i]) + 2*gamma*(1 - (1-l)*var_eq[i]/sigma0 - l*var_eq[i]/sigma)
        kl_evo[i+1] = KL(mean_eq[i+1], var_eq[i+1], mu, sigma)
    return mean_eq, var_eq, kl_evo