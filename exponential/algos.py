import torch
import numpy as np
from exponential.utils import find_bingham, convert_eps
from scipy.linalg import null_space

def EMCov(X, args, b_budget=False, b_fleig=True):
    rho = args.total_budget
    delta = args.delta
    n = args.n
    d = args.d
    cov = torch.mm(X.t(),X)
    if not(delta > 0.0):
        eps_total = np.sqrt(2*rho)
    else:   
        eps_total = rho+2.*np.sqrt(rho*np.log(1/delta))
    eps0 = 0.5*eps_total
    Uc, Dc, Vc = cov.svd()
    lap = torch.distributions.laplace.Laplace(0,2./eps0).sample((d,))
    Lamb_hat = torch.diag(Dc) + torch.diag(lap)
    Lamb_round = torch.zeros(d)
    if b_fleig:
        for i in range(d):
            lamb = max(min(Lamb_hat[i,i],n),0)
            Lamb_round[i] = lamb
    else:
        Lamb_round = torch.diag(Lamb_hat)
    P1 = torch.eye(d)
    if not(b_budget):
        if (delta > 0):
            ep = convert_eps(eps0,d,delta)
        else:
            ep = eps0/d
        eps = [ep for j in range(d)]
    else:
        tau = 2./eps0*np.log(2.*d/args.beta)
        numer = [np.sqrt(Lamb_round[j]+tau) for j in range(d)]
        denom = sum(numer)
        eps = [eps0*numer[j]/denom for j in range(d)]
    Ci = cov
    Pi = torch.eye(d)
    theta = torch.zeros(d,d)
    for i in range(d):
        Ci, Pi = EMStep(cov, Ci, Pi, eps[i], d, i, theta)
    C_hat = torch.zeros(d,d)
    for i in range(d):
        C_hat = C_hat + Lamb_round[i]*torch.outer(theta[i],theta[i])
    return C_hat/n
    
    
def EMStep(C, Ci, Pi, epi, d, i, theta):
    u_hat = find_bingham(Ci, epi, (d-i), int(np.sqrt(d)))
    theta_hat = torch.matmul(Pi,u_hat)
    theta[i] = theta_hat
    if not(i==d-1):
        Pi = torch.from_numpy(null_space(theta))
        Ci = torch.matmul(torch.matmul(Pi.t(),C),Pi)
    return Ci, Pi


    
        
        
    