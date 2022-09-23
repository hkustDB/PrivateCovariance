import numpy as np
import torch
from adaptive.utils import get_gauss_wigner_matrix, get_lap_wigner_matrix, get_gauss_noise_vector, get_lap_noise_vector, SVT, get_bincounts, gaussian_tailbound, laplace_tailbound, wigner_gauss_fnormbound, wigner_lap_fnormbound, wigner_gauss_tailbound, wigner_lap_tailbound, clip


def GaussCov(X, n, d, rho, delta=0.0,r=1.0,b_fleig=False):
    cov = torch.mm(X.t(),X)/n
    W = get_gauss_wigner_matrix(d)
    sens = np.sqrt(2)*r*r/n
    if delta > 0.0:
        sens = sens*np.sqrt(2*np.log(1.25/delta))
        eps = rho+2.*np.sqrt(rho*np.log(1/delta))
        cov_tilde = cov + sens/eps*W
    else:
        cov_tilde = cov + sens/np.sqrt(2*rho)*W
    if b_fleig:
        D, U = torch.linalg.eigh(cov_tilde)
        for i in range(d):
            D[i] = max(min(D[i],r*r),0)
        cov_tilde = torch.mm(U,torch.mm(D.diag_embed(),U.t()))
    return cov_tilde

def GaussApproxCov(X, n, d, eps, delta,r=1.0,b_fleig=False):
    cov = torch.mm(X.t(),X)/n
    W = get_gauss_wigner_matrix(d)
    sens = np.sqrt(2)*r*r/n
    sens = sens*np.sqrt(2*np.log(1.25/delta))
    cov_tilde = cov + sens/eps*W
    if b_fleig:
        D, U = torch.linalg.eigh(cov_tilde)
        for i in range(d):
            D[i] = max(min(D[i],r*r),0)
        cov_tilde = torch.mm(U,torch.mm(D.diag_embed(),U.t()))
    return cov_tilde

def LapCov(X, n, d, eps,r=1.0,b_fleig=False):
    cov = torch.mm(X.t(),X)/n
    W = get_lap_wigner_matrix(d)
    sens = np.sqrt(2)*d*r*r/n
    cov_tilde = cov + sens/eps*W
    if b_fleig:
        D, U = torch.linalg.eigh(cov_tilde)
        for i in range(d):
            D[i] = max(min(D[i],r*r),0)
        cov_tilde = torch.mm(U,torch.mm(D.diag_embed(),U.t()))
    return cov_tilde

def SeparateCov(X, n, d, rho, r=1.0,b_fleig=False):
    cov = torch.mm(X.t(),X)/n
    cov_gauss = GaussCov(X, n, d, 0.5*rho, r=r)
    Ug, Dg, Vg = cov_gauss.svd()
    U, D, V = cov.svd()
    Z = get_gauss_noise_vector(d)
    sens = r*r*np.sqrt(2)/n
    D_tilde = torch.diag(D) + torch.diag(sens/np.sqrt(rho)*Z)
    if b_fleig:
        for i in range(d):
            D_tilde[i,i] = max(min(D_tilde[i,i],r*r),0)
    cov_tilde = torch.mm(Ug,torch.mm(D_tilde,Vg.t()))
    return cov_tilde


def SeparateLapCov(X, n, d, eps, r=1.0,b_fleig=False):
    cov = torch.mm(X.t(),X)/n
    eps0 = 0.5*eps
    cov_lap = LapCov(X, n, d, eps0, r=r)
    Ug, Dg, Vg = cov_lap.svd()
    U, D, V = cov.svd()
    Z = get_lap_noise_vector(d)
    sens = r*r*2./n
    D_tilde = torch.diag(D) + torch.diag(sens/eps0*Z)
    if b_fleig:
        for i in range(d):
            D_tilde[i,i] = max(min(D_tilde[i,i],r*r),0)
    cov_tilde = torch.mm(Ug,torch.mm(D_tilde,Vg.t()))
    return cov_tilde


def get_bias(counts, tup): 
    (parts, n, k, k1) = tup
    i = k1-k+1
    bias = sum([counts[l]*parts[l] for l in range(i)])
    bias = (bias - sum(counts[:i])*2**(2*k))/n  
    return bias
   
def get_diff(counts, tup):
    (parts, n, k, noise1, noise2, k1, r) = tup
    i = k1-k+1
    bias = sum([counts[l]*parts[l] for l in range(i)])
    bias = (bias - sum(counts[:i])*2**(2*k))/n
    noise = 2**k*noise1 + 2**(2*k)*noise2
    diff = n*(bias - noise)/r/r
    return diff

def get_diff2(counts, tup):
    (parts, n, k, gaussnoise, sepnoise1, sepnoise2, k1, r) = tup
    i = k1-k+1
    bias = sum([counts[l]*parts[l] for l in range(i)])
    bias = (bias - sum(counts[:i])*2**(2*k))/n
    noise = min(2**k*gaussnoise, 2**k*sepnoise1 + 2**(2*k)*sepnoise2)
    diff = n*(bias - noise)/r/r
    return diff            

def ClippedCov(X, n, d, rho, beta, tr_tilde, r=1.0):
    eta = gaussian_tailbound(d,0.5*beta)
    nu = wigner_gauss_tailbound(d,0.5*beta)
    rho1 = 0.75*rho
    noise1 = 2**(1.25)*np.sqrt(tr_tilde)/np.sqrt(n)/(rho1**0.25)*np.sqrt(nu)
    noise2 = np.sqrt(2)*eta/np.sqrt(rho1)/n
    k1 = int(np.log2(r))
    k3 = -max(int(np.ceil(np.log2(noise1+noise2)-np.log2(r*r/np.sqrt(rho1)/n*np.log(1./beta))))+1,k1)
    t = k1-k3+1
    x_norm = torch.linalg.norm(X, dim=1, keepdim=True)
    parts = [2**(2*(k1-l)+2) for l in range(t)]
    counts = get_bincounts(x_norm, n, t)
    args = [(parts,n,k1-l,noise1,noise2,k1,r) for l in range(t)]
    j = SVT(0,np.sqrt(rho/2.),counts,get_diff,args)
    k2_tilde = k1-j#k1+1-j
    if k2_tilde < k1:
        r_tilde =2**(k2_tilde+1)
    else:
        r_tilde = 2**k2_tilde
    X_tilde = clip(X,x_norm,r_tilde)
    Sigma = SeparateCov(X_tilde,n,d,rho1,r=r_tilde,b_fleig=True)
    return Sigma

def ClippedLapCov(X, n, d, eps, beta, tr_tilde, r=1.0):
    eta = laplace_tailbound(d,0.5*beta)
    nu = wigner_lap_tailbound(d,0.5*beta)
    eps1 = 0.75*eps
    eps2 = 0.25*eps
    noise1 = np.sqrt(8*np.sqrt(2)*d*tr_tilde)/np.sqrt(eps1*n)*np.sqrt(nu)
    noise2 = 4*eta/eps1/n
    k1 = int(np.log2(r))
    k3 = -max(int(np.ceil(np.log2(noise1+noise2)-np.log2(r*r/eps1/n*np.log(1./beta))))+1,-k1)
    t = k1-k3+1
    x_norm = torch.linalg.norm(X, dim=1, keepdim=True)
    parts = [2**(2*(k1-l)+2) for l in range(t)]
    counts = get_bincounts(x_norm, n, t)
    args = [(parts,n,k1-l,noise1,noise2,k1,r) for l in range(t)]
    j = SVT(0,eps2,counts,get_diff,args)
    k2_tilde = k1-j
    if k2_tilde < k1:
        r_tilde = 2**(k2_tilde+1)
    else:
        r_tilde = 2**k2_tilde
    X_tilde = clip(X,x_norm,r_tilde)
    Sigma = SeparateLapCov(X_tilde,n,d,eps1,r=r_tilde,b_fleig=True)
    return Sigma


def AdaptiveCov(X, args,r=1.0):
    rho = args.total_budget
    n = args.n
    d = args.d
    beta = args.beta
    cov = torch.mm(X.t(),X)/n
    tr = torch.trace(cov)
    factor = np.sqrt(8./rho*np.log(8./beta))
    tr_tilde = tr + r*r*(2./np.sqrt(rho)/n*np.random.normal(0,1) + factor/n)
    tr_tilde = min(tr_tilde, r)
    tr_tilde = max(tr_tilde, 1e-16)
    rho1 = 0.75*rho
    beta1 = 0.75*beta
    eta = gaussian_tailbound(d,0.5*beta1)
    nu = wigner_gauss_tailbound(d,0.5*beta1)
    omega = wigner_gauss_fnormbound(d,beta1)
    
    sepnoise1 = 2**(1.25)*np.sqrt(tr_tilde)/np.sqrt(n)/(rho1**0.25)*np.sqrt(nu)/6.
    sepnoise2 = np.sqrt(2)*eta/np.sqrt(rho1)/n
    gaussnoise1 = 1./np.sqrt(rho1)/n*omega
    k1 = int(np.log2(r))
    k3 = -min(int(d*n),-int(np.log2(1e-24)))
    t = k1-k3+1
    x_norm = torch.linalg.norm(X, dim=1, keepdim=True)
    parts = [2**(2*(k1-l)+2) for l in range(t)]
    counts = get_bincounts(x_norm, n, t)
    args = [(parts,n,k1-l,gaussnoise1,sepnoise1,sepnoise2,k1,r) for l in range(t)]
    j = SVT(0,np.sqrt(rho)/2.,counts,get_diff2,args)
    k2_tilde = k1-j
    r_tilde = min(2**(k2_tilde+1),r)
    X_tilde = clip(X,x_norm,r_tilde)
    sepnoise = (2**k2_tilde)*sepnoise1+(2**(2*k2_tilde))*sepnoise2
    gaussnoise = (2**(2*k2_tilde))*gaussnoise1
    
    if sepnoise>=gaussnoise:
        Sigma = GaussCov(X_tilde,n,d,rho1,r=r_tilde,b_fleig=True)
    else:
        Sigma = SeparateCov(X_tilde,n,d,rho1,r=r_tilde,b_fleig=True)
    return Sigma


def AdaptiveLapCov(X, args,r=1.0):
    rho = args.total_budget
    eps = np.sqrt(2*rho)
    n = args.n
    d = args.d
    beta = args.beta
    cov = torch.mm(X.t(),X)/n
    tr = torch.trace(cov)
    factor = 8.*r*r/eps*np.log(4./beta)
    tr_tilde = tr + 8.*r*r/eps/n*np.random.laplace(0,1) + factor/n
    tr_tilde = min(tr_tilde, r)
    tr_tilde = max(tr_tilde, 1e-16)
    eps1 = 0.75*eps
    beta1 = 0.75*beta
    eta = laplace_tailbound(d,0.5*beta1)
    nu = wigner_lap_tailbound(d,0.5*beta1)
    omega = wigner_lap_fnormbound(d,beta1)
    
    sepnoise1 = np.sqrt(8*np.sqrt(2)*d*tr_tilde)/np.sqrt(eps1*n)*np.sqrt(nu)/14.
    sepnoise2 = 4*eta/eps1/n
    lapnoise1 = np.sqrt(2)*d*r*r/n*omega
    k1 = int(np.log2(r))
    k3 = -min(int(d*n),-int(np.log2(1e-24)))
    t = k1-k3+1
    x_norm = torch.linalg.norm(X, dim=1, keepdim=True)
    parts = [2**(2*(k1-l)+2) for l in range(t)]
    counts = get_bincounts(x_norm, n, t)
    args = [(parts,n,k1-l,lapnoise1,sepnoise1,sepnoise2,k1,r) for l in range(t)]
    j = SVT(0,eps/8.,counts,get_diff2,args)
    k2_tilde = k1-j
    r_tilde = min(2**(k2_tilde+1),r)
    X_tilde = clip(X,x_norm,r_tilde)
    sepnoise = (2**k2_tilde)*sepnoise1+(2**(2*k2_tilde))*sepnoise2
    lapnoise = (2**(2*k2_tilde))*lapnoise1

    if sepnoise>=lapnoise:
        Sigma = LapCov(X_tilde,n,d,eps1,r=r_tilde,b_fleig=True)
    else:
        Sigma = SeparateLapCov(X_tilde,n,d,eps1,r=r_tilde,b_fleig=True)
    return Sigma