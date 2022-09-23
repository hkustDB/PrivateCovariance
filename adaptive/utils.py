import torch
import numpy as np
from scipy.optimize import root_scalar


def SVT(T,eps,D,func,args):
    T_tilde = T + np.random.laplace(scale=2.0/eps)
    i = 0
    m = len(args)
    while i < m:
        Qi = func(D,args[i]) + np.random.laplace(scale=4.0/eps)
        if Qi >= T_tilde:
            break
        i = i + 1
    return i

def convert_symm_mat(ZZ,d):
    S = torch.empty([d,d])
    k = 0
    for i in range(d):
        for j in range(i,d):
            S[i,j] = ZZ[0,k]
            k = k+1

    for i in range(d):
        for j in range(i+1,d):
            S[j,i] = S[i,j]
    return S

def get_gauss_wigner_matrix(d):
    Z = torch.normal(0,1,size=(1,int(d*(d+1)/2)))
    W = convert_symm_mat(Z,d)
    return W

def get_lap_wigner_matrix(d):
    Z = torch.distributions.laplace.Laplace(0,1).sample((1,int(d*(d+1)/2)))
    W = convert_symm_mat(Z,d)
    return W

def get_gauss_noise_vector(d):
    Z = torch.normal(0,1,size=(1,d)).squeeze()#np.random.normal(0,1,d)
    return Z

def get_lap_noise_vector(d):
    Z = torch.distributions.laplace.Laplace(0,1).sample((d,))
    return Z
    
def inv_sqrt(S):
    S_inv = torch.inverse(S)
    U, D, V = S_inv.svd()
    S_inv_sqrt = torch.mm(U,torch.mm(D.sqrt().diag_embed(),V.t()))
    return S_inv_sqrt


# def get_bincounts(x_norm,n,t,k1=0):
#     k3 = k1+1-t
#     counts = np.zeros(t)
#     counts_sum = [0]
#     counts_sum.extend([int(sum(x_norm>2**k)) for k in range(k1-1,k3-1,-1)])
#     counts[0] = 0
#     for k in range(k1-1,k3-1,-1):
#         j = -k-k1
#         counts[j] = counts_sum[j] - counts_sum[j-1]
#     return counts

def get_bincounts(x_norm,n,t,k1=0):
    counts = np.zeros(t)
    #counts[0] = 0
    for i in range(n):
        l1 = -int(np.floor(np.log2(x_norm[i])))
        if not(x_norm[i] > 2**(-l1)):
            l1 = l1+1
        l1 = max(l1,1)
        if l1 < t:
            counts[l1] = counts[l1] + 1
    return counts


def gaussian_tailbound(d,b):
    bound = np.sqrt(d+2*np.sqrt(d*np.log(1/b))+2*np.log(1/b))
    return bound

def laplace_tailbound(d,b):
    K = 2*max((np.sqrt(2)+1)/(np.sqrt(2)-1),2*np.log(d)/np.log(2))
    bound = 1.5*np.sqrt(d)+np.log(2./b)*K
    return bound

def wigner_gauss_tailbound(d,b):
    theta = np.log(d)/d
    theta = min(theta**0.3, 0.5)
    bound = (1+theta)*(2*np.sqrt(d)+6*np.sqrt(np.log(d))/np.sqrt(np.log(1+theta)))
    bound = bound + np.sqrt(2*np.log(2/b))
    return bound
            
def wigner_lap_tailbound(d,b):
    K = 2*max((np.sqrt(2)+1)/(np.sqrt(2)-1),2*np.log(0.5*(d*d+d))/np.log(2))
    theta = np.log(d)/d
    theta = min(theta**0.3, 0.5)
    bound = (1+theta)*(2*np.sqrt(d)+6*np.log(d)/np.sqrt(np.log(1+theta)))
    bound = bound + np.sqrt(2)*np.log(2/b)*K
    return bound
    
def wigner_gauss_fnormbound(d,b):
    bound = np.sqrt(d*d+2.*np.sqrt(d*np.log(2./b))*(1+np.sqrt(2*(d-1)))+6*np.log(2./b))
    return bound

def wigner_lap_fnormbound(d,b):
    K = 2*max((np.sqrt(2)+1)/(np.sqrt(2)-1),2*np.log(0.5*(d*d+d))/np.log(2))
    bound = 1.5*d + np.log(2./b)*K
    return bound
            
def clip(X, x_norm,r):
    scale = r / x_norm
    clipped = (scale < 1).squeeze()
    W = X.detach().clone()
    W[clipped] = W[clipped] * scale[clipped]
    return W

def rho_eps_eq(x,eps0,delta):
    f = x+2*np.sqrt(x*np.log(1./delta))
    return f-eps0

def get_rho(eps,delta):
    rho = root_scalar(rho_eps_eq, args=(eps,delta),bracket=[0,eps]).root
    return rho