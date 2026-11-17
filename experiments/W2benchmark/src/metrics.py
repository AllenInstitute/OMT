import torch
import numpy as np
from scipy.linalg import sqrtm, inv
from .tools import freeze
from scipy.stats import wasserstein_distance

import gc

def score_fitted_maps(benchmark, D, D_conj, size=4096):
    '''Estimates L2-UVP and cosine metrics for transport map by the gradients of potentials.'''
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True); Y.requires_grad_(True)
    
    freeze(D); freeze(D_conj)
    X_push = D.push_nograd(X)
    Y_inv = D_conj.push_nograd(Y)
    
    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
        L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()
        
        cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()
        
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv

def score_baseline_maps(benchmark, baseline='linear', size=4096):
    '''Estimates L2-UVP and cosine similarity metrics for the baseline transport map (identity, const or linear).'''
    assert baseline in ['identity', 'linear', 'constant']
    
    X = benchmark.input_sampler.sample(size); X.requires_grad_(True)
    Y = benchmark.map_fwd(X, nograd=True)
    
    with torch.no_grad():
        if baseline == 'linear':  
            X_push = benchmark.linear_map_fwd(X)
            Y_inv = benchmark.linear_map_inv(Y)
        elif baseline == 'constant':
            X_push = torch.tensor(
                benchmark.output_sampler.mean.reshape(1, -1).repeat(size, 0),
                dtype=torch.float32
            ).to(X)
            Y_inv = torch.tensor(
                benchmark.input_sampler.mean.reshape(1, -1).repeat(size, 0),
                dtype=torch.float32
            ).to(Y)
        elif baseline == 'identity':
            X_push = X
            Y_inv = Y

        if baseline == 'constant':
            L2_UVP_fwd, L2_UVP_inv = 100., 100.
        else:
            L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
            L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()

        if baseline == 'identity':
            cos_fwd, cos_inv = 0., 0.
        else:
            cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
            (np.sqrt(2 * benchmark.cost * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
            cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
            (np.sqrt(2 * benchmark.cost * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()

    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv

def metrics_to_dict(L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv):
    return dict(L2_UVP_fwd=L2_UVP_fwd, cos_fwd=cos_fwd, L2_UVP_inv=L2_UVP_inv, cos_inv=cos_inv)



def omt_score_fitted_maps(benchmark, X, Y, X_push, Y_inv):
    
    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
        L2_UVP_inv = 100 * (((X - Y_inv) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()
        
        cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
        (np.sqrt((2 * benchmark.cost) * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()
        
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv



def L2_UVP_cost(benchmark, X, Y, X_push, direction='fwd'):
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32).cuda(benchmark.device)
    if not isinstance(X_push, torch.Tensor):
        X_push = torch.tensor(X_push, dtype=torch.float32).cuda(benchmark.device)
    
    with torch.no_grad():
        if direction == 'fwd':
            L2_UVP = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
            cos = (((Y - X) * (X_push - X)).sum(dim=1).mean() / (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        elif direction == 'inv':
            L2_UVP = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.input_sampler.var).item()
            cos = (((Y - X) * (X_push - X)).sum(dim=1).mean() / (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
        else:
            raise ValueError("Invalid direction. Choose 'fwd' or 'inv'.")

    gc.collect(); torch.cuda.empty_cache()
    return L2_UVP, cos


def L2_UVP_omt_cost(omt_solver, Y, X_push):

    comp_y = omt_solver['gmm_t'].predict(Y)
    mu_T = omt_solver['gmm_t'].means_
    cov_T = omt_solver['gmm_t'].covariances_

    L2_UVP = []
    for cmp in np.unique(comp_y):
        indx = np.where(comp_y == cmp)[0]
        L2_UVP.append(compute_BW_metrics(X_push[indx], mu_T[cmp], cov_T[cmp]))
        print(f"Component {cmp}: BW_UVP = {L2_UVP[-1]:.4f}")
    return np.mean(L2_UVP)

    
    # with torch.no_grad():
    #     L2_UVP_fwd = 100 * (((Y.mean(dim=0) - X_push.mean(dim=0)) ** 2).sum() / benchmark.output_sampler.var).item()
    #     L2_UVP_inv = 100 * (((X.mean(dim=0) - Y_pull.mean(dim=0)) ** 2).sum() / benchmark.input_sampler.var).item()
        
    #     cos_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
    #     (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
    #     cos_inv = (((X - Y) * (Y_pull - Y)).sum(dim=1).mean() / \
    #     (np.sqrt((2 * benchmark.cost) * ((Y_pull - Y) ** 2).sum(dim=1).mean().item()))).item()
        
    # gc.collect(); torch.cuda.empty_cache() 
    # return L2_UVP_fwd, cos_fwd, L2_UVP_inv, cos_inv


def score_fitted_fwdmaps(benchmark, X, Y, X_push, size=4096):
    
    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y.mean(dim=0) - X_push.mean(dim=0)) ** 2).sum() / benchmark.output_sampler.var).item()
        L2_UVP_fwd2 = 100 * (((Y - X_push) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
        
        cost_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / \
                (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
       
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd, cost_fwd, L2_UVP_fwd2, Y


def compute_cost(benchmark, X, Y, X_push):
    
    # cost_fwd = (((Y - X) * (X_push - X)).sum(dim=1).mean() / (np.sqrt((2 * benchmark.cost) * ((X_push - X) ** 2).sum(dim=1).mean().item()))).item()
    # cos_inv = (((X - Y) * (Y_inv - Y)).sum(dim=1).mean() / \
    # (np.sqrt((2 * benchmark.cost) * ((Y_inv - Y) ** 2).sum(dim=1).mean().item()))).item()
    
    cost_fwd = (X_push - X).sum(dim=1).mean().item() #/ ((X_push - X) ** 2).sum(dim=1).mean().item()
        
    gc.collect(); torch.cuda.empty_cache() 
    return cost_fwd #, cos_inv


def symmetrize(X):
    return np.real((X + X.T) / 2)


def compute_BW_UVP(samples, true_mu, true_covariance):
    samples_covariance = np.cov(samples.T)
    samples_mu = samples.mean(axis=0)
    samples_covariance_sqrt = symmetrize(sqrtm(samples_covariance))

    mu_term = 0.5*((true_mu - samples_mu)**2).sum()
    covariance_term = (
        0.5*np.trace(samples_covariance) + 
        0.5*np.trace(true_covariance) -
        np.trace(symmetrize(sqrtm(samples_covariance_sqrt@true_covariance@samples_covariance_sqrt)))
    )

    BW = mu_term + covariance_term
    BW_UVP = 100*(BW/(0.5*np.trace(true_covariance)))
        
    return BW_UVP


def compute_BW_metrics(T_X, mu_t, cov_t):

    T_X_covariance = np.cov(T_X.T)
    T_X_mu = T_X.mean(axis=0)
    
    true_mu = mu_t
    true_covariance = cov_t

    T_X_covariance_sqrt = symmetrize(sqrtm(T_X_covariance))
    
    mu_term = 0.5*((true_mu - T_X_mu)**2).sum()
    covariance_term = (
                        0.5*np.trace(T_X_covariance) + 
                        0.5*np.trace(true_covariance) -
                        np.trace(symmetrize(sqrtm(T_X_covariance_sqrt@true_covariance@T_X_covariance_sqrt)))
                        )
    
    BW = mu_term + covariance_term
    BW_UVP = 100*(BW/(0.5*np.trace(true_covariance)))
    
        
    return BW_UVP



def compute_w2_divergence(u_values, v_values, u_weights=None, v_weights=None):
    """
    Computes the Wasserstein-2 (W2) divergence between two 1D discrete distributions.

    Parameters:
    u_values (1D array_like): A sample from or the support of the first distribution.
    v_values (1D array_like): A sample from or the support of the second distribution.
    u_weights (1D array_like, optional): Weights or counts for u_values.
    v_weights (1D array_like, optional): Weights or counts for v_values.

    Returns:
    float: The Wasserstein-2 distance.
    """
    w1_distance = wasserstein_distance(u_values, v_values, u_weights, v_weights)
    return w1_distance**2

