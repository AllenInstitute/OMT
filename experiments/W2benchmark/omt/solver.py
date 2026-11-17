from shlex import split
from posixpath import split
import numpy as np
import cvxpy as cvx
import ot 
import time
import pdb
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.linalg import cholesky
import itertools
from sklearn.ensemble import IsolationForest

from .functions import eogt, vectorized_eogt_dist


#@profile
def omt(
        data_source, 
        data_target, 
        n_components=(100, 100), 
        eps_gmm=0.1, 
        eps_w=0.0, 
        method='cvx', 
        cov_type='full', 
        reg_covar=1e-6, 
        max_iter=10000, 
        stop_thr=1e-6, 
        random_seed=None, 
        verbose=False, 
        vectorize=False,
        combine=False,
        check_outlier=True,
        ):
    
    smp_scale = 100
    dim = data_source.shape[1]
    Ks, Kt = n_components

    if Ks > data_source.shape[0] or Kt > data_target.shape[0]:
        raise ValueError("Number of components cannot exceed number of samples!")
    if eps_gmm < 0:
        raise ValueError("eps_gmm must be non-negative!")
    if eps_w < 0:
        raise ValueError("eps_w must be non-negative!")
    if method not in ['cvx', 'sinkhorn', 'w2', 'cvx']:
        raise ValueError("Method not recognized. Choose 'cvx', 'sinkhorn', or 'w2'.")
    
    if cov_type == 'diag':
        source_param = Ks * (2*dim) + (1 - Ks)
        target_param = Kt * (2*dim) + (1 - Kt)
    else:
        source_param = Ks * (dim**2/2 + 1.5*dim) + (1 - Ks)
        target_param = Kt * (dim**2/2 + 1.5*dim) + (1 - Kt)
    
    # min_num_smp = int(np.max([source_param, target_param]) * smp_scale)
    # if n_samples < min_num_smp:
    #     n_samples = min(min_num_smp, max_n_samples)

    t0 = time.time()
    
    if check_outlier:
        Isofor = IsolationForest(max_samples=10000, n_estimators=1000, contamination=0.1)
        Isofor.fit(data_source)
        scores = Isofor.predict(data_source)
        data_source = data_source[scores == 1]
        Isofor.fit(data_target)
        scores = Isofor.predict(data_target)
        data_target = data_target[scores == 1]

    gmm_s = shallow_gmm(
                        data_source, 
                        n_components[0], 
                        covariance_type=cov_type, 
                        reg_covar=reg_covar, 
                        max_iter=max_iter, 
                        random_state=random_seed, 
                        verbose=verbose, 
                        combine=combine,
                        )
    gmm_t = shallow_gmm(
                        data_target, 
                        n_components[1], 
                        covariance_type=cov_type, 
                        reg_covar=reg_covar, 
                        max_iter=max_iter, 
                        random_state=random_seed, 
                        verbose=verbose, 
                        combine=combine,
                        )
    t1 = time.time()
    mean_s = gmm_s.means_
    mean_t = gmm_t.means_
    if cov_type == 'diag':
        gmm_s.covariances_ = np.array([np.diag(gmm_s.covariances_[i]) for i in range(len(gmm_s.covariances_))])
        gmm_t.covariances_ = np.array([np.diag(gmm_t.covariances_[i]) for i in range(len(gmm_t.covariances_))])
   
    sigma_s = gmm_s.covariances_
    sigma_t = gmm_t.covariances_
        
    alpha_s = gmm_s.weights_
    alpha_t = gmm_t.weights_
    if verbose:
        print("Number of components in source GMM:", len(alpha_s))
        print("Number of components in target GMM:", len(alpha_t))
        print("Time taken for GMM fitting:", t1 - t0, "seconds")
        
    if vectorize:
        omt_w, omt_mu, omt_sigma, solver_dict = gmmomt_vectorized(
                                                        mean_s=mean_s,
                                                        mean_t=mean_t,
                                                        sigma_s=sigma_s,
                                                        sigma_t=sigma_t,
                                                        alpha_s=alpha_s,
                                                        alpha_t=alpha_t,
                                                        eps_gmm=eps_gmm,
                                                        eps_w=eps_w, 
                                                        method=method, 
                                                        max_iter=max_iter, 
                                                        stop_thr=stop_thr,
                                                        verbose=verbose,
                                                        )
    else:
        omt_w, omt_mu, omt_sigma, solver_dict = gmmomt(
                                                        mean_s=mean_s,
                                                        mean_t=mean_t,
                                                        sigma_s=sigma_s,
                                                        sigma_t=sigma_t,
                                                        alpha_s=alpha_s,
                                                        alpha_t=alpha_t,
                                                        eps_gmm=eps_gmm,
                                                        eps_w=eps_w, 
                                                        method=method, 
                                                        max_iter=max_iter, 
                                                        stop_thr=stop_thr,
                                                        verbose=verbose,
                                                        )
    solver_dict['gmm_s'] = gmm_s
    solver_dict['gmm_t'] = gmm_t
    solver_dict['source_suprng'] = (np.min(data_source, axis=0), np.max(data_source, axis=0))
    solver_dict['target_suprng'] = (np.min(data_target, axis=0), np.max(data_target, axis=0))
    solver_dict['gmm_time_taken'] = t1 - t0
    solver_dict['n_total_iter'] = gmm_s.n_iter_ + gmm_t.n_iter_ + solver_dict['n_iter']
    solver_dict['time'] = solver_dict['omt_time_taken'] + solver_dict['gmm_time_taken']

    if verbose:
        print("Number of total iterations:", solver_dict['n_total_iter'])
        print("Total time taken:", solver_dict['time'], "seconds")
        
    
    return omt_w, omt_mu, omt_sigma, solver_dict
    

def gmmomt(mean_s, mean_t, sigma_s, sigma_t, alpha_s=1., alpha_t=1., eps_gmm=0.1, eps_w=0.0, method='cvx', max_iter=10000, stop_thr=1e-6, verbose=False):
    
    if isinstance(alpha_s, (list, tuple, np.ndarray)):
        K_s = len(alpha_s)
    else:
        K_s = 1
        
    if isinstance(alpha_t, (list, tuple, np.ndarray)):
        K_t = len(alpha_t)
    else:
        K_t = 1
    
    if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
        eps_gmm = np.ones(K_s*K_t) * eps_gmm
    
    _, d = mean_s.shape
    C = np.zeros((K_s, K_t))
    # omt_mu = np.empty((K_s, K_t), dtype=object)
    # omt_sigma = np.empty((K_s, K_t), dtype=object)
    omt_mu = np.zeros((K_s, K_t, d))
    omt_sigma = np.zeros((K_s, K_t, d, d))

    t0 = time.time()
    
    for i in range(K_s):
        for j in range(K_t):
            t1 = time.time()
            dist, _, _ = eogt(mean_s[i], mean_t[j], sigma_s[i], sigma_t[j], eps_gmm[i*K_t + j])
            C[i, j] = dist
            # print(f"Time taken for GMM {i+1}/{K_s} and {j+1}/{K_t}: {time.time() - t1:.2f} seconds")
    
    error_raised = False
    
    if K_s == 1 and K_t == 1:
        omt_w = np.ones((1, 1))
        total_cost = C[0, 0]
        t1 = time.time()
        n_iter = 0.
    else:
        if method == 'cvx': # ECOS: Embedded Conic Solver
            x = cvx.Variable((K_s, K_t), nonneg=True)
            objective = cvx.Minimize(cvx.sum(cvx.multiply(C, x)) + eps_w * cvx.sum(cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :])))

            A = np.ones(K_t)
            B = np.ones(K_s)
            constraints = [x @ A == alpha_s]
            constraints += [x.T @ B == alpha_t]
            problem = cvx.Problem(objective, constraints)
            problem.solve(solver=cvx.SCS, max_iters=max_iter, eps=stop_thr, verbose=verbose)
            t1 = time.time()
            total_cost  = problem.value
            omt_w = x.value
            omt_w /= np.sum(omt_w)
            n_iter = problem.solver_stats.num_iters
        
        elif method == 'sinkhorn':
            eps_w = np.min([eps_w, 0.01])
            omt_w, solver_log = ot.bregman.sinkhorn(alpha_s, alpha_t, C, eps_w, numItermax=max_iter, stopThr=stop_thr, log=True)
            t1 = time.time()
            total_cost = np.sum(omt_w * C)
            n_iter = solver_log['niter']

        elif method == 'w2':
            omt_w = ot.emd(alpha_s, alpha_t, C, numItermax=max_iter)
            t1 = time.time()
            total_cost = np.sum(omt_w * C)
            n_iter = max_iter
        else:
            raise ValueError("Method not recognized. Choose 'cvx', 'sinkhorn', or 'w2'.")
            error_raised = True
            t1 = time.time()
            n_iter = 0
        
    if not error_raised:
        delta_source = np.linalg.norm(np.sum(omt_w, axis=1) - alpha_s)
        delta_target = np.linalg.norm(np.sum(omt_w, axis=0) - alpha_t)
        
        if verbose:
            print("Total transportation cost:", total_cost)
            print("Discrepence in marginals' weights:", delta_source, delta_target)
            print("Number of iterations:", n_iter)
            print("Time taken for transportation optimzation:", t1 - t0, "seconds")

        solver_dict = {
                        'method': method,
                        'total_cost': total_cost,
                        'delta_source': delta_source,
                        'delta_target': delta_target,
                        'omt_time_taken': t1 - t0,
                        'n_iter': n_iter,
                        }
    
        return omt_w, omt_mu, omt_sigma, solver_dict
    


def gmmomt_vectorized(mean_s, mean_t, sigma_s, sigma_t, alpha_s=1., alpha_t=1., eps_gmm=0.1, eps_w=0.0, method='cvx', max_iter=10000, stop_thr=1e-6, verbose=False, n_jobs=8):

    mean_s = np.atleast_2d(mean_s)
    mean_t = np.atleast_2d(mean_t)
    sigma_s = np.atleast_3d(sigma_s)
    sigma_t = np.atleast_3d(sigma_t)

    K_s, d = mean_s.shape
    K_t = mean_t.shape[0]

    if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
        eps_gmm = np.full(K_s * K_t, eps_gmm)
    else:
        eps_gmm = np.asarray(eps_gmm).flatten()
    
    t0 = time.time()
    mean_s_vec = mean_s[:, np.newaxis, :]       # Shape: (K_s, 1, d)
    mean_t_vec = mean_t[np.newaxis, :, :]       # Shape: (1, K_t, d)

    sigma_s_vec = sigma_s[:, np.newaxis, ...]   # Shape: (K_s, 1, d, d)
    sigma_t_vec = sigma_t[np.newaxis, :, ...]   # Shape: (1, K_t, d, d)

    eps_gmm_mat = eps_gmm.reshape(K_s, K_t)     # Shape: (K_s, K_t)

    # --- 2. Vectorized Calculation ---
    t_start = time.time()

    # Call the new batch-aware function once.
    C, omt_mu, omt_sigma = vectorized_eogt_dist(mean_s_vec, mean_t_vec, sigma_s_vec, sigma_t_vec, eps_gmm_mat)
    # print(f"Total time for vectorized calculation: {time.time() - t_start:.2f} seconds")

    realobj = np.isrealobj(C)
    if not realobj:
        print("C matrix is not real-valued, converting to real.")
        C = np.real(C)
        C_real = True
    else:
        C_real = False

    error_raised = False
    
    if K_s == 1 and K_t == 1:
        omt_w = np.ones((1, 1))
        total_cost = C[0, 0]
        t1 = time.time()
        n_iter = 0.
    else:
        if method == 'cvx': # ECOS: Embedded Conic Solver
            x = cvx.Variable((K_s, K_t), nonneg=True)
            A = np.ones(K_t)
            B = np.ones(K_s)
            constraints = [x @ A == alpha_s]
            constraints += [x.T @ B == alpha_t]
            nan_weights = True
            while nan_weights:
                objective = cvx.Minimize(cvx.sum(cvx.multiply(C, x)) + eps_w * cvx.sum(cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :])))
                problem = cvx.Problem(objective, constraints)
                problem.solve(solver=cvx.SCS, max_iters=max_iter, eps=stop_thr) #, verbose=verbose)
                # problem.solve(solver=cvx.MOSEK, mosek_params={
                #                                                 'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iter,  # instead of max_iters
                #                                                 'MSK_DPAR_INTPNT_TOL_REL_GAP': stop_thr,     # relative gap tolerance
                #                                                 'MSK_DPAR_INTPNT_TOL_PFEAS': stop_thr,       # primal feasibility tol
                #                                                 'MSK_DPAR_INTPNT_TOL_DFEAS': stop_thr       # dual feasibility tol
                #                                             }, verbose=verbose)
                t1 = time.time()
                total_cost  = problem.value
                omt_w = x.value
                if np.sum(omt_w) is None:
                    nan_weights = True
                    eps_w += 0.01
                    print("Warning: NaN weights detected.")
                else:
                    nan_weights = False
                    
            omt_w /= np.sum(omt_w)    
            n_iter = problem.solver_stats.num_iters
        
        elif method == 'sinkhorn':
            eps_w = np.min([eps_w, 0.01])
            omt_w, solver_log = ot.bregman.sinkhorn(alpha_s, alpha_t, C, eps_w, numItermax=max_iter, stopThr=stop_thr, log=True)
            t1 = time.time()
            total_cost = np.sum(omt_w * C)
            n_iter = solver_log['niter']

        elif method == 'w2':
            omt_w = ot.emd(alpha_s, alpha_t, C, numItermax=max_iter)
            t1 = time.time()
            total_cost = np.sum(omt_w * C)
            n_iter = max_iter
        else:
            raise ValueError("Method not recognized. Choose 'cvx', 'sinkhorn', or 'w2'.")
            error_raised = True
            t1 = time.time()
            n_iter = 0
        
    if not error_raised:
        delta_source = np.linalg.norm(np.sum(omt_w, axis=1) - alpha_s)
        delta_target = np.linalg.norm(np.sum(omt_w, axis=0) - alpha_t)
        
        if verbose:
            print("Total transportation cost:", total_cost)
            print("Discrepence in marginals' weights:", delta_source, delta_target)
            print("Number of iterations:", n_iter)
            print("Time taken for transportation optimzation:", t1 - t0, "seconds")
        
        solver_dict = {
                        'method': method,
                        'total_cost': total_cost,
                        'delta_source': delta_source,
                        'delta_target': delta_target,
                        'omt_time_taken': t1 - t0,
                        'n_iter': n_iter,
                        'realnum_issue': C_real,
                        }
        
        return omt_w, omt_mu, omt_sigma, solver_dict



def shallow_gmm(x, n_components, covariance_type='full', max_iter=1000, n_init=5, reg_covar=1e-6, random_state=None, verbose=False, combine=False):

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        raise ValueError("random_state must be an integer or None.")

    # check if x is a 2D array
    if x.ndim != 2:
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        else:
            raise ValueError("Input data must be a 2D array.")
    
    dim = x.shape[1]
    
    # --- Fit GMM ---
    if combine and dim > 16:
        gmm_ = []
        num_gmm = dim // 16  + (int(dim % 16 != 0))
        n_components_ = n_components // num_gmm + (int(n_components % num_gmm != 0))
        if verbose:
            print(f"Splitting GMM into {num_gmm} components with {n_components_} each for {dim} dimensions.")
            n_components__ = [n_components_ for m in range(num_gmm)]
            index_combinations = list(itertools.product(*[range(n) for n in n_components__]))
            print(len(index_combinations), "combinations of components will be created.")
            
        for n_gmm in range(num_gmm):
            gmm__ = GaussianMixture(
                        n_components=n_components_,
                        covariance_type=covariance_type,
                        n_init=n_init,
                        max_iter=max_iter,
                        warm_start=True,
                        tol=1e-3/x.shape[-1],
                        init_params='k-means++',
                        reg_covar=reg_covar,
                        )
            gmm__.fit(x[:, n_gmm*16:min((n_gmm+1)*16, dim)])
            gmm_.append(gmm__)
            
        if verbose:
            print(f"Learning splitted GMM for {dim} dimensions.")  

        gmm = combine_gmms(gmm_)    
    else:
        gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_init=n_init,
                        max_iter=max_iter,
                        warm_start=True,
                        tol=1e-3/x.shape[-1],
                        init_params='k-means++',
                        reg_covar=reg_covar,
                        )
        gmm.fit(x)
    
    if verbose:
        print(f"Fitted a GMM with {len(gmm.weights_)} components.")
        print(f"Model converged: {gmm.converged_}")
        print(f"Number of iterations: {gmm.n_iter_}")  
        print("-" * 50)
    
    return gmm


def combine_gmms(gmm_list):
    """
    Combine multiple GMMs (fitted on disjoint subsets of features)
    into one high-dimensional GMM by Cartesian product of components.

    Parameters
    ----------
    gmm_list : list of GaussianMixture
        Each GaussianMixture must be fitted on disjoint subsets of features.

    Returns
    -------
    combined : GaussianMixture
        New GaussianMixture object with combined weights, means, covariances.
    """
    n_models = len(gmm_list)
    n_components = [len(m.weights_) for m in gmm_list]
    dim_parts = [m.means_.shape[1] for m in gmm_list]
    total_dim = sum(dim_parts)

    # Generate all combinations of component indices
    index_combinations = list(itertools.product(*[range(n) for n in n_components]))

    combined_weights = []
    combined_means = []
    combined_covariances = []

    for combo in index_combinations:
        # --- Combined weight ---
        w = 1.0
        for mi, ci in enumerate(combo):
            w *= gmm_list[mi].weights_[ci]
        combined_weights.append(w)

        # --- Combined mean ---
        mean_parts = [gmm_list[mi].means_[ci] for mi, ci in enumerate(combo)]
        mean_full = np.concatenate(mean_parts)
        combined_means.append(mean_full)

        # --- Combined covariance (block diagonal) ---
        cov_blocks = [gmm_list[mi].covariances_[ci] for mi, ci in enumerate(combo)]
        cov_full = np.zeros((total_dim, total_dim))
        start = 0
        for part_dim, cov in zip(dim_parts, cov_blocks):
            cov_full[start:start+part_dim, start:start+part_dim] = cov
            start += part_dim
        combined_covariances.append(cov_full)
    
    combined_iterations = [gmm_list[mi].n_iter_ for mi, ci in enumerate(combo)]
    combined_convergence = all(gmm_list[mi].converged_ for mi, ci in enumerate(combo))

    combined_weights = np.array(combined_weights)
    combined_means = np.array(combined_means)
    combined_covariances = np.array(combined_covariances)

    # --- Create a scikit-learn GaussianMixture with our params ---
    combined_gmm = GaussianMixture(
                                    n_components=len(combined_weights),
                                    covariance_type='full'
                                    )
    combined_gmm.weights_ = combined_weights
    combined_gmm.means_ = combined_means
    combined_gmm.covariances_ = combined_covariances
    combined_gmm.n_iter_ = np.sum(combined_iterations)
    combined_gmm.converged_ = combined_convergence

    # Compute precisions as sklearn expects
    try:
        precisions = np.linalg.inv(combined_covariances)
    except:
        precisions = np.linalg.pinv(combined_covariances + 1e-6 * np.eye(total_dim))

    precisions_cholesky = np.array([
        cholesky(prec, lower=False) for prec in precisions
    ])

    combined_gmm.precisions_ = precisions
    combined_gmm.precisions_cholesky_ = precisions_cholesky

    return combined_gmm


def ot_w2(x_source, x_target, data_source, data_target, max_iter=10000, verbose=False):
    
    time0 = time.time()
    M = ot.dist(x_source, x_target)
    M /= M.max()
    # Solve the OT problem - Kantorovich formulation
    pi_w2, log_w2 = ot.emd(data_source, data_target, M, numItermax=max_iter, log=True)
    time1 = time.time()

    if verbose:
        print("Total transportation cost:", np.sum(pi_w2 * M))
        print("Discrepence in marginals' weights:", sum((np.sum(pi_w2, axis=1) - data_source)**2), sum((np.sum(pi_w2, axis=0) - data_target)**2))
        print("Time taken for transportation optimzation:", time1 - time0, "seconds")
    
    log_w2['time'] = time1 - time0
    
    return pi_w2, log_w2
    
    
def ot_sinkh(x_source, x_target, data_source, data_target, eps=0.1, max_iter=10000, stop_thr=1e-6, verbose=False):
    
    time0 = time.time()
    M = ot.dist(x_source.reshape((-1, 1)), x_target.reshape((-1, 1)))
    M /= M.max()
    # Solve the OT problem - Kantorovich formulation
    pi_sinkh, log_sinkh = ot.bregman.sinkhorn(data_source, data_target, M, reg=eps, log=True, stopThr=stop_thr)
    time1 = time.time()

    if verbose:
        print("Total transportation cost:", np.sum(pi_sinkh * M))
        print("Discrepence in marginals' weights:", sum((np.sum(pi_sinkh, axis=1) - data_source)**2), sum((np.sum(pi_sinkh, axis=0) - data_target)**2))
        print("Number of iterations:", log_sinkh['niter'])
        print("Time taken for transportation optimzation:", time1 - time0, "seconds")
    
    log_sinkh['time'] = time1 - time0
    
    return pi_sinkh, log_sinkh
