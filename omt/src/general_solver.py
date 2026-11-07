import numpy as np
import cvxpy as cvx
import ot
import time
import pdb
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from scipy.linalg import sqrtm as scipy_sqrtm

from exp_functions import em_mixture, exp_fitting, vectorized_W2_cost_matrix

try:
    from functions import eogt, vectorized_eogt_dist
except ImportError:
    eogt = vectorized_eogt_dist = None


def _safe_sqrtm(A):
    S = scipy_sqrtm(A)
    if np.isrealobj(A):
        S = S.real
    return S


def _sqrtm_2x2(M):
    """Closed-form square root of a 2x2 symmetric PD matrix."""
    t = M[0, 0] + M[1, 1]
    d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    s = np.sqrt(max(d, 0.0))
    tau = np.sqrt(max(t + 2.0 * s, 0.0))
    if tau < 1e-15:
        return np.zeros_like(M)
    return (M + s * np.eye(2)) / tau


def _bures_W2_cost_matrix(means_s, covs_s, means_t, covs_t):
    K_s = means_s.shape[0]
    K_t = means_t.shape[0]
    D = means_s.shape[1]
    use_2x2 = (D == 2)

    diff_mu_sq = np.sum((means_s[:, np.newaxis, :] - means_t[np.newaxis, :, :]) ** 2, axis=2)
    tr_s = np.trace(covs_s, axis1=1, axis2=2)
    tr_t = np.trace(covs_t, axis1=1, axis2=2)

    C = diff_mu_sq + tr_s[:, np.newaxis] + tr_t[np.newaxis, :]

    if use_2x2:
        sqrt_cache = np.array([_sqrtm_2x2(covs_s[i]) for i in range(K_s)])
        for i in range(K_s):
            sqrt_Si = sqrt_cache[i]
            for j in range(K_t):
                inner = sqrt_Si @ covs_t[j] @ sqrt_Si
                sqrt_inner = _sqrtm_2x2(inner)
                C[i, j] -= 2.0 * np.trace(sqrt_inner)
    else:
        for i in range(K_s):
            sqrt_Si = _safe_sqrtm(covs_s[i])
            for j in range(K_t):
                inner = sqrt_Si @ covs_t[j] @ sqrt_Si
                sqrt_inner = _safe_sqrtm(inner)
                C[i, j] -= 2.0 * np.trace(sqrt_inner)

    np.maximum(C, 0.0, out=C)
    return C


def _fit_one(data, mode, n_components, cov_type, reg_covar, n_iter_em,
             random_seed, verbose, force_diag=False):
    """Fit a single side (source or target) according to its mode."""
    valid_modes = ('gauss', 'exp', 'mix')
    if mode not in valid_modes:
        raise ValueError(f"fitting_mode '{mode}' not recognized. Choose from {valid_modes}.")

    if mode == 'gauss':
        ct = 'diag' if force_diag else cov_type
        mm = gmm_fitting(data, n_components, covariance_type=ct,
                         reg_covar=reg_covar, random_state=random_seed, verbose=verbose)
    elif mode == 'exp':
        mm = exp_fitting(data, K=n_components, n_iter=n_iter_em, bw=0.15)
    else:
        mm = em_mixture(data, K=n_components, n_iter=n_iter_em, bw=0.15)
        mm["n_iter"] = n_iter_em
    return mm


def omt(
        data_source,
        data_target,
        n_components=(100, 100),
        eps_gmm=0.1,
        eps_w=0.0,
        method='cvx',
        fitting_mode='mix',
        fitting_mode_source=None,
        fitting_mode_target=None,
        cov_type='full',
        reg_covar=1e-6,
        max_iter=10000,
        n_iter_em=500,
        stop_thr=1e-6,
        random_seed=None,
        verbose=False,
        check_outlier=False,
        reshape_cov=False,
        ):

    fm_s = fitting_mode_source if fitting_mode_source is not None else fitting_mode
    fm_t = fitting_mode_target if fitting_mode_target is not None else fitting_mode

    if n_components[0] > data_source.shape[0] or n_components[1] > data_target.shape[0]:
        raise ValueError("Number of components cannot exceed number of samples!")
    if eps_gmm < 0:
        raise ValueError("eps_gmm must be non-negative!")
    if eps_w < 0:
        raise ValueError("eps_w must be non-negative!")
    if method not in ['cvx', 'sinkhorn', 'w2']:
        raise ValueError("Method not recognized. Choose 'cvx', 'sinkhorn', or 'w2'.")

    t0 = time.time()

    if check_outlier:
        Isofor = IsolationForest(max_samples=10000, n_estimators=1000, contamination=0.1, random_state=random_seed)
        Isofor.fit(data_source)
        scores = Isofor.predict(data_source)
        data_source = data_source[scores == 1]
        Isofor.fit(data_target)
        scores = Isofor.predict(data_target)
        data_target = data_target[scores == 1]

    both_gauss = (fm_s == 'gauss' and fm_t == 'gauss')
    mm_s = _fit_one(data_source, fm_s, n_components[0], cov_type, reg_covar, n_iter_em,
                    random_seed, verbose, force_diag=(fm_s == 'gauss' and not both_gauss))
    mm_t = _fit_one(data_target, fm_t, n_components[1], cov_type, reg_covar, n_iter_em,
                    random_seed, verbose, force_diag=(fm_t == 'gauss' and not both_gauss))

    t1 = time.time()
    if verbose:
        print("Number of components in source:", n_components[0])
        print("Number of components in target:", n_components[1])
        print("Fitting mode: source=%s, target=%s" % (fm_s, fm_t))
        print("Time taken for fitting:", t1 - t0, "seconds")

    c_type = 'full' if (both_gauss and cov_type == 'full') else 'diag'
    omt_w, solver_dict = mm_ot(
                                mm_s=mm_s,
                                mm_t=mm_t,
                                c_type=c_type,
                                eps_gmm=eps_gmm,
                                eps_w=eps_w,
                                method=method,
                                max_iter=max_iter,
                                stop_thr=stop_thr,
                                verbose=verbose,
                                )

    solver_dict['mm_s'] = mm_s
    solver_dict['mm_t'] = mm_t
    solver_dict['fitting_mode_source'] = fm_s
    solver_dict['fitting_mode_target'] = fm_t
    solver_dict['mm_time_taken'] = t1 - t0
    solver_dict['n_total_iter'] = mm_s["n_iter"] + mm_t["n_iter"] + solver_dict['n_iter']
    solver_dict['time'] = solver_dict['omt_time_taken'] + solver_dict['mm_time_taken']

    if verbose:
        print("Number of total iterations:", solver_dict['n_total_iter'])
        print("Total time taken:", solver_dict['time'], "seconds")

    return omt_w, solver_dict
    

def mm_ot(mm_s, 
          mm_t,
          c_type='full', 
          eps_gmm=0.1, 
          eps_w=0.0, 
          method='cvx', 
          max_iter=10000, 
          stop_thr=1e-6, 
          verbose=False, 
          n_jobs=8):
    
        
    alpha_s = mm_s["weights"]
    alpha_t = mm_t["weights"]
    
    K_s = mm_s["K"]
    K_t = mm_t["K"]


    if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
        eps_gmm = np.full(K_s * K_t, eps_gmm)
    else:
        eps_gmm = np.asarray(eps_gmm).flatten()
    
    t0 = time.time()
    
    # --- 2. Vectorized Calculation ---
    t_start = time.time()

    if c_type == 'full':
        mean_s = np.atleast_2d(mm_s["means"])
        mean_t = np.atleast_2d(mm_t["means"])
        sigma_s = np.atleast_3d(mm_s["covariances"])
        sigma_t = np.atleast_3d(mm_t["covariances"])

        if vectorized_eogt_dist is not None:
            mean_s_vec = mean_s[:, np.newaxis, :]
            mean_t_vec = mean_t[np.newaxis, :, :]
            sigma_s_vec = sigma_s[:, np.newaxis, ...]
            sigma_t_vec = sigma_t[np.newaxis, :, ...]
            eps_gmm_mat = eps_gmm.reshape(K_s, K_t)
            C = vectorized_eogt_dist(mean_s_vec, mean_t_vec, sigma_s_vec, sigma_t_vec, eps_gmm_mat)
        else:
            C = _bures_W2_cost_matrix(mean_s, sigma_s, mean_t, sigma_t)
    else:
        mean_s = np.atleast_2d(mm_s["means"])
        mean_t = np.atleast_2d(mm_t["means"])
        c_rates_s = np.atleast_2d(mm_s["c_rates"])
        c_rates_t = np.atleast_2d(mm_t["c_rates"])
        d_rates_s = np.atleast_2d(mm_s["d_rates"])
        d_rates_t = np.atleast_2d(mm_t["d_rates"])
        
        types_s = mm_s["types"]
        types_t = mm_t["types"]
        C = vectorized_W2_cost_matrix(
                                    mean_s, c_rates_s, d_rates_s, types_s,
                                    mean_t, c_rates_t, d_rates_t, types_t,
                                    )
    error_raised = False
    
    realobj = np.isrealobj(C)
    if not realobj:
        print("C matrix is not real-valued, converting to real.")
        pdb.set_trace()
    
    if K_s == 1 and K_t == 1:
        omt_w = np.ones((1, 1))
        total_cost = C[0, 0]
        t1 = time.time()
        n_iter = 0.
    else:
        if method == 'cvx': # ECOS: Embedded Conic Solver
            A = np.ones(K_t)
            B = np.ones(K_s)
            
            nan_weights = True
            while nan_weights:
                x = cvx.Variable((K_s, K_t), nonneg=True)
                constraints = [x @ A == alpha_s]
                constraints += [x.T @ B == alpha_t]
                objective = cvx.Minimize(cvx.sum(cvx.multiply(C, x)) + eps_w * cvx.sum(cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :])))
                problem = cvx.Problem(objective, constraints)
                problem.solve(solver=cvx.SCS, max_iters=max_iter, eps=stop_thr) 
                t1 = time.time()
                total_cost  = problem.value
                omt_w = x.value
                if np.sum(omt_w) is None:
                    nan_weights = True
                    eps_w += 0.01
                    print("Warning: NaN weights detected.")
                else:
                    nan_weights = False
                    
            t1 = time.time()
            omt_w /= np.sum(omt_w)
            n_iter = problem.solver_stats.num_iters
        
        elif method == 'sinkhorn':
            eps_w = np.min([eps_w, 0.1])
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
        
        return omt_w, solver_dict
    
    
        
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
        
        return omt_w, solver_dict



def gmm_fitting(x, n_components, covariance_type='full', reg_covar=1e-6, random_state=None, verbose=False):

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
    
    # --- Fit GMM ---
    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=random_state,
                        reg_covar=reg_covar,
                        )
    gmm.fit(x)
    
    if verbose:
        print(f"Fitted a GMM with {n_components} components.")
        print(f"Model converged: {gmm.converged_}")
        print(f"Number of iterations: {gmm.n_iter_}")  
        print("-" * 50)
    
    covariances = gmm.covariances_
    result = dict(K=n_components,
                  types=['Gaussian'] * n_components,
                  weights=gmm.weights_,
                  means=gmm.means_,
                  covariances=covariances,
                  n_iter=gmm.n_iter_)

    if covariance_type == 'diag':
        covariances = np.array([np.diag(c) for c in covariances])
        result['covariances'] = covariances
        D = gmm.means_.shape[1]
        sigmas = np.sqrt(gmm.covariances_)
        result['c_rates'] = 1.0 / (sigmas * np.sqrt(2.0))
        result['d_rates'] = 1.0 / (sigmas * np.sqrt(2.0))

    return result


def ot_w2(x_source, x_target, data_source, data_target, max_iter=10000, verbose=False):
    
    time0 = time.time()
    M = ot.dist(x_source.reshape((-1, 1)), x_target.reshape((-1, 1)))
    M /= M.max()
    # Solve the OT problem - Kantorovich formulation
    pi_w2, log_w2 = ot.emd(data_source, data_target, M, numItermax=max_iter, log=True)
    time1 = time.time()
    cost = np.sum(pi_w2 * M)

    if verbose:
        print("Total transportation cost:", cost)
        print("Discrepence in marginals' weights:", sum((np.sum(pi_w2, axis=1) - data_source)**2), sum((np.sum(pi_w2, axis=0) - data_target)**2))
        print("Time taken for transportation optimzation:", time1 - time0, "seconds")
    
    log_w2['time'] = time1 - time0
    
    return pi_w2, log_w2, cost
    
    
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


def sample_mm(mm_params, n_samples, random_state=None):
    rng = np.random.default_rng(random_state)
    weights = mm_params['weights']
    means = mm_params['means']
    K = mm_params['K']
    D = means.shape[1] if means.ndim > 1 else 1

    labels = rng.choice(K, size=n_samples, p=weights)
    samples = np.zeros((n_samples, D))

    if 'covariances' in mm_params:
        covs = mm_params['covariances']
        for k in range(K):
            mask = labels == k
            nk = mask.sum()
            if nk == 0:
                continue
            samples[mask] = rng.multivariate_normal(means[k], covs[k], size=nk)
    else:
        from exp_functions import DoubleExponential, GaussianLike
        types = mm_params['types']
        c_rates = mm_params['c_rates']
        d_rates = mm_params['d_rates']
        for k in range(K):
            mask = labels == k
            nk = mask.sum()
            if nk == 0:
                continue
            for d in range(D):
                if types[k] in ('Gaussian', 'G'):
                    dist = GaussianLike(means[k, d], c_rates[k, d], d_rates[k, d])
                else:
                    dist = DoubleExponential(means[k, d], c_rates[k, d], d_rates[k, d])
                u = rng.random(nk)
                samples[mask, d] = dist.inv_cdf(u)

    return samples, labels


def mmd_rbf(X, Y, gamma=1.0):
    from scipy.spatial.distance import cdist
    XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    return XX.mean() + YY.mean() - 2 * XY.mean()


def points_transport_mm(data_source, omt_w, solver_dict):
    from exp_functions import points_transport
    return points_transport(data_source, omt_w, solver_dict)