import numpy as np
import cvxpy as cvx
import ot 
import time
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture

from .functions import eogt


def omt(data_source, data_target, n_components=(100, 100), eps_gmm=0.1, eps_w=0.0, method='cvx', cov_type='full', max_iter=10000, stop_thr=1e-6, random_seed=None, verbose=False, vectorize=True):
    
    if n_components[0] > data_source.shape[0] or n_components[1] > data_target.shape[0]:
        raise ValueError("Number of components cannot exceed number of samples!")
    if eps_gmm < 0:
        raise ValueError("eps_gmm must be non-negative!")
    if eps_w < 0:
        raise ValueError("eps_w must be non-negative!")
    if method not in ['cvx', 'sinkhorn', 'w2', 'cvx']:
        raise ValueError("Method not recognized. Choose 'cvx', 'sinkhorn', or 'w2'.")

    t0 = time.time()
    gmm_s = shallow_gmm(data_source, n_components[0], covariance_type=cov_type, random_state=random_seed, verbose=verbose)
    gmm_t = shallow_gmm(data_target, n_components[1], covariance_type=cov_type, random_state=random_seed, verbose=verbose)
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
        omt_w, omt_mu, omt_sigma, solver_dict = gmmomt_mapreduce(
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
    
    C = np.zeros((K_s, K_t))
    omt_mu = np.empty((K_s, K_t), dtype=object)
    omt_sigma = np.empty((K_s, K_t), dtype=object)
    
    t0 = time.time()
    for i in range(K_s):
        for j in range(K_t):
            omt_mu[i,j] = []
            omt_sigma[i,j] = []
            # try:    
            dist, pi_mu, pi_sigma = eogt(mean_s[i], mean_t[j], sigma_s[i], sigma_t[j], eps_gmm[i*K_t + j])
            # except:
            #     mu_0 = np.expand_dims(mean_s[i], axis=0)
            #     mu_1 = np.expand_dims(mean_t[j], axis=0)
            #     sig_0 = np.expand_dims(sigma_s[i], axis=0)
            #     sig_1 = np.expand_dims(sigma_t[j], axis=0)
            #     dist, pi_mu, pi_sigma = eogt(mu_0, mu_1, sig_0, sig_1, eps_gmm[i*K_t + j])
            C[i, j] = dist
            omt_mu[i,j].append(pi_mu)
            omt_sigma[i,j].append(pi_sigma)

    error_raised = False
    
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
    

def gmmomt_mapreduce(mean_s, mean_t, sigma_s, sigma_t, alpha_s=1., alpha_t=1., eps_gmm=0.1, eps_w=0.0, method='cvx', max_iter=10000, stop_thr=1e-6, verbose=False, n_jobs=10):

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
    # Create all input tuples
    inputs = [
        (mean_s[i], mean_t[j], sigma_s[i], sigma_t[j], eps_gmm[i * K_t + j])
        for i in range(K_s)
        for j in range(K_t)
    ]

    def wrapper(args):
        return eogt(*args)

    results = list(map(wrapper, inputs))

    # Unpack results
    C = np.array([r[0] for r in results]).reshape(K_s, K_t)
    omt_mu = np.array([r[1] for r in results], dtype=object).reshape(K_s, K_t, -1)
    omt_sigma = np.array([r[2] for r in results], dtype=object).reshape(K_s, K_t, 2*sigma_s.shape[-1], 2*sigma_s.shape[-1])

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
            problem.solve(solver=cvx.SCS, max_iters=max_iter, eps=stop_thr)
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



def shallow_gmm(x, n_components, covariance_type='full', random_state=None, verbose=False):
    
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
    gmm = GaussianMixture(n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=random_state)
    gmm.fit(x)
    
    if verbose:
        print(f"Fitted a GMM with {n_components} components.")
        print(f"Model converged: {gmm.converged_}")
        print(f"Number of iterations: {gmm.n_iter_}")  
        print("-" * 50)
    
    return gmm


def expomt(param_s, param_t, alpha_s=1., alpha_t=1., eps_mm=0.1, eps_w=0.0, method='cvx', max_iter=10000, stop_thr=1e-6, verbose=False, n_jobs=10):

    K_s = param_s['K_g'] + param_s['K_exp']
    K_t = param_t['K_g'] + param_t['K_exp']
    
    if param_s['K_g'] > 0:
        mean_s = param_s['gaussians']['mean']
        sigma_s = param_s['gaussians']['sigma']
    if param_s['K_exp'] > 0:
        mean_s = np.concatenate((mean_s, param_s['exponentials']['mean']), axis=0)
        sigma_s = np.concatenate((sigma_s, param_s['exponentials']['scale']), axis=0)

    K_s, d = mean_s.shape
    K_t = mean_t.shape[0]

    if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
        eps_gmm = np.full(K_s * K_t, eps_gmm)
    else:
        eps_gmm = np.asarray(eps_gmm).flatten()
    
    t0 = time.time()
    # Create all input tuples
    inputs = [
        (mean_s[i], mean_t[j], sigma_s[i], sigma_t[j], eps_gmm[i * K_t + j])
        for i in range(K_s)
        for j in range(K_t)
    ]

    def wrapper(args):
        return eogt(*args)

    results = list(map(wrapper, inputs))

    # Unpack results
    C = np.array([r[0] for r in results]).reshape(K_s, K_t)
    omt_mu = np.array([r[1] for r in results], dtype=object).reshape(K_s, K_t, -1)
    omt_sigma = np.array([r[2] for r in results], dtype=object).reshape(K_s, K_t, 2*sigma_s.shape[-1], 2*sigma_s.shape[-1])

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
            problem.solve(solver=cvx.SCS, max_iters=max_iter, eps=stop_thr)
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
    

def ot_w2(x_source, x_target, data_source, data_target, max_iter=10000, verbose=False):
    
    time0 = time.time()
    M = ot.dist(x_source.reshape((-1, 1)), x_target.reshape((-1, 1)))
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
