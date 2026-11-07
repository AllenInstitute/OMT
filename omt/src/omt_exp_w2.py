import os, sys
sys.path.append("..")
import matplotlib
import numpy as np
import pdb
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import gc
import jax.numpy as jnp
import jax
from sklearn.decomposition import PCA
import tracemalloc
import pickle

from src import distributions
from src import map_benchmark

from ott.tools import sinkhorn_divergence
from ott.geometry import costs, geometry, pointcloud

import src.map_benchmark as mbm
from src.metrics import metrics_to_dict, omt_score_fitted_maps, score_fitted_fwdmaps, compute_cost, compute_BW_metrics, L2_UVP_cost, compute_w2_divergence


from functions import sample_gmm_kdim, compute_mmd, transport_cost
# from omt.solver import omt
from general_solver import omt, sample_mm, mmd_rbf
from exp_functions import points_transport

@jax.jit
def sinkhorn_loss(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: float = 0.01
) -> float:
    """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
    a = jnp.ones(len(x)) / len(x)
    b = jnp.ones(len(y)) / len(y)

    sdiv, _ = sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud, x, y, epsilon=epsilon, a=a, b=b
    )
    return sdiv


def sinkh_div(x: jnp.ndarray, y: jnp.ndarray, batch_size: int, epsilon: float = 0.01) -> float:
    len_x = x.shape[0]
    cost = []
    for i in range(0, len_x, batch_size):
        end_idx = min(i + batch_size, len_x)
        x_batch = x[i:end_idx]
        y_batch = y[i:end_idx]
        cost.append(sinkhorn_loss(x_batch, y_batch, epsilon=epsilon))

    return np.mean(cost)



eps = 0.01
max_iter = int(1e6)
stop_thr = 1e-8
fontsize = 16

BENCHMARK = 'Mix3toMix10'
GPU_DEVICE = 0
torch.cuda.set_device(GPU_DEVICE)
batch_size = 1024
cost_fn = costs.SqEuclidean()


dim = [2, 16, 64, 128, 256]
n_sampless = [10000, 20000, 30000, 50000, 50000] #[10000, 10000, 20000, 20000, 20000, 30000, 50000, 50000]
test_smp = 10000
repetitions = 10
smp_scale = 100
max_n_smp = int(1e6)
combined = False
check_outlier = False
K0 = 10
K1 = 50


for rep in range(repetitions):
    for id, d in enumerate(dim):
        for noise_scale, p_drop_out in [(0. , 0.), (.1, 0.), (.25, 0.), (.5, 0.), (1., 0.), (0., 0.05), (0., .1), (0., .25), (0., .5)]:
            n_samples = n_sampless[id]  
            EXP_NAME = f'{BENCHMARK}_EPSILON_{eps}_MOT_EXP_DIM_{d}_Epoch_{max_iter}_{K0}_{K1}_run_{rep}_numSmp_{n_samples}' + \
                            f'_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'
                            
            save_path = f'./results/gmmot/{EXP_NAME}'
            os.makedirs(save_path, exist_ok=True)
            print(f"Running {rep} for dim {d}")
            tracemalloc.start()
            benchmark = mbm.Mix3ToMix10Benchmark(d)

            # X = benchmark.input_sampler.sample(int(1e6))
            # X = X.cpu().detach().numpy()
            # mu_S = np.mean(X, axis=0)
            # cov_S = np.cov(X.T)

            # Y = benchmark.output_sampler.sample(int(1e6))
            # Y = Y.cpu().detach().numpy()
            # mu_T = np.mean(Y, axis=0)
            # cov_T = np.cov(Y.T)


            # min_num_smp = int(np.max([source_param, target_param]) * smp_scale)
            # if n_samples < min_num_smp:
            #     n_samples = min(min_num_smp, max_n_smp)
            #     print(f"Warning: n_samples is less than the minimum required samples. Setting n_samples to {n_samples}.")
            
            n_samples = n_sampless[id]    
            x_noise = torch.randn(n_samples, d) * noise_scale**2
            y_noise = torch.randn(n_samples, d) * noise_scale**2
            X_train = benchmark.input_sampler.sample(n_samples) + x_noise.cuda(GPU_DEVICE)
            Y_train = benchmark.output_sampler.sample(n_samples) + y_noise.cuda(GPU_DEVICE)
            X_train = F.dropout(X_train, p=p_drop_out, training=True)                    
            Y_train = F.dropout(Y_train, p=p_drop_out, training=True)
            
            X_train_ = X_train.cpu().detach().numpy()
            Y_train_ = Y_train.cpu().detach().numpy()
            
            # d_s, d_t = 1.0, 1.0
            # while d_s + d_t > 1e-3:
            # omt_w, _, _, solver_dict = omt(
            #                                 data_source=X_train_, 
            #                                 data_target=Y_train_, 
            #                                 n_components=(K0, K1), 
            #                                 eps_gmm=0., 
            #                                 eps_w=0., 
            #                                 method='w2', 
            #                                 cov_type=cov_type[id],
            #                                 max_iter=max_iter,
            #                                 stop_thr=stop_thr,
            #                                 verbose=False,
            #                                 vectorize=True,
            #                                 combine=combined,
            #                                 check_outlier=check_outlier,
            #                                 )
            omt_w, solver_dict = omt(
                        data_source=X_train_,
                        data_target=Y_train_,
                        n_components=(K0, K1),
                        eps_gmm=eps,
                        eps_w=eps,
                        method='cvx',
                        fitting_mode_source='exp',
                        fitting_mode_target='exp',
                        cov_type='diag',
                        max_iter=max_iter,
                        n_iter_em=1000,
                        stop_thr=stop_thr,
                        verbose=True,
                    )
            
            # d_s = solver_dict['delta_source']
            # d_t = solver_dict['delta_target']
            # print(d_s, d_t)
            
            _, cpu_peak = tracemalloc.get_traced_memory()
            gpu_peak = torch.cuda.max_memory_allocated()
            tracemalloc.stop()
            print(f"Memory Peak after OMT fitting: CPU {cpu_peak / 1024**2} MB, GPU {gpu_peak / 1024**2} MB")
                
       
            X_test = benchmark.input_sampler.sample(test_smp) 
            X_test.requires_grad_(True)
            Y_test = benchmark.map_fwd(X_test, nograd=True) 
            Y_test_noise = benchmark.map_fwd(X_test, nograd=True)
            
            X_test_ = X_test.cpu().detach().numpy()
            Y_test_ = Y_test.cpu().detach().numpy()
            

            y_ = points_transport(X_test_, omt_w, solver_dict)
            
            # x_ = points_transport(Y_test_, omt_w, solver_dict, direction="inv")
            # xx_ = points_transport(y_, omt_w, solver_dict, direction="inv")
            # yy_ = points_transport(x_, omt_w, solver_dict, direction="fwd")
            
            # Y_push = torch.tensor(x_, dtype=torch.float32).cuda(GPU_DEVICE)
            # YY_push = torch.tensor(xx_, dtype=torch.float32).cuda(GPU_DEVICE)
            XX_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)
            
            # mmd_T_inv = compute_mmd(X=X_test_, Y=x_) * 100
            # mmd_T_inv_2 = compute_mmd(X=X_test_, Y=xx_) * 100
            # mmd_T_2 = compute_mmd(X=Y_test_, Y=yy_) * 100
            
            # L2_T_inv = np.mean(np.linalg.norm(X_test_ - x_, ord=2, axis=1))
            # L2_T_inv_2 = np.mean(np.linalg.norm(X_test_ - xx_, ord=2, axis=1))
            # L2_T_2 = np.mean(np.linalg.norm(Y_test_ - yy_, ord=2, axis=1))
            
            # L2_UVP_inv, cos_sim_inv = L2_UVP_cost(benchmark, Y_test, X_test, Y_push, direction='inv')
            # L2_UVP_inv_2, cos_sim_inv_2 = L2_UVP_cost(benchmark, Y_test, X_test, YY_push, direction='inv')
            # L2_UVP_2, cos_sim_2 = L2_UVP_cost(benchmark, X_test, Y_test, XX_push)
            
            # dist_inv = sinkh_div(x_, X_test_, batch_size=1024, epsilon=0.001)
            # dist_inv_2 = sinkh_div(xx_, X_test_, batch_size=1024, epsilon=0.001)
            # dist_2 = sinkh_div(yy_, Y_test_, batch_size=1024, epsilon=0.001)

            # print(f"Costs Inv: {mmd_T_inv}, {L2_T_inv}, {L2_UVP_inv}, {dist_inv}")
            # print(f"Costs Inv 2: {mmd_T_inv_2}, {L2_T_inv_2}, {L2_UVP_inv_2}, {dist_inv_2}") 
            # print(f"Costs Fwd 2: {mmd_T_2}, {L2_T_2}, {L2_UVP_2}, {dist_2}")

            

            print('Calculating metrics...')
            X_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)

            # BW_UVP = compute_BW_metrics(y_, mu_T, cov_T)
            mmd_T = compute_mmd(X=Y_test_, Y=y_) * 100
            mmd_T_noise = compute_mmd(X=Y_test_, Y=y_) * 100
            
            L2_T = np.mean(np.linalg.norm(Y_test_ - y_, ord=2, axis=1))
            L2_T_noise = np.mean(np.linalg.norm(Y_test_ - y_, ord=2, axis=1)) 
            
            trans_cost = transport_cost(solver_dict, omt_w)
            
            L2_UVP, cos_sim = L2_UVP_cost(benchmark, X_test, Y_test, X_push)
            L2_UVP_noise, cos_sim_noise = L2_UVP_cost(benchmark, X_test, Y_test_noise, X_push)
            
            dist = sinkh_div(y_, Y_test_, batch_size=1024, epsilon=0.001) 
            dist_noise = sinkh_div(y_, Y_test_, batch_size=1024, epsilon=0.001)
            
            print(f"Costs for noise scale {noise_scale} and dropout {p_drop_out}: {mmd_T}, {L2_T}, {L2_UVP}, {dist}")
            # cost_fwd = sinkh_div(X_test_, Y_test_, batch_size=batch_size, epsilon=0.001)
            # cost_transport = sinkh_div(X_test_, y_, batch_size=batch_size, epsilon=0.001)
            
            omt_dict = dict()
            omt_dict['solver'] = solver_dict
            omt_dict['omt_w'] = omt_w
            filename = f'./results/gmmot/{EXP_NAME}/omt_solver.pkl'
            with open(filename, 'wb') as pkl_file:
                pickle.dump(omt_dict, pkl_file)

            final_metrics = {
                            'mmd': [mmd_T, 0],
                            'mmd_noise': mmd_T_noise,
                            'mmd_inv': [0, 0],
                            'cost': trans_cost,
                            'sinkh_div_baseline': 0.,
                            'sinkh_div_trans': 0.,
                            'sinkh_div_y': [dist, 0],
                            'sinkh_div_y_noise': dist_noise,
                            'sinkh_div_x': [0, 0],
                            'L2_dist': [L2_T, 0],
                            'L2_dist_inv': [0, 0],
                            'L2_dist_noise': L2_T_noise,
                            'L2_UVP': [L2_UVP, 0],
                            'L2_UVP_noise': L2_UVP_noise, 
                            'L2_UVP_inv': [0, 0],
                            'cos': [cos_sim, 0],
                            'cos_noise': cos_sim_noise,
                            'cos_inv': [0, 0],
                            'num_iter': solver_dict['n_total_iter'],
                            'time': solver_dict['time'],
                            'cpu_memory_peak_MB': cpu_peak / 1024**2,
                            'gpu_memory_peak_MB': gpu_peak / 1024**2,
                            }
        
            np.save(f'{save_path}/metrics.npy', final_metrics)
            pdb.set_trace()