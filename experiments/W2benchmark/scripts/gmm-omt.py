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
import os

from src import distributions
from src import map_benchmark

from ott.tools import sinkhorn_divergence
from ott.geometry import costs, geometry, pointcloud
import ot

import src.map_benchmark as mbm
from src.metrics import metrics_to_dict, omt_score_fitted_maps, score_fitted_fwdmaps, compute_cost, compute_BW_metrics, L2_UVP_cost, compute_w2_divergence
import matplotlib.pyplot as plt

from omt.functions import sample_gmm_kdim, points_transport, compute_mmd, transport_cost, vectorized_eogt_dist
from omt.solver import omt, shallow_gmm, ot_w2, ot_sinkh




BENCHMARK = 'Mix3toMix10'
GPU_DEVICE = 0
# torch.cuda.set_device(GPU_DEVICE)
batch_size = 1024
cost_fn = costs.SqEuclidean()


cov_types =  'full'   

test_smps = [10000]
repetitions = 10
smp_scale = 100
max_n_smp = int(1e6)
combined = False
check_outlier = False
# cov_types =  3*['diag']
# cov_types =  4*['full'] #+ 2*['diag']
K0 = 30 # 3
K1 = 50 #15
eps_omt = 0.1


d = 16
id = 0
benchmark = mbm.Mix3ToMix10Benchmark(d)
n_samples = 50000

max_iter = int(1e6)
stop_thr = 1e-6

repetitions = 10
# epss = [0.01, 0.05, 0.1, 0.2, 0.25, .5]
eps_omt = 0.1
noise_scale = [0.1, 0.25, 0.5, 1]

performance_dict = {}
performance_dict['OMT'] = pd.DataFrame(data=np.zeros((len(noise_scale), 4)), columns=['dC_mean', 'dC_std', 'dW_mean', 'dW_std'])
performance_dict['GMMOT'] = pd.DataFrame(data=np.zeros((len(noise_scale), 4)), columns=['dC_mean', 'dC_std', 'dW_mean', 'dW_std'])


X_train = benchmark.input_sampler.sample(n_samples) #+ x_noise.cuda(GPU_DEVICE)
Y_train = benchmark.output_sampler.sample(n_samples) #+ y_noise.cuda(GPU_DEVICE)
Y_train_ = Y_train.cpu().detach().numpy()

for i_n, noise in enumerate(noise_scale):
    cost_omt, cost_gmm = [], []
    w_omt, w_gmm = [], []

    for rep in range(repetitions):
        
        x_noise = torch.randn(n_samples, d) * noise
        
        if rep > 0:
            X_train_noisy = X_train + x_noise.cuda(GPU_DEVICE)
            X_train_ = X_train_noisy.cpu().detach().numpy()
        else:
            X_train_ = X_train.cpu().detach().numpy()

        gmm_s = shallow_gmm(
                        X_train_, 
                        K0, 
                        covariance_type='full', 
                        reg_covar=1e-4, 
                        max_iter=max_iter, 
                        )
        gmm_t = shallow_gmm(
                            Y_train_, 
                            K1, 
                            covariance_type='full', 
                            reg_covar=1e-4, 
                            max_iter=max_iter, 
                            )

        mean_s = gmm_s.means_
        mean_t = gmm_t.means_

        if cov_types[id] == 'diag':
            gmm_s.covariances_ = np.array([np.diag(gmm_s.covariances_[i]) for i in range(len(gmm_s.covariances_))])
            gmm_t.covariances_ = np.array([np.diag(gmm_t.covariances_[i]) for i in range(len(gmm_t.covariances_))])

        sigma_s = gmm_s.covariances_
        sigma_t = gmm_t.covariances_

        alpha_s = gmm_s.weights_
        alpha_t = gmm_t.weights_
        
        mean_s_vec = mean_s[:, np.newaxis, :]       # Shape: (K_s, 1, d)
        mean_t_vec = mean_t[np.newaxis, :, :]       # Shape: (1, K_t, d)

        sigma_s_vec = sigma_s[:, np.newaxis, ...]   # Shape: (K_s, 1, d, d)
        sigma_t_vec = sigma_t[np.newaxis, :, ...]   # Shape: (1, K_t, d, d)

        eps_gmm = eps_omt
        if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
            eps_gmm = np.full(K0 * K1, eps_gmm)
        else:
            eps_gmm = np.asarray(eps_gmm).flatten()
            
        eps_gmm_mat = eps_gmm.reshape(K0, K1) 
        C, omt_mu, omt_sigma = vectorized_eogt_dist(mean_s_vec, mean_t_vec, sigma_s_vec, sigma_t_vec, eps_gmm_mat)
        C_omt = C / C.max()


        eps_gmm = 0.0
        if not isinstance(eps_gmm, (list, tuple, np.ndarray)):
            eps_gmm = np.full(K0 * K1, eps_gmm)
        else:
            eps_gmm = np.asarray(eps_gmm).flatten()
            
        eps_gmm_mat = eps_gmm.reshape(K0, K1) 
        C, omt_mu, omt_sigma = vectorized_eogt_dist(mean_s_vec, mean_t_vec, sigma_s_vec, sigma_t_vec, eps_gmm_mat)
        C_gmm = C / C.max()

        w_, solver_log_omt = ot.bregman.sinkhorn(alpha_s, alpha_t, C_omt, reg=eps_omt, log=True, stopThr=stop_thr)
        cost_omt.append(np.sum(w_ * C_omt))
        w_omt.append(w_)

        w_, solver_log_gmm = ot.emd(alpha_s, alpha_t, C_gmm, numItermax=max_iter, log=True)
        cost_gmm.append(np.sum(w_ * C_gmm))
        w_gmm.append(w_)

        # fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        # axs[0].imshow(w_omt[-1], cmap='Blues', interpolation='nearest')
        # axs[0].set_title('OMT Transport Plan')
        # axs[1].imshow(w_gmm[-1], cmap='Blues', interpolation='nearest')
        # axs[1].set_title('GMMOT Transport Plan')
        # plt.show()
        
        

    dcost_omt_mean = np.mean(np.abs(np.diff(cost_omt) / cost_omt[0])[1:])
    dcost_omt_std = np.std(np.abs(np.diff(cost_omt) / cost_omt[0])[1:])

    dcost_gmm_mean = np.mean(np.abs(np.diff(cost_gmm) / cost_gmm[0])[1:])
    dcost_gmm_std = np.std(np.abs(np.diff(cost_gmm) / cost_gmm[0])[1:])

    print(f"---------- sigma: {noise} ------------------------------")
    print(f"OMT relative cost change mean: {dcost_omt_mean}, std: {dcost_omt_std}")
    print(f"GMMOT relative cost change mean: {dcost_gmm_mean}, std: {dcost_gmm_std}")

    dW_omt, dW_gmm = [], []
    for rep in range(1, repetitions):
        dW_omt.append(np.linalg.norm(w_omt[0] - w_omt[rep], ord='fro') / np.linalg.norm(w_omt[0], ord='fro'))
        dW_gmm.append(np.linalg.norm(w_gmm[0] - w_gmm[rep], ord='fro') / np.linalg.norm(w_gmm[0], ord='fro'))

    dw_omt_mean = np.mean(dW_omt)
    dw_omt_std = np.std(dW_omt)
    dw_gmm_mean = np.mean(dW_gmm)
    dw_gmm_std = np.std(dW_gmm)

    print(f"OMT relative W change mean: {dw_omt_mean}, std: {dw_omt_std}")
    print(f"GMMOT relative W change mean: {dw_gmm_mean}, std: {dw_gmm_std}")
    
    performance_dict['OMT'].loc[i_n, 'dC_mean'] = dcost_omt_mean
    performance_dict['GMMOT'].loc[i_n, 'dC_mean'] = dcost_gmm_mean
    performance_dict['OMT'].loc[i_n, 'dC_std']  = dcost_omt_std
    performance_dict['GMMOT'].loc[i_n, 'dC_std'] = dcost_gmm_std

    performance_dict['OMT'].loc[i_n, 'dW_mean'] = dw_omt_mean
    performance_dict['GMMOT'].loc[i_n, 'dW_mean'] = dw_gmm_mean
    performance_dict['OMT'].loc[i_n, 'dW_std']  = dw_omt_std
    performance_dict['GMMOT'].loc[i_n, 'dW_std'] = dw_gmm_std
    

performance_dict['eps'] = eps_omt
performance_dict['d'] = d
performance_dict['n_samples'] = n_samples
performance_dict['K0'] = K0
performance_dict['K1'] = K1
performance_dict['repetitions'] = repetitions

EXP_NAME = f'GMMOT_vs_OMT_Mix3toMix10_Dim_{d}_K0_{K0}_K1_{K1}_2'
save_path = f'./results/{EXP_NAME}'
os.makedirs(save_path, exist_ok=True)
np.save(f'{save_path}/summary.npy', performance_dict)