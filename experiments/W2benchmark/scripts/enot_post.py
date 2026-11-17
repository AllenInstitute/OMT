import matplotlib
import math
import gc
import wandb
import numpy as np
import os
import time
import pdb
import tracemalloc
from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
from ott.geometry import costs, geometry, pointcloud
from ott.neural.networks import potentials
from ott.tools import sinkhorn_divergence

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from torch.utils.data import DataLoader, TensorDataset

# from src.icnn import DenseICNN
# from src.tools import compute_l1_norm, ewma
from src_2.fid_score import calculate_frechet_distance
from src_2.tools import unfreeze, freeze
from sklearn.decomposition import PCA

from src import distributions
from src import map_benchmark as mbm
from src.metrics import compute_cost, L2_UVP_cost

from src.icnn import DenseICNN
from src.tools import train_identity_map
from src.plotters import plot_benchmark_emb, plot_W2

from tqdm import tqdm


from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from utils.functions import (
    make_net, SDE, calculate_noise_norm, integrate, get_D_sigma, get_C_sigma,
    get_mu_t, get_covariance_t, get_optimal_plan_covariance,
    compute_metrics, compute_BW_UVP
)

from omt.functions_ import sample_gmm_kdim, points_transport, compute_mmd

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


def sinkh_div(x: jnp.ndarray, y: jnp.ndarray, batch_size: int, epsilon: float = 0.001) -> float:
    len_x = x.shape[0]
    cost = []
    for i in range(0, len_x, batch_size):
        end_idx = min(i + batch_size, len_x)
        x_batch = x[i:end_idx]
        y_batch = y[i:end_idx]
        cost.append(sinkhorn_loss(x_batch, y_batch, epsilon=epsilon))

    return np.mean(cost)



VERSION = "v2" # v1
GPU_DEVICE = 0
BATCH_SIZE = 1024

# OUTPUT_SEED = 0xC0FFEE
L1 = 1e-10
GPU_DEVICE = 0
EPSILON = 0.01
N_STEPS = 50
TIME_DIM = 1
T_LR = 3e-4
D_LR = 3e-4
T_ITERS = 10
CONSTANT_TIME = False
USE_POSITIONAL_ENCODING = False

T_GRADIENT_MAX_NORM = float("inf")
D_GRADIENT_MAX_NORM = float("inf")
IS_RESNET_GENERATOR = False
PREDICT_SHIFT = True
N_LAST_STEPS_WITHOUT_NOISE = 1

T_N_HIDDEN = 512
T_N_LAYERS = 3

D_N_HIDDEN = 512
D_N_LAYERS = 3

BENCHMARK = 'Mix3toMix10'

LR = 1e-3


OUTPUT_PATH = '../benchmarks/' + BENCHMARK
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


dim = [128, 256] #[2, 4, 8, 16, 32, 64, 128, 256]
n_sampless = [50000, 50000] #[10000, 10000, 20000, 20000, 20000, 30000, 50000, 50000]
test_smp = 10000
repetitions = 10

# torch.random.manual_seed(0xBADBEEF)
for rep in range(4, 5):
    for id, DIM in enumerate(dim):
        print(f"------------ Running {rep} for dim {DIM} ----------------")
        tracemalloc.start()
        
        INTEGRAL_SCALE = 1/(DIM)
        LAMBDA = DIM # usually works like a charm
        D_HYPERPARAMS = {
                        'dim' : DIM,
                        'rank' : 1,
                        'hidden_layer_sizes' : [max(2*DIM, 64), max(2*DIM, 64), max(DIM, 32)],
                        'strong_convexity' : 1e-4
                        }

        benchmark = mbm.Mix3ToMix10Benchmark(DIM)
        
        MAX_STEPS = 500 # int(10000 / (Num_SMP / BATCH_SIZE))
        Num_SMP = n_sampless[id]

        torch.cuda.empty_cache()

        D_fwd = make_net(DIM, 1, n_layers=D_N_LAYERS, n_hiddens=D_N_HIDDEN).cuda()
        T_fwd = make_net(DIM+TIME_DIM, DIM, n_layers=T_N_LAYERS, n_hiddens=T_N_HIDDEN).cuda()
        T_fwd = SDE(T_fwd, EPSILON, N_STEPS, TIME_DIM, IS_RESNET_GENERATOR).cuda()

        D_inv = make_net(DIM, 1, n_layers=D_N_LAYERS, n_hiddens=D_N_HIDDEN).cuda()
        T_inv = make_net(DIM+TIME_DIM, DIM, n_layers=T_N_LAYERS, n_hiddens=T_N_HIDDEN).cuda()
        T_inv = SDE(T_inv, EPSILON, N_STEPS, TIME_DIM, IS_RESNET_GENERATOR).cuda()

        # Create the results directory if it doesn't exist
        origin_name = EXP_NAME = f'series_2_{BENCHMARK}_EPSILON_{EPSILON}_ENOT_DIM_{DIM}_Epoch_{MAX_STEPS}_run_{rep}_numSmp_{Num_SMP}_noise_Gauss_0.0_p_drop_0.0'
        master_path = f'./results/enot/{origin_name}'
        training_matrics_fwd = np.load(f'{master_path}/training_metrics_fwd.npy', allow_pickle=True).item()
        training_matrics_inv = np.load(f'{master_path}/training_metrics_inv.npy', allow_pickle=True).item()
            

        beta_loaded_file = torch.load(master_path + f'/beta_fwd.pt', map_location='cpu')
        D_fwd.load_state_dict(beta_loaded_file['model_state_dict'])
        sde_loaded_file = torch.load(master_path + '/sde_fwd.pt', map_location='cpu')
        T_fwd.load_state_dict(sde_loaded_file['model_state_dict'])
        
        beta_loaded_file = torch.load(master_path + f'/beta_inv.pt', map_location='cpu')
        D_inv.load_state_dict(beta_loaded_file['model_state_dict'])
        sde_loaded_file = torch.load(master_path + '/sde_inv.pt', map_location='cpu')
        T_inv.load_state_dict(sde_loaded_file['model_state_dict'])

        for noise_scale, p_drop_out in [(0. , 0.), (.1, 0.), (.25, 0.), (.5, 0.), (1., 0.), (0., 0.05), (0., .1), (0., .25), (0., .5)]:
            x_noise = torch.randn(test_smp, DIM) * noise_scale
            X_test = benchmark.input_sampler.sample(test_smp) 
            X_test.requires_grad_(True)
            Y_test = benchmark.map_fwd(X_test, nograd=True) 
            X_test_noise = X_test + x_noise.cuda(GPU_DEVICE)
            X_test_noise = F.dropout(X_test_noise, p=p_drop_out, training=True)
            Y_test_noise = benchmark.map_fwd(X_test_noise, nograd=True)
            
            X_test_ = X_test.cpu().detach().numpy()
            X_test_noise_ = X_test_noise.cpu().detach().numpy()
            Y_test_ = Y_test.cpu().detach().numpy()
            Y_test_noise_ = Y_test_noise.cpu().detach().numpy()

            trajectory, times, shifts = T_fwd(X_test)
            y_ = trajectory[:, -1].cpu().detach()
            
            norm = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1)**2
            integral = INTEGRAL_SCALE*integrate(norm, times)
            D_X1 = D_fwd(X_test)
            D_XN = D_fwd(y_.to(GPU_DEVICE))
            trans_cost_test_fwd = np.mean(np.linalg.norm(X_test_ - y_.numpy(), ord=2, axis=1)) 
            trans_cost_test_fwd2 = (-integral - D_X1 + D_XN).mean().item()
            del trajectory, times, shifts, D_X1, D_XN, 
            torch.cuda.empty_cache()
            
            if noise_scale == 0 and p_drop_out == 0:
                trajectory, times, shifts = T_inv(Y_test)
                x_ = trajectory[:, -1].cpu().detach()
                norm = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1)**2
                integral = INTEGRAL_SCALE*integrate(norm, times)
                D_X1 = D_inv(Y_test)
                D_XN = D_inv(x_.to(GPU_DEVICE))
                trans_cost_test_inv = np.mean(np.linalg.norm(Y_test_ - x_.numpy(), ord=2, axis=1)) 
                trans_cost_test_inv2 = (-integral - D_X1 + D_XN).mean().item()
                del trajectory, times, shifts, D_X1, D_XN
                torch.cuda.empty_cache()
                
                trajectory, times, shifts = T_inv(y_.to(GPU_DEVICE))
                xx_ = trajectory[:, -1].cpu().detach()
                del trajectory, times, shifts
                torch.cuda.empty_cache()
                
                trajectory, times, shifts = T_fwd(x_.to(GPU_DEVICE))
                yy_ = trajectory[:, -1].cpu().detach()
                del trajectory, times, shifts
                torch.cuda.empty_cache()

                Y_push = x_.cuda(GPU_DEVICE)
                YY_push = xx_.cuda(GPU_DEVICE)
                XX_push = yy_.cuda(GPU_DEVICE)

                x_ = x_.numpy()
                xx_ = xx_.numpy()
                yy_ = yy_.numpy()

                mmd_T_inv = compute_mmd(X=X_test_, Y=x_) * 100
                mmd_T_inv_2 = compute_mmd(X=X_test_, Y=xx_) * 100
                mmd_T_2 = compute_mmd(X=Y_test_, Y=yy_) * 100

                L2_T_inv = np.mean(np.linalg.norm(X_test_ - x_, ord=2, axis=1))
                L2_T_inv_2 = np.mean(np.linalg.norm(X_test_ - xx_, ord=2, axis=1))
                L2_T_2 = np.mean(np.linalg.norm(Y_test_ - yy_, ord=2, axis=1))

                L2_UVP_inv, cos_sim_inv = L2_UVP_cost(benchmark, Y_test, X_test, Y_push, direction='inv')
                L2_UVP_inv_2, cos_sim_inv_2 = L2_UVP_cost(benchmark, Y_test, X_test, YY_push, direction='inv')
                L2_UVP_2, cos_sim_2 = L2_UVP_cost(benchmark, X_test, Y_test, XX_push)
                
                dist_inv = sinkh_div(x_, X_test_, batch_size=1024, epsilon=0.001)
                dist_inv_2 = sinkh_div(xx_, X_test_, batch_size=1024, epsilon=0.001)
                dist_2 = sinkh_div(yy_, Y_test_, batch_size=1024, epsilon=0.001) 

                print(f"Costs Inv: {mmd_T_inv}, {L2_T_inv}, {L2_UVP_inv}, {dist_inv}")
                print(f"Costs Inv 2: {mmd_T_inv_2}, {L2_T_inv_2}, {L2_UVP_inv_2}, {dist_inv_2}") 
                print(f"Costs 2: {mmd_T_2}, {L2_T_2}, {L2_UVP_2}, {dist_2}")
            else:
                mmd_T_inv, mmd_T_inv_2 = 0., 0.
                L2_T_inv, L2_T_inv_2 = 0., 0.
                L2_UVP_inv, L2_UVP_inv_2 = 0., 0.
                cos_sim_inv, cos_sim_inv_2 = 0., 0.
                dist_inv, dist_inv_2 = 0., 0.

            y_ = y_.numpy()
            X_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)
            mmd_T = compute_mmd(X=Y_test_, Y=y_) * 100
            mmd_T_noise = compute_mmd(X=Y_test_noise_, Y=y_) * 100
            
            L2_T = np.mean(np.linalg.norm(Y_test_ - y_, ord=2, axis=1))
            L2_T_noise = np.mean(np.linalg.norm(Y_test_noise_ - y_, ord=2, axis=1)) 
            
            L2_UVP, cos_sim = L2_UVP_cost(benchmark, X_test_noise, Y_test, X_push)
            L2_UVP_noise, cos_sim_noise = L2_UVP_cost(benchmark, X_test_noise, Y_test_noise, X_push)
            
            dist = sinkh_div(y_, Y_test_, batch_size=1024, epsilon=0.001) 
            dist_noise = sinkh_div(y_, Y_test_noise_, batch_size=1024, epsilon=0.001)
            
            print(f"Costs for noise scale {noise_scale} and dropout {p_drop_out}: {mmd_T}, {L2_T}, {L2_UVP}, {dist}")
                
            EXP_NAME = f'series_2_{BENCHMARK}_EPSILON_{EPSILON}_ENOT_DIM_{DIM}_Epoch_{MAX_STEPS}_run_{rep}_numSmp_{Num_SMP}_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'
            save_path = f'./results/enot/{EXP_NAME}'
            os.makedirs(save_path, exist_ok=True)

            final_metrics = {
                'mmd': [mmd_T, mmd_T_2],
                'mmd_noise': mmd_T_noise,
                'mmd_inv': [mmd_T_inv, mmd_T_inv_2],
                'sinkh_div_baseline': 0.,
                'sinkh_div_trans': 0.,
                'sinkh_div_y': [dist, dist_2],
                'sinkh_div_y_noise': dist_noise,
                'sinkh_div_x': [dist_inv, dist_inv_2],
                'L2_dist': [L2_T, L2_T_2],
                'L2_dist_inv': [L2_T_inv, L2_T_inv_2],
                'L2_dist_noise': L2_T_noise,
                'L2_UVP': [L2_UVP, L2_UVP_2],
                'L2_UVP_noise': L2_UVP_noise, 
                'L2_UVP_inv': [L2_UVP_inv, L2_UVP_inv_2],
                'cos': [cos_sim, cos_sim_2],
                'cos_noise': cos_sim_noise,
                'cos_inv': [cos_sim_inv, cos_sim_inv_2],
                'cost_train': 0.,
                'cost': [trans_cost_test_fwd, trans_cost_test_inv],
                'cost2': [trans_cost_test_fwd2, trans_cost_test_inv2],
                'time': training_matrics_fwd['time'] + training_matrics_inv['time'],
                'cpu_memory_peak_MB': training_matrics_fwd['cpu_memory_peak_MB'] + training_matrics_inv['cpu_memory_peak_MB'],
                'gpu_memory_peak_MB': training_matrics_fwd['gpu_memory_peak_MB'] + training_matrics_inv['gpu_memory_peak_MB'],
                'num_iter': training_matrics_fwd['num_iter'] + training_matrics_inv['num_iter'],
            }
            
            np.save(f'{save_path}/metrics.npy', final_metrics)
