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


dim = [2, 4, 8, 16, 32, 64, 128, 256]
n_sampless = [10000, 10000, 20000, 20000, 20000, 30000, 50000, 50000]
test_smp = 10000
repetitions = 10

# torch.random.manual_seed(0xBADBEEF)
for rep in range(repetitions):
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
        CONTINUE = -1
        Num_SMP = n_sampless[id]

        config = dict(
            DIM=DIM,
            T_ITERS=T_ITERS,
            D_LR=D_LR, T_LR=T_LR,
            BATCH_SIZE=BATCH_SIZE,
            N_STEPS=N_STEPS,
            EPSILON=EPSILON,
            CONSTANT_TIME=CONSTANT_TIME,
            USE_POSITIONAL_ENCODING=USE_POSITIONAL_ENCODING,
            TIME_DIM=TIME_DIM,
            INTEGRAL_SCALE=INTEGRAL_SCALE,
            T_GRADIENT_MAX_NORM=T_GRADIENT_MAX_NORM,
            D_GRADIENT_MAX_NORM=D_GRADIENT_MAX_NORM,
            IS_RESNET_GENERATOR=IS_RESNET_GENERATOR,
            T_N_HIDDEN=T_N_HIDDEN,
            T_N_LAYERS=T_N_LAYERS,
            D_N_HIDDEN=D_N_HIDDEN,
            D_N_LAYERS=D_N_LAYERS,
            PREDICT_SHIFT=PREDICT_SHIFT
        )

        torch.cuda.set_device(GPU_DEVICE)

        # np.random.seed(OUTPUT_SEED)
        # torch.manual_seed(OUTPUT_SEED)

        torch.cuda.empty_cache()


        D = make_net(DIM, 1, n_layers=D_N_LAYERS, n_hiddens=D_N_HIDDEN).cuda()
        T = make_net(DIM+TIME_DIM, DIM, n_layers=T_N_LAYERS, n_hiddens=T_N_HIDDEN).cuda()
        T = SDE(T, EPSILON, N_STEPS, TIME_DIM, IS_RESNET_GENERATOR).cuda()
            
        print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
        print('D params:', np.sum([np.prod(p.shape) for p in D.parameters()]))
        T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
        D_opt = torch.optim.Adam(D.parameters(), lr=D_LR, weight_decay=1e-10)
        
        X_train = benchmark.input_sampler.sample(Num_SMP) 
        Y_train = benchmark.output_sampler.sample(Num_SMP) 
        
        train_data = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        metrics = []

        noise_norm = calculate_noise_norm(DIM, N_STEPS, EPSILON)
        # wandb.log({f'Noise norm' : noise_norm}, step=0)
        t0 = time.time()
        num_iter = 0
        for step in tqdm(range(CONTINUE + 1, MAX_STEPS)):
            trans_cost_train = 0
            for X_batch, Y_batch in train_loader:
                unfreeze(T); freeze(D)
                for t_iter in range(T_ITERS):
                    T_opt.zero_grad()
                    
                    X0, X1 = X_batch, Y_batch
                    # X0, X1 = benchmark.input_sampler.sample(BATCH_SIZE), benchmark.output_sampler.sample(BATCH_SIZE)
                    X0.requires_grad_()
                    
                    trajectory, times, shifts = T(X0)
                    XN = trajectory[:, -1]
                    norm = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1)**2
                    integral = INTEGRAL_SCALE*integrate(norm, times)
                    
                    T_loss = (integral + D(X1) - D(XN)).mean()
                    T_loss.backward()
                    T_gradient_norm = torch.nn.utils.clip_grad_norm_(T.parameters(), max_norm=T_GRADIENT_MAX_NORM)
                    T_opt.step()
                    num_iter += 1
            
                # wandb.log({f'T gradient norm' : T_gradient_norm.item()}, step=step)
                # wandb.log({f'Mean norm' : torch.sqrt(norm).mean().item()}, step=step)
                # wandb.log({f'T_loss' : T_loss.item()}, step=step)
                    
                del T_loss, X0, X1, XN; gc.collect(); torch.cuda.empty_cache()
                    
                freeze(T); unfreeze(D)
                
                D_opt.zero_grad()
                
                X0, X1 = X_batch, Y_batch
                # X0, X1 = benchmark.input_sampler.sample(BATCH_SIZE), benchmark.output_sampler.sample(BATCH_SIZE)
                trajectory, times, shifts = T(X0)
                XN = trajectory[:, -1]
                norm = torch.norm(shifts.flatten(start_dim=2), p=2, dim=-1)**2
                integral = INTEGRAL_SCALE*integrate(norm, times)
                
                D_X1 = D(X1)
                D_XN = D(XN)
                
                D_loss = (-integral - D_X1 + D_XN).mean()
                D_loss.backward()
                D_gradient_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=D_GRADIENT_MAX_NORM)
                D_opt.step()
                trans_cost_train += D_loss.item()
                num_iter += 1
        

        trajectory, times, shifts = T(X0)
        XN = trajectory[:, -1]
        print(XN)
        
        t1 = time.time()
        _, cpu_peak = tracemalloc.get_traced_memory()
        gpu_peak = torch.cuda.max_memory_allocated()
        tracemalloc.stop()

        # Create the results directory if it doesn't exist
        noise_scale, p_drop_out = 0., 0.
        EXP_NAME = f'series_2_{BENCHMARK}_EPSILON_{EPSILON}_ENOT_DIM_{DIM}_Epoch_{MAX_STEPS}_run_{rep}_numSmp_{Num_SMP}_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'

        save_path = f'./results/enot/{EXP_NAME}'
        os.makedirs(save_path, exist_ok=True)
        
        sde_model = save_path + f'/sde_fwd.pt'
        torch.save({'model_state_dict': T.state_dict(), 'optimizer_state_dict': T_opt.state_dict()}, sde_model)
        beta_model = save_path + f'/beta_fwd.pt'
        torch.save({'model_state_dict': D.state_dict(), 'optimizer_state_dict': D_opt.state_dict()}, beta_model)

        # Save the final metrics for visualization
        final_metrics = {
                        'time': t1 - t0,
                        'cpu_memory_peak_MB': cpu_peak / 1024**2,
                        'gpu_memory_peak_MB': gpu_peak / 1024**2,
                        'num_iter': num_iter,
                        }

        # beta_loaded_file = torch.load(save_path + f'/beta.pt', map_location='cpu')
        # D.load_state_dict(beta_loaded_file['model_state_dict'])
        # sde_loaded_file = torch.load(save_path + '/sde.pt', map_location='cpu')
        # T.load_state_dict(sde_loaded_file['model_state_dict'])
        # trajectory, times, shifts = T(X0)
        # XN = trajectory[:, -1]
        # print(XN)
        
        # pdb.set_trace()
        
        np.save(f'{save_path}/training_metrics_fwd.npy', final_metrics)

