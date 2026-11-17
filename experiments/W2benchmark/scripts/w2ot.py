import pickle
import math
import gc
import wandb
import numpy as np
import os
import pdb
import time
import tracemalloc
from tqdm import tqdm
from flax.training import checkpoints

import warnings
warnings.filterwarnings('ignore')

# --- CRITICAL GPU SETUP ---
# Ensure JAX sees the GPU and prevent Pre-allocation blocking
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# GPU_DEVICE is defined below, but we set visible devices early if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import jax
import jax.numpy as jnp
import optax
import ot
import torch
import torch.nn.functional as F
from flax import linen as nn
from flax.training import train_state
from typing import Sequence, Callable

# OTT Imports
from ott.geometry import costs, geometry, pointcloud
from ott.neural.methods import neuraldual  # <--- CHANGED
from ott.neural.networks import potentials, icnn  # <--- ADDED icnn
from ott.tools import sinkhorn_divergence
from flax.training import checkpoints

from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

# Custom src imports (Assuming these exist in your environment)
from src import distributions
from src import map_benchmark as mbm
from src.metrics import compute_cost, L2_UVP_cost
from src.icnn import DenseICNN
from src.tools import train_identity_map
from src.plotters import plot_benchmark_emb, plot_W2

from tqdm import tqdm
from functools import partial

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from utils.functions import (
    make_net, SDE, calculate_noise_norm, integrate, get_D_sigma, get_C_sigma,
    get_mu_t, get_covariance_t, get_optimal_plan_covariance,
    compute_metrics, compute_BW_UVP
)

from omt.functions_ import sample_gmm_kdim, points_transport, compute_mmd


# --------- pip install diffrax --------- (for neuraldoual)

# Force PyTorch to CPU to save GPU memory for JAX
torch.set_default_tensor_type('torch.FloatTensor')

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


# --- DATA LOADERS ---

class SourceFetcher:
    def __init__(self, manager): 
        self.manager = manager

    def __iter__(self):
        iter(self.manager)
        return self

    def __next__(self):
        return self.manager.get_source()

class TargetFetcher:
    def __init__(self, manager): 
        self.manager = manager

    def __iter__(self):
        iter(self.manager) 
        return self

    def __next__(self):
        return self.manager.get_target()


class PairedIteratorManager:
    def __init__(self, pytorch_dataloader: DataLoader):
        self.pytorch_loader = pytorch_dataloader
        if hasattr(self.pytorch_loader.dataset, '__len__') and len(self.pytorch_loader.dataset) == 0: 
            raise ValueError("PyTorch DataLoader's dataset is empty.")
        self.iterator = iter(self.pytorch_loader)
        self.current_batch_jax = None
        self.source_pending = False
        self.target_pending = False

    def _renew_iterator_and_fetch_first_batch(self):
        self.iterator = iter(self.pytorch_loader)
        try:
            source_torch, target_torch = next(self.iterator)
            return source_torch, target_torch
        except StopIteration:
            raise RuntimeError("DataLoader raised StopIteration immediately after reset.")

    def _fetch_and_prepare_batch(self):
        if not self.source_pending and not self.target_pending: 
            try:
                source_torch, target_torch = next(self.iterator)
            except StopIteration: 
                source_torch, target_torch = self._renew_iterator_and_fetch_first_batch()
            
            # Convert to JAX arrays (Push to GPU here)
            source_np = source_torch.detach().cpu().numpy()
            target_np = target_torch.detach().cpu().numpy()
            self.current_batch_jax = (jax.device_put(source_np), jax.device_put(target_np))
            self.source_pending = True
            self.target_pending = True

    def get_source(self):
        self._fetch_and_prepare_batch()
        if not self.source_pending or self.current_batch_jax is None:
            raise RuntimeError("Source not available.")
        source_data = self.current_batch_jax[0]
        self.source_pending = False 
        return source_data

    def get_target(self):
        if self.current_batch_jax is None or not self.target_pending or self.source_pending:
            raise RuntimeError("Target requested in an invalid state.")
        target_data = self.current_batch_jax[1]
        self.target_pending = False 
        return target_data

    def __iter__(self):
        self.iterator = iter(self.pytorch_loader)
        self.current_batch_jax = None
        self.source_pending = False
        self.target_pending = False
        return self


class PotentialMLP(nn.Module):
    dim_hidden: Sequence[int]
    act_fn: Callable = nn.elu

    @nn.compact
    def __call__(self, x):
        for feat in self.dim_hidden[:-1]:
            x = nn.Dense(feat)(x)
            x = self.act_fn(x)
        # Final layer outputs scalar (dim 1), usually linear for potentials
        x = nn.Dense(self.dim_hidden[-1])(x)
        return x

    # --- THIS IS THE MISSING METHOD ---
    def create_train_state(self, rng, optimizer, input_dim):
        """Creates the initial TrainState."""
        # Initialize the parameters using a dummy input
        dummy_input = jnp.ones((1, input_dim))
        params = self.init(rng, dummy_input)['params']
        
        # Create and return the TrainState
        return train_state.TrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer
        )

# --- CONFIGURATION ---

VERSION = "v2"
GPU_DEVICE = 0
BATCH_SIZE = 1024
EPSILON = 0.001
BENCHMARK = 'Mix3toMix10'

# Ensure this matches the device JAX uses
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)

dim = [128, 256] # [2, 4, 8, 16, 32, 64, 128, 256]
n_sampless = [50000, 50000] # [10000, 10000, 20000, 20000, 20000, 30000, 50000, 50000]
max_iters = [50000, 50000] #[10000, 10000, 25000, 25000, 50000, 50000, int(1e5), int(1e5)]
test_smp = 10000
repetitions = 5

print(f"JAX Devices: {jax.devices()}")

# --- MAIN LOOP ---

for rep in range(repetitions):
    for id, DIM in enumerate(dim):
            print(f"Running {rep} for dim {DIM}")
            tracemalloc.start()
            benchmark = mbm.Mix3ToMix10Benchmark(DIM)
            
            Num_SMP = n_sampless[id]
            max_iter = max_iters[id]

            # Define Potentials (MLP or ICNN)
            # Note: W2NeuralDual works best if potentials are close to convex (ICNN), 
            # but standard MLPs are allowed.
            # initialize models using identity initialization (default)
            
            # older version with ICNNs:
            # neural_f = icnn.ICNN(dim_hidden=[64, 64, 64, 64], dim_data=2, init_fn=nn.initializers.normal())
            # neural_g = icnn.ICNN(dim_hidden=[64, 64, 64, 64], dim_data=2, init_fn=nn.initializers.normal())
            # lr_schedule_f = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            # lr_schedule_g = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            # optimizer_f = optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9, eps=1e-8)
            # optimizer_g = optax.adam(learning_rate=1e-4, b1=0.5, b2=0.9, eps=1e-8)
            
            # neural_f = potentials.MLP(dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu)
            # neural_g = potentials.MLP(dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu)
            # lr_schedule_f = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            # lr_schedule_g = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            # optimizer_f = optax.adam(learning_rate=lr_schedule_f, b1=0.9, b2=0.999)
            # optimizer_g = optax.adam(learning_rate=lr_schedule_g, b1=0.9, b2=0.999)
            
            # 1. Architecture Configuration for PotentialNN
            # Table spec: [max(2D, 64), max(2D, 64), max(D, 32)]
            # Note: We append '1' at the end because potentials must output a scalar.
            hidden_dims = [128, 128, 128, 128, 1]

            # 2. Hyperparameters
            # max_iter = 250000       # Table: # training iterations
            lr_val = 5e-4           # Table: initial learning rate
            end_lr_factor = 1e-4    # Table: cosine annealing alpha
            adam_betas = (0.5, 0.5) # Table: Adam beta

            # 3. Model Definitions
            # Applied to both f and g assuming symmetric dual potentials
            neural_f = potentials.PotentialMLP(dim_hidden=hidden_dims, act_fn=jax.nn.elu)
            neural_g = potentials.PotentialMLP(dim_hidden=hidden_dims, act_fn=jax.nn.elu)


            # 4. Learning Rate Schedules
            lr_schedule_f = optax.cosine_decay_schedule(
                init_value=lr_val, 
                decay_steps=max_iter, 
                alpha=end_lr_factor
            )
            lr_schedule_g = optax.cosine_decay_schedule(
                init_value=lr_val, 
                decay_steps=max_iter, 
                alpha=end_lr_factor
            )

            # 5. Optimizers
            optimizer_f = optax.adam(
                learning_rate=lr_schedule_f, 
                b1=adam_betas[0], 
                b2=adam_betas[1]
            )
            optimizer_g = optax.adam(
                learning_rate=lr_schedule_g, 
                b1=adam_betas[0], 
                b2=adam_betas[1]
            )
            
            # Data Generation (Using Torch on CPU)
            X_train_pt = benchmark.input_sampler.sample(Num_SMP)
            Y_train_pt = benchmark.output_sampler.sample(Num_SMP)
            X_valid_pt = benchmark.input_sampler.sample(Num_SMP) 
            Y_valid_pt = benchmark.output_sampler.sample(Num_SMP)

            train_data = TensorDataset(X_train_pt, Y_train_pt)
            valid_data = TensorDataset(X_valid_pt, Y_valid_pt)
            
            # Workers=0 ensures data stays in main process until pushed to GPU by Manager
            train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
            valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Create managed iterators
            train_manager = PairedIteratorManager(train_dataloader)
            train_src_fetcher = SourceFetcher(train_manager)
            train_tgt_fetcher = TargetFetcher(train_manager)

            valid_manager = PairedIteratorManager(valid_dataloader)
            valid_src_fetcher = SourceFetcher(valid_manager)
            valid_tgt_fetcher = TargetFetcher(valid_manager)

            t0 = time.time()
            
            # --- UPDATED SOLVER: W2NeuralDual ---
            neural_dual_solver = neuraldual.W2NeuralDual(
                dim_data=DIM,
                neural_f=neural_f,
                neural_g=neural_g,
                optimizer_f=optimizer_f,
                optimizer_g=optimizer_g,
                num_train_iters=max_iter,
                amortization_loss='regression',
            )
            
            print(f"Starting training for DIM={DIM} with {max_iter} iterations...")
            
            # Train
            learned_potentials = neural_dual_solver(
                train_src_fetcher,
                train_tgt_fetcher,
                valid_src_fetcher,
                valid_tgt_fetcher
            )
            
            # Ensure completion
            # jax.block_until_ready(learned_potentials.f.params)
            
            t1 = time.time()            
            print("Training finished.")
            _, cpu_peak = tracemalloc.get_traced_memory()
            
            # GPU Peak Memory (Torch only tracks its own, need nvidia-smi for JAX generally, 
            # but we keep the placeholder)
            gpu_peak = torch.cuda.max_memory_allocated() 
            tracemalloc.stop()
            
            trans_cost_train = learned_potentials.distance(X_train_pt.cpu().detach().numpy(), Y_train_pt.cpu().detach().numpy())
                
            for noise_scale, p_drop_out in [(0. , 0.)]: #, (.1, 0.), (.25, 0.), (.5, 0.), (1., 0.), (0., 0.05), (0., .1), (0., .25), (0., .5)]:
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
                
                y_ = np.array(learned_potentials.transport(X_test_, forward=True))
                
                if noise_scale == 0 and p_drop_out == 0:
                    x_ = np.array(learned_potentials.transport(Y_test_, forward=False))
                    xx_ = np.array(learned_potentials.transport(y_, forward=False)) 
                    yy_ = np.array(learned_potentials.transport(x_, forward=True))

                    Y_push = torch.tensor(x_, dtype=torch.float32).cuda(GPU_DEVICE)
                    YY_push = torch.tensor(xx_, dtype=torch.float32).cuda(GPU_DEVICE)
                    XX_push = torch.tensor(yy_, dtype=torch.float32).cuda(GPU_DEVICE)

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

                X_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)
                mmd_T = compute_mmd(X=Y_test_, Y=y_) * 100
                mmd_T_noise = compute_mmd(X=Y_test_noise_, Y=y_) * 100
                
                L2_T = np.mean(np.linalg.norm(Y_test_ - y_, ord=2, axis=1))
                L2_T_noise = np.mean(np.linalg.norm(Y_test_noise_ - y_, ord=2, axis=1)) 
                
                L2_UVP, cos_sim = L2_UVP_cost(benchmark, X_test_noise, Y_test, X_push)
                L2_UVP_noise, cos_sim_noise = L2_UVP_cost(benchmark, X_test_noise, Y_test_noise, X_push)
                
                dist = sinkh_div(y_, Y_test_, batch_size=1024, epsilon=0.001) 
                dist_noise = sinkh_div(y_, Y_test_noise_, batch_size=1024, epsilon=0.001)
                
                trans_cost_test = learned_potentials.distance(X_test_, Y_test_)
                
                print(f"Costs for noise scale {noise_scale} and dropout {p_drop_out}: {mmd_T}, {L2_T}, {L2_UVP}, {dist}")
                    
                # Updated EXP_NAME to reflect W2OT
                EXP_NAME = f'series_2_{BENCHMARK}_W2OT_MLP_DIM_{DIM}_Epoch_{max_iter}_run_{rep}_numSmp_{Num_SMP}_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'
                save_path = f'./results/w2ot/{EXP_NAME}'
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
                    'cost_train': trans_cost_train,
                    'cost': trans_cost_test,
                    'time': t1 - t0,
                    'cpu_memory_peak_MB': cpu_peak / 1024**2,
                    'gpu_memory_peak_MB': gpu_peak / 1024**2,
                    'num_iter': max_iter,
                }
                np.save(f'{save_path}/metrics.npy', final_metrics)
                
                # Clean JAX cache to avoid OOM in loop
                jax.clear_caches()
                gc.collect()