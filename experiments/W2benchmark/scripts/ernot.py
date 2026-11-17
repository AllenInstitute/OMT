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
import jax
import jax.numpy as jnp
import optax
import ot
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from ott.geometry import costs, geometry, pointcloud
from ott.neural.methods import expectile_neural_dual
from ott.neural.networks import potentials
from ott.tools import sinkhorn_divergence
from ott.neural.methods.expectile_neural_dual import ENOTPotentials
from flax.training import checkpoints

from scipy.stats import ortho_group
from scipy.linalg import sqrtm, inv
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA

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

# Removed original JaxBatchIterator as it's replaced by the manager/fetcher system.


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



class SourceFetcher:
    def __init__(self, manager): # Use PairedIteratorManager type hint if available
        self.manager = manager

    def __iter__(self):
        iter(self.manager) # Calls PairedIteratorManager.__iter__ to reset it
        return self

    def __next__(self):
        return self.manager.get_source()

# Definition of TargetFetcher:
class TargetFetcher:
    def __init__(self, manager): # Use PairedIteratorManager type hint if available
        self.manager = manager

    def __iter__(self):
        iter(self.manager) # Resets the manager
        return self

    def __next__(self):
        return self.manager.get_target()


class PairedIteratorManager:
    def __init__(self, pytorch_dataloader: DataLoader):
        self.pytorch_loader = pytorch_dataloader
        # Basic check for an entirely empty DataLoader based on dataset length
        if hasattr(self.pytorch_loader.dataset, '__len__') and len(self.pytorch_loader.dataset) == 0: # type: ignore
            raise ValueError(
                "PyTorch DataLoader's dataset is empty (0 samples). "
                "Cannot initialize PairedIteratorManager."
            )
        self.iterator = iter(self.pytorch_loader)
        self.current_batch_jax = None
        self.source_pending = False
        self.target_pending = False

    def _renew_iterator_and_fetch_first_batch(self):
        """Resets the PyTorch iterator and fetches the first batch of a new epoch."""
        # print("PairedIteratorManager: Renewing iterator and fetching first batch.") # For debugging
        self.iterator = iter(self.pytorch_loader)
        try:
            source_torch, target_torch = next(self.iterator)
            return source_torch, target_torch
        except StopIteration:
            # This should only happen if the dataloader is truly empty or
            # became empty, which would be an issue with the DataLoader setup itself
            # if the initial check passed.
            error_msg = (
                "PairedIteratorManager: DataLoader raised StopIteration "
                "immediately after reset. This implies the DataLoader is "
                "fundamentally empty or yields no data."
            )
            print(f"ERROR: {error_msg}") # Keep print for visibility of this critical error
            raise RuntimeError(error_msg)


    def _fetch_and_prepare_batch(self):
        """
        Ensures a JAX batch is loaded and pending. If the current PyTorch iterator
        is exhausted, it renews the iterator (starts a new epoch) and loads
        the first batch of that new epoch. This method is designed to always
        provide a batch by looping over the dataset, assuming the dataset is not empty.
        Sets self.current_batch_jax, self.source_pending, self.target_pending.
        """
        if not self.source_pending and not self.target_pending:  # Need a new batch pair
            try:
                source_torch, target_torch = next(self.iterator)
            except StopIteration:  # End of current epoch
                # print("PairedIteratorManager: End of epoch, renewing iterator.") # For debugging
                source_torch, target_torch = self._renew_iterator_and_fetch_first_batch()
            
            # Convert to JAX arrays
            source_np = source_torch.detach().cpu().numpy()
            target_np = target_torch.detach().cpu().numpy()
            self.current_batch_jax = (jax.device_put(source_np), jax.device_put(target_np))
            self.source_pending = True
            self.target_pending = True

    def get_source(self):
        self._fetch_and_prepare_batch()  # Ensures a batch is loaded/reloaded and pending
        
        if not self.source_pending or self.current_batch_jax is None:
            # This state should ideally not be reached if _fetch_and_prepare_batch works correctly.
            raise RuntimeError("PairedIteratorManager: Source not available or batch not loaded after _fetch_and_prepare_batch.")
        
        source_data = self.current_batch_jax[0]
        self.source_pending = False  # Mark source as consumed for this batch
        return source_data

    def get_target(self):
        # _fetch_and_prepare_batch() is NOT called here.
        # It's assumed get_source() was called first for the current batch,
        # which would have called _fetch_and_prepare_batch().
        
        if self.current_batch_jax is None or not self.target_pending or self.source_pending:
            # If source_pending is True, it means get_source() was not called for this batch.
            # If target_pending is False, it means target was already consumed or not ready.
            # If current_batch_jax is None, something went wrong in fetching.
            raise RuntimeError(
                "PairedIteratorManager: Target requested in an invalid state. "
                f"Current batch is {'set' if self.current_batch_jax is not None else 'None'}, "
                f"Target pending: {self.target_pending}, Source pending: {self.source_pending}."
            )

        target_data = self.current_batch_jax[1]
        self.target_pending = False  # Mark target as consumed for this batch
        return target_data

    def __iter__(self):
        """
        Allows the iterator to be reset externally, e.g., if a different training
        loop structure were to explicitly call iter() on the loader for each epoch.
        """
        # print("PairedIteratorManager: __iter__ called, resetting DataLoader iterator state.") # For debugging
        self.iterator = iter(self.pytorch_loader)
        self.current_batch_jax = None
        self.source_pending = False
        self.target_pending = False
        return self

VERSION = "v2" # v1
GPU_DEVICE = 0
BATCH_SIZE = 256

# OUTPUT_SEED = 0xC0FFEE
L1 = 1e-10
# GPU_DEVICE = 0 # Already defined
EPSILON = 0.001
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
# GPU_DEVICE = 0 # Already defined
torch.cuda.set_device(GPU_DEVICE)


LR = 1e-3
eps = 0.01

max_iter = int(1e5) # This will be num_train_iters for the solver
# torch.random.manual_seed(0xBADBEEF) # Consider if seeding is needed for reproducibility

dim =  [2, 4, 8, 16, 32, 64, 128, 256]
n_sampless = [10000, 10000, 20000, 20000, 20000, 30000, 50000, 50000]
max_iters = [10000, 10000, 25000, 25000, 50000, 50000, int(1e5), int(1e5)]
test_smp = 10000
repetitions = 10

for rep in range(repetitions):
    for id, DIM in enumerate(dim):
            print(f"Running {rep} for dim {DIM}")
            tracemalloc.start()
            benchmark = mbm.Mix3ToMix10Benchmark(DIM)
            
            Num_SMP = n_sampless[id]
            max_iter = max_iters[id]

            neural_f = potentials.MLP(dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu)
            neural_g = potentials.MLP(dim_hidden=[128, 128, 128, 128, 1], act_fn=jax.nn.elu)
            lr_schedule_f = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            lr_schedule_g = optax.cosine_decay_schedule(init_value=5e-4, decay_steps=max_iter, alpha=1e-2)
            optimizer_f = optax.adam(learning_rate=lr_schedule_f, b1=0.9, b2=0.999)
            optimizer_g = optax.adam(learning_rate=lr_schedule_g, b1=0.9, b2=0.999)
            
            
            # x_noise = torch.randn(Num_SMP, DIM) * noise_scale**2
            # y_noise = torch.randn(Num_SMP, DIM) * noise_scale**2
            
            X_train_pt = benchmark.input_sampler.sample(Num_SMP) #+ x_noise.cuda(GPU_DEVICE)
            Y_train_pt = benchmark.output_sampler.sample(Num_SMP) #+ y_noise.cuda(GPU_DEVICE)

            # X_train_pt = F.dropout(X_train_pt, p=p_drop_out, training=True)                    
            # Y_train_pt = F.dropout(Y_train_pt, p=p_drop_out, training=True)
                
            X_valid_pt = benchmark.input_sampler.sample(Num_SMP) 
            Y_valid_pt = benchmark.output_sampler.sample(Num_SMP)

            train_data = TensorDataset(X_train_pt, Y_train_pt)
            valid_data = TensorDataset(X_valid_pt, Y_valid_pt)
            train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
            valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Create managed iterators for the neural dual solver
            train_manager = PairedIteratorManager(train_dataloader)
            train_src_fetcher = SourceFetcher(train_manager)
            train_tgt_fetcher = TargetFetcher(train_manager)

            valid_manager = PairedIteratorManager(valid_dataloader)
            valid_src_fetcher = SourceFetcher(valid_manager)
            valid_tgt_fetcher = TargetFetcher(valid_manager)

            t0 = time.time()
            neural_dual_solver = expectile_neural_dual.ExpectileNeuralDual(
                                                                            DIM, # Corrected: dim parameter should be passed
                                                                            neural_f=neural_f, # Corrected: pass neural_f
                                                                            neural_g=neural_g, # Corrected: pass neural_g
                                                                            optimizer_f=optimizer_f, # Corrected: pass optimizer_f
                                                                            optimizer_g=optimizer_g, # Corrected: pass optimizer_g
                                                                            cost_fn=costs.SqEuclidean(),
                                                                            num_train_iters=max_iter,
                                                                            expectile=0.98,
                                                                            expectile_loss_coef=0.25,
                                                                            is_bidirectional=True,
                                                                            use_dot_product=False,
                                                                            )
            
            print(f"Starting training for DIM={DIM} with {max_iter} iterations...")
            learned_potentials = neural_dual_solver(
                                                    train_src_fetcher,
                                                    train_tgt_fetcher,
                                                    valid_src_fetcher,
                                                    valid_tgt_fetcher,
                                                    # logger_fn=your_logger_function if you have one
                                                    )
            t1 = time.time()            
            print("Training finished.")
            _, cpu_peak = tracemalloc.get_traced_memory()
            gpu_peak = torch.cuda.max_memory_allocated()
            tracemalloc.stop()
            
            trans_cost_train = learned_potentials.distance(X_train_pt.cpu().detach().numpy(), Y_train_pt.cpu().detach().numpy())
                
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
                
                y_ = learned_potentials.transport(X_test_, forward=True)
                
                if noise_scale == 0 and p_drop_out == 0:
                    x_ = learned_potentials.transport(Y_test_, forward=False) 
                    xx_ = learned_potentials.transport(y_, forward=False) 
                    yy_ = learned_potentials.transport(x_, forward=True)

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
                    
                EXP_NAME = f'series_2_{BENCHMARK}_ErNOT_DIM_{DIM}_Epoch_{max_iter}_run_{rep}_numSmp_{Num_SMP}_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'
                save_path = f'./results/ernot/{EXP_NAME}'
                os.makedirs(save_path, exist_ok=True)

                # pdb.set_trace()
                # with open(save_path + '/potentials.pkl', 'wb') as f:
                #     pickle.dump(learned_potentials, f)

                # with open(save_path + '/potentials.pkl', 'rb') as f:
                #     learned_potentials_ = pickle.load(f)

                # pdb.set_trace()
                # trans_cost_test_ = learned_potentials_.distance(X_test_, Y_test_)
                # print(trans_cost_test, trans_cost_test_)
                # pdb.set_trace()

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


