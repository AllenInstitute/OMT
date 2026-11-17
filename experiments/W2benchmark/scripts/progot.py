import os, sys
sys.path.append("..")
import numpy as np
import time
import pdb
import torch
from typing import Any, Optional
import torch.nn.functional as F
import tracemalloc
import jax
import jax.numpy as jnp

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem, potentials
from ott.solvers import linear
from ott.tools import progot

from ott.tools import sinkhorn_divergence

from src.plotters import plot_benchmark_emb, plot_W2, plot_benchmark_metrics
import src.map_benchmark as mbm
from src.metrics import score_fitted_fwdmaps, compute_cost, compute_BW_metrics, L2_UVP_cost

from omt.functions import sample_gmm_kdim, compute_mmd
from omt.solver import omt


def run_progot(x: jnp.ndarray, y: jnp.ndarray, cost_fn: costs.TICost, **kwargs: Any) -> progot.ProgOTOutput:
    geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
    prob = linear_problem.LinearProblem(geom)
    estim = progot.ProgOT(**kwargs)
    out = estim(prob)
    return out


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
stop_thr = 1e-11
fontsize = 16

BENCHMARK = 'Mix3toMix10'
GPU_DEVICE = 0
torch.cuda.set_device(GPU_DEVICE)
batch_size = 1024

dim = [2, 4, 8, 16, 32, 64, 128, 256]
n_sampless = [10000, 10000, 10000, 20000, 20000, 20000, 20000, 20000]
test_smp = 10000
repetitions = 10

cost_fn = costs.SqEuclidean()
K = 4

alphas = progot.get_alpha_schedule("exp", num_steps=K)
end_epsilon_scales = jnp.logspace(-10, -9, 4, base=2)

for rep in range(4, repetitions):
    for id, d in enumerate(dim):
        print(f"Running {rep} for dim {d}")
        tracemalloc.start()
                            
        benchmark = mbm.Mix3ToMix10Benchmark(d)
        X = benchmark.input_sampler.sample(int(1e6))
        X = X.cpu().detach().numpy()
        mu_S = np.mean(X, axis=0)
        cov_S = np.cov(X.T)

        Y = benchmark.output_sampler.sample(int(1e6))
        Y = Y.cpu().detach().numpy()
        mu_T = np.mean(Y, axis=0)
        cov_T = np.cov(Y.T)
        
        n_samples = n_sampless[id]
        
        X_train = benchmark.input_sampler.sample(n_samples) 
        Y_train = benchmark.output_sampler.sample(n_samples) 
        X_train_ = X_train.cpu().detach().numpy()
        Y_train_ = Y_train.cpu().detach().numpy()
        
        t0 = time.time()
        train_eval_split = int(0.8 * n_samples)
        geom_fwd = pointcloud.PointCloud(X_train_, Y_train_, cost_fn=cost_fn)
        geom_inv = pointcloud.PointCloud(Y_train_, X_train_, cost_fn=cost_fn)
        
        tuned_epsilons_fwd = progot.get_epsilon_schedule(
                                                    geom_fwd,
                                                    alphas=alphas,
                                                    epsilon_scales=end_epsilon_scales,
                                                    start_epsilon_scale=0.5,
                                                    y_eval=Y_train_[train_eval_split:],
                                                    )
        tuned_epsilons_inv = progot.get_epsilon_schedule(
                                                    geom_inv,
                                                    alphas=alphas,
                                                    epsilon_scales=end_epsilon_scales,
                                                    start_epsilon_scale=0.5,
                                                    y_eval=X_train_[train_eval_split:],
                                                    )
        
        # out = run_progot(X_train_, Y_train_, cost_fn, alphas=alphas, epsilons=None)

        out_fwd = run_progot(X_train_, Y_train_, cost_fn, alphas=alphas, epsilons=tuned_epsilons_fwd)
        pi_fwd = out_fwd.get_output(-1).matrix

        out_inv = run_progot(Y_train_, X_train_, cost_fn, alphas=alphas, epsilons=tuned_epsilons_inv)
        pi_inv = out_inv.get_output(-1).matrix
        
        t1 = time.time()
        _, cpu_peak = tracemalloc.get_traced_memory()
        gpu_peak = torch.cuda.max_memory_allocated()
        tracemalloc.stop()

        trans_cost_train_fwd = jnp.sum(pi_fwd * geom_fwd.cost_matrix)
        trans_cost_train_inv = jnp.sum(pi_inv * geom_inv.cost_matrix)

        for noise_scale, p_drop_out in [(0. , 0.), (.1, 0.), (.25, 0.), (.5, 0.), (1., 0.), (0., 0.05), (0., .1), (0., .25), (0., .5)]:
            x_noise = torch.randn(test_smp, d) * noise_scale
            X_test = benchmark.input_sampler.sample(test_smp) 
            X_test.requires_grad_(True)
            Y_test = benchmark.map_fwd(X_test,  nograd=True) 
            X_test_noise = X_test + x_noise.cuda(GPU_DEVICE)
            X_test_noise = F.dropout(X_test_noise, p=p_drop_out, training=True)
            Y_test_noise = benchmark.map_fwd(X_test_noise, nograd=True)
            
            X_test_ = X_test.cpu().detach().numpy()
            X_test_noise_ = X_test_noise.cpu().detach().numpy()
            Y_test_ = Y_test.cpu().detach().numpy()
            Y_test_noise_ = Y_test_noise.cpu().detach().numpy()
            
            _, y_ = out_fwd.transport(X_test_, return_intermediate=False)
            trans_cost_transported_fwd = jnp.sum(pi_fwd * geom_fwd.cost_matrix)

            if noise_scale == 0 and p_drop_out == 0:
                _, x_ = out_inv.transport(Y_test_, return_intermediate=False)
                trans_cost_transported_inv = jnp.sum(pi_inv * geom_inv.cost_matrix)
                _, xx_ = out_inv.transport(y_, return_intermediate=False)
                _, yy_ = out_fwd.transport(x_, return_intermediate=False)

                Y_push = torch.tensor(x_, dtype=torch.float32).cuda(GPU_DEVICE)
                YY_push = torch.tensor(xx_, dtype=torch.float32).cuda(GPU_DEVICE)
                XX_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)
                
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
                print(f"Costs Fwd 2: {mmd_T_2}, {L2_T_2}, {L2_UVP_2}, {dist_2}")

            else:
                mmd_T_inv, mmd_T_inv_2 = 0., 0.
                L2_T_inv, L2_T_inv_2 = 0., 0.
                L2_UVP_inv, L2_UVP_inv_2 = 0., 0.
                cos_sim_inv, cos_sim_inv_2 = 0., 0.
                dist_inv, dist_inv_2 = 0., 0.
            
            
            X_push = torch.tensor(y_, dtype=torch.float32).cuda(GPU_DEVICE)

            # BW_UVP = compute_BW_metrics(y_, mu_T, cov_T)
            mmd_T = compute_mmd(X=Y_test_, Y=y_) * 100
            mmd_T_noise = compute_mmd(X=Y_test_noise_, Y=y_) * 100
            
            L2_T = np.mean(np.linalg.norm(Y_test_ - y_, ord=2, axis=1))
            L2_T_noise = np.mean(np.linalg.norm(Y_test_noise_ - y_, ord=2, axis=1)) 
            
            L2_UVP, cos_sim = L2_UVP_cost(benchmark, X_test_noise, Y_test, X_push)
            L2_UVP_noise, cos_sim_noise = L2_UVP_cost(benchmark, X_test_noise, Y_test_noise, X_push)
            
            dist = sinkh_div(y_, Y_test_, batch_size=1024, epsilon=0.001) 
            dist_noise = sinkh_div(y_, Y_test_noise_, batch_size=1024, epsilon=0.001)
            
            print(f"Costs for noise scale {noise_scale} and dropout {p_drop_out}: {mmd_T}, {L2_T}, {L2_UVP}, {dist}")
                   
            EXP_NAME = f'series_2_{BENCHMARK}_EPSILON_{eps}_ProgOT_DIM_{d}_run_{rep}_numSmp_{n_samples}_noise_Gauss_{noise_scale}_p_drop_{p_drop_out}'
            save_path = f'./results/progot/{EXP_NAME}'
            os.makedirs(save_path, exist_ok=True)

            # Save the final metrics for visualization
            final_metrics = {
                            'mmd': [mmd_T, mmd_T_2],
                            'mmd_noise': mmd_T_noise,
                            'mmd_inv': [mmd_T_inv, mmd_T_inv_2],
                            'cost_train': [trans_cost_train_fwd, trans_cost_train_inv],
                            'cost': [trans_cost_transported_fwd, trans_cost_transported_inv],
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
                            'time': t1 - t0,
                            'num_iter': [out_fwd.num_iters, out_inv.num_iters],
                            'num_step': [out_fwd.num_steps, out_inv.num_steps],
                            'cpu_memory_peak_MB': cpu_peak / 1024**2,
                            'gpu_memory_peak_MB': gpu_peak / 1024**2,
                            }
            
            np.save(f'{save_path}/metrics.npy', final_metrics)