#!/usr/bin/env python3
"""
Fit a diagonal Gaussian mixture to the ImageNet ResNet50 embeddings, once for the
source cloud and once for the target cloud, and save both under <model_dir>.

Steps (exactly the requested pipeline -- no EM refinement):
  1. GPU KMeans (faiss) with K clusters on the embeddings.
  2. From the hard cluster assignment, estimate each component's parameters:
         weight_k = n_k / n
         mean_k   = mean of the points in cluster k
         var_k    = per-dimension variance of the points in cluster k   (+ eps)
     Only the DIAGONAL of the covariance is estimated -- we are not learning a
     full covariance.
  3. Package as an sklearn GaussianMixture(covariance_type='diag') so downstream
     code (omt component OT, points_transport) can consume gmm.means_/
     .covariances_/.weights_/.predict just as with a fitted mixture.

Run once per side; the source and target mixtures are fit on the SAME
class-stratified split used by run_omt.py / hiref.py so everything
downstream stays comparable.

Output (under <model_dir> from config.toml by default):
    <model_dir>/gmm_s_n<n>_c<n_classes>_K<K>.pkl      # source mixture
    <model_dir>/gmm_t_n<n>_c<n_classes>_K<K>.pkl      # target mixture
    <model_dir>/gmm_n<n>_c<n_classes>_K<K>.pkl        # both, {"gmm_s":..., "gmm_t":...}

Usage
-----
    python fit_gmm.py --per-class 366 --n-components 1000
    python fit_gmm.py --per-class 1   --n-components 200
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()

# Same class-stratified split as the OMT / HiRef experiments.
from run_omt import EMB_PATH, MODEL_DIR, load_source_target



# ── GPU KMeans -> diagonal GMM ────────────────────────────────────────────────
def fit_diagonal_gmm_gpu(data, n_components, seed, eps=1e-6, niter=25,
                         nredo=1, label=""):
    """GPU KMeans (faiss) then diagonal-Gaussian parameter estimation.

    Returns (gmm, info).  `gmm` is an sklearn GaussianMixture(covariance_type=
    'diag') whose parameters come straight from the KMeans clusters -- weight,
    per-cluster mean, and per-dimension variance (the diagonal of the covariance).
    """
    import faiss

    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    if n_components > n:
        raise ValueError(f"{n_components} components > {n} samples in {label}")

    # 1. GPU KMeans.
    print(f"[{label}] faiss GPU KMeans: K={n_components} on {n:,} x {d} points "
          f"({niter} iters, {nredo} redo) ...", flush=True)
    t0 = time.time()
    km = faiss.Kmeans(d, n_components, niter=niter, nredo=nredo, verbose=False,
                      seed=seed, gpu=True, spherical=False)
    km.train(data)
    # faiss>=1.14 exposes centroids as a numpy array already; older builds need
    # vector_to_array. Handle both.
    if isinstance(km.centroids, np.ndarray):
        centers = np.asarray(km.centroids).reshape(n_components, d)
    else:
        centers = faiss.vector_to_array(km.centroids).reshape(n_components, d)
    _, assign = km.index.search(data, 1)      # nearest centroid per point (GPU)
    labels = assign.ravel()
    time_kmeans = time.time() - t0
    print(f"[{label}]   KMeans in {time_kmeans:.1f}s", flush=True)

    # 2. Diagonal Gaussian parameters from the hard assignment.
    #    Vectorised sum / sum-of-squares per cluster via np.add.at.
    t0 = time.time()
    counts = np.bincount(labels, minlength=n_components).astype(np.float64)
    sum_x = np.zeros((n_components, d), dtype=np.float64)
    sum_x2 = np.zeros((n_components, d), dtype=np.float64)
    np.add.at(sum_x, labels, data.astype(np.float64))
    np.add.at(sum_x2, labels, data.astype(np.float64) ** 2)

    weights = np.zeros(n_components, dtype=np.float64)
    means = centers.astype(np.float64).copy()
    variances = np.ones((n_components, d), dtype=np.float64)

    nonempty = counts > 0
    weights[nonempty] = counts[nonempty] / n
    means[nonempty] = sum_x[nonempty] / counts[nonempty, None]
    variances[nonempty] = (sum_x2[nonempty] / counts[nonempty, None]
                           - means[nonempty] ** 2)
    variances = np.clip(variances, 0.0, None) + eps      # diagonal terms only

    n_empty = int((~nonempty).sum())
    if n_empty:
        # Empty clusters keep the KMeans centroid, a tiny weight and unit variance.
        weights[~nonempty] = eps
        print(f"[{label}]   {n_empty} empty cluster(s) -> tiny weight", flush=True)
    weights /= weights.sum()
    time_params = time.time() - t0

    # 3. Wrap in an sklearn GaussianMixture(diag) so downstream code just works.
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = variances
    gmm.precisions_cholesky_ = _compute_precision_cholesky(variances, "diag")

    info = {"n": int(n), "dim": int(d), "n_components": int(n_components),
            "n_empty_clusters": n_empty, "time_kmeans_s": time_kmeans,
            "time_params_s": time_params, "seed": int(seed)}
    print(f"[{label}]   diagonal params in {time_params:.1f}s "
          f"(weights sum={weights.sum():.4f}, mean var={variances.mean():.3e})",
          flush=True)
    return gmm, info


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="approximate samples per side; rounded down to a "
                             "whole number of images per class")
    parser.add_argument("--per-class", type=int, default=None,
                        help="images per class per side; overrides --n-pairs")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="use a random subset of this many classes")
    parser.add_argument("--n-components", type=int, default=1000,
                        help="mixture components per side (K)")
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--kmeans-redo", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="floor added to the diagonal variances")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb", default=EMB_PATH)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Class-stratified split -> source cloud X, target cloud Y (same as OMT/HiRef).
    X, class_X, Y, class_Y = load_source_target(args.emb, args.n_pairs, args.seed,
                                                per_class=args.per_class,
                                                n_classes_subset=args.n_classes)
    n_side = X.shape[0]
    n_classes = len(np.unique(class_X))

    # Fit once for source, once for target.
    gmm_s, info_s = fit_diagonal_gmm_gpu(X, args.n_components, args.seed,
                                         eps=args.eps, niter=args.kmeans_iters,
                                         nredo=args.kmeans_redo, label="source")
    gmm_t, info_t = fit_diagonal_gmm_gpu(Y, args.n_components, args.seed,
                                         eps=args.eps, niter=args.kmeans_iters,
                                         nredo=args.kmeans_redo, label="target")

    os.makedirs(args.model_dir, exist_ok=True)
    tag = f"n{n_side}_c{n_classes}_K{args.n_components}"
    meta = {"n_per_side": int(n_side), "n_classes": int(n_classes),
            "n_components": int(args.n_components), "cov_type": "diag",
            "seed": int(args.seed), "emb": args.emb,
            "per_class": int(n_side // n_classes)}

    p_s = os.path.join(args.model_dir, f"gmm_s_{tag}.pkl")
    p_t = os.path.join(args.model_dir, f"gmm_t_{tag}.pkl")
    p_both = os.path.join(args.model_dir, f"gmm_{tag}.pkl")
    with open(p_s, "wb") as f:
        pickle.dump({"gmm": gmm_s, "info": info_s, "meta": meta}, f)
    with open(p_t, "wb") as f:
        pickle.dump({"gmm": gmm_t, "info": info_t, "meta": meta}, f)
    with open(p_both, "wb") as f:
        pickle.dump({"gmm_s": gmm_s, "gmm_t": gmm_t,
                     "info_s": info_s, "info_t": info_t, "meta": meta}, f)

    print("\n" + "=" * 60)
    print(f"Diagonal GMMs fit (GPU KMeans)  |  {tag}")
    print(f"  source: {gmm_s.means_.shape[0]} comps, {info_s['n_empty_clusters']} empty, "
          f"{info_s['time_kmeans_s']:.1f}s KMeans")
    print(f"  target: {gmm_t.means_.shape[0]} comps, {info_t['n_empty_clusters']} empty, "
          f"{info_t['time_kmeans_s']:.1f}s KMeans")
    print(f"  saved:\n    {p_s}\n    {p_t}\n    {p_both}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
