#!/usr/bin/env python3
"""
GMM-OT alignment of ImageNet ResNet50 embeddings, with the same class-preservation
evaluation used in run_omt.py / hiref.py.

Method
------
The GMM pipeline this file always used, now run on the same stratified split that
the other experiments use:

1. KMeans on the source cloud X and on the target cloud Y (n_components each).
2. Build a sklearn GaussianMixture from those centres (weights = cluster shares,
   means = cluster means, diagonal or full covariances) and refine with EM
   (warm_start), exactly as before.
3. Solve OT between the two sets of mixture components.  The component cost is
   the exact Bures-Wasserstein W2^2 between the two diagonal Gaussians
   (non-entropic; --cost-eps 0, the default), or the entropic-Gaussian distance
   from omt/functions.py (compute_distance_diagonal) if --cost-eps > 0.
   Solver is one of omt/solver.py's three:
     --method w2       : exact, non-entropic OT (ot.emd) -- the default
     --method sinkhorn : entropic OT, needs --reg > 0
     --method cvx      : conic programme via cvxpy/SCS
4. Push each source point forward through the diagonal Gaussian map between its
   assigned source component i and the matched target component j:
       T(x) = (x - mu_i) * sqrt(var_j / var_i) + mu_j
5. Report transport cost *and* class preservation -- matched-pair class accuracy
   plus the frozen ResNet50 `fc` head applied to the transported samples.

Unlike the `--method gmmot` path of run_omt.py (which loads the
precomputed <model_dir>/gmm_1000.pkl fitted on a *different*, unstratified split of
all 1.28M embeddings), this script fits the mixtures on the very X and Y it
evaluates -- so its numbers are directly comparable to hiref / sinkhorn / cvx.

Usage
-----
    python gmm.py --per-class 2 --n-components 1000            # w2, exact
    python gmm.py --per-class 1 --n-components 200 --method sinkhorn --reg 0.01
    python gmm.py --n-classes 100 --per-class 2 --n-components 50

Output
------
    <results_dir>/imagenet_gmmot_<method>_n<n>_c<n_classes>_K<n_components>.json
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import ot
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()

# Shared data loading / classifier / metrics -- identical to the OMT experiment.
from run_omt import (EMB_PATH, OUT_DIR, RESNET_PATH, load_classifier,
                          load_source_target, nearest_target, predict, topk_hit)
from functions import compute_distance_diagonal   # omt/functions.py


# ── GMM fitting ──────────────────────────────────────────────────────────────
def fit_gmm(data, n_components, cov_type, seed, gmm_max_samples=None, eps=1e-6,
            label=""):
    """KMeans init -> sklearn GaussianMixture (warm start) -> EM refinement.

    This is the pipeline this script has always used; only the input data and the
    KMeans backend (sklearn instead of cuML, which isn't installed here) differ.
    """
    n, d = data.shape
    if n_components > n:
        raise ValueError(f"{n_components} components > {n} samples in {label}")

    print(f"[{label}] KMeans with {n_components} clusters on {n:,} points ...",
          flush=True)
    t0 = time.time()
    kmeans = KMeans(n_clusters=n_components, random_state=seed, n_init=1)
    kmeans.fit(data)
    time_kmeans = time.time() - t0
    print(f"[{label}]   KMeans in {time_kmeans:.1f}s", flush=True)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    weights = np.zeros(n_components, dtype=np.float64)
    means = centers.copy().astype(np.float64)
    if cov_type == "diag":
        covariances = np.zeros((n_components, d), dtype=np.float64)
    elif cov_type == "full":
        covariances = np.zeros((n_components, d, d), dtype=np.float64)
    else:
        raise ValueError("cov_type must be 'diag' or 'full'")

    for k in range(n_components):
        mask = labels == k
        Xk = data[mask]
        nk = Xk.shape[0]
        if nk == 0:                       # empty cluster: tiny weight, tiny cov
            weights[k] = 1e-6
            covariances[k] = (np.ones(d) if cov_type == "diag" else np.eye(d)) * eps
            continue
        weights[k] = nk / n
        means[k] = Xk.mean(axis=0)
        Xc = Xk - means[k]
        if cov_type == "diag":
            covariances[k] = Xc.var(axis=0) + eps
        else:
            covariances[k] = (Xc.T @ Xc) / nk + np.eye(d) * eps
    weights /= weights.sum()

    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                          n_init=1, warm_start=True, random_state=seed)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, cov_type)

    fit_data = data
    if gmm_max_samples and n > gmm_max_samples:
        rng = np.random.default_rng(seed)
        fit_data = data[rng.choice(n, gmm_max_samples, replace=False)]
    print(f"[{label}] Fitting GMM (EM) on {fit_data.shape[0]:,} points ...", flush=True)
    t0 = time.time()
    gmm.fit(fit_data)
    time_gmm = time.time() - t0
    print(f"[{label}]   GMM in {time_gmm:.1f}s ({gmm.n_iter_} EM iters)", flush=True)

    return gmm, {"time_kmeans_s": time_kmeans, "time_gmm_s": time_gmm,
                 "n_iter": int(gmm.n_iter_)}


def diag_params(gmm):
    """(means, variances) with variances as (K, d) vectors whatever sklearn stored."""
    cov = gmm.covariances_
    if cov.ndim == 3:
        # np.diagonal returns a read-only view; copy so torch can wrap it.
        cov = np.ascontiguousarray(np.diagonal(cov, axis1=1, axis2=2))
    return gmm.means_, cov


# ── Component cost ───────────────────────────────────────────────────────────
def bures_wasserstein_diagonal(mu_s, var_s, mu_t, var_t, batch_size=100):
    """Exact (non-entropic) W2^2 between diagonal Gaussians.

        W2^2(N(mu_s, S), N(mu_t, T)) = ||mu_s - mu_t||^2
                                       + sum_k (sqrt(S_k) - sqrt(T_k))^2

    This is the eps -> 0 limit of omt/functions.py's compute_distance_diagonal:
    at eps = 0 that formula's log terms are 0*log(0) = nan, so the limit is
    written out directly here.
    """
    mu_s, mu_t = np.asarray(mu_s), np.asarray(mu_t)
    sd_s, sd_t = np.sqrt(np.asarray(var_s)), np.sqrt(np.asarray(var_t))
    K_s, K_t = mu_s.shape[0], mu_t.shape[0]
    out = np.zeros((K_s, K_t), dtype=np.float64)
    for i in range(0, K_s, batch_size):
        end = min(i + batch_size, K_s)
        d_mu = np.sum((mu_s[i:end][:, None, :] - mu_t[None, :, :]) ** 2, axis=-1)
        d_sd = np.sum((sd_s[i:end][:, None, :] - sd_t[None, :, :]) ** 2, axis=-1)
        out[i:end] = d_mu + d_sd
    return out


def component_cost(mu_s, var_s, mu_t, var_t, eps, batch_size=100):
    """Component-level cost matrix: exact W2^2 at eps=0, entropic-Gaussian above."""
    if eps == 0:
        print("Component cost: exact Bures-Wasserstein W2^2 (non-entropic)", flush=True)
        return bures_wasserstein_diagonal(mu_s, var_s, mu_t, var_t, batch_size)
    print(f"Component cost: entropic-Gaussian, eps={eps}", flush=True)
    C, _, _ = compute_distance_diagonal(mu_s, var_s, mu_t, var_t, eps=eps,
                                        batch_size=batch_size)
    return np.asarray(C, dtype=np.float64)


# ── Component-level OT ───────────────────────────────────────────────────────
def solve_component_ot(alpha_s, alpha_t, C_norm, method, reg, max_iter, stop_thr,
                       cvx_solver="SCS"):
    """OT between mixture components. Same three solvers as omt/solver.py."""
    t0 = time.time()
    if method == "sinkhorn":
        if reg <= 0:
            raise ValueError("--method sinkhorn needs --reg > 0")
        print(f"Solving component OT (sinkhorn, reg={reg}) ...", flush=True)
        w, log = ot.bregman.sinkhorn(alpha_s, alpha_t, C_norm, reg,
                                     numItermax=max_iter, stopThr=stop_thr, log=True)
        n_iter = int(log.get("niter", max_iter))
    elif method == "w2":
        print("Solving component OT (exact, ot.emd -- non-entropic) ...", flush=True)
        w = ot.emd(alpha_s, alpha_t, C_norm, numItermax=max_iter)
        n_iter = None
    elif method == "cvx":
        import cvxpy as cvx

        K_s, K_t = C_norm.shape
        print(f"Solving component OT (cvxpy/{cvx_solver}, reg={reg}) -- "
              f"{K_s * K_t:,} variables ...", flush=True)
        x = cvx.Variable((K_s, K_t), nonneg=True)
        objective = cvx.sum(cvx.multiply(C_norm, x))
        if reg > 0:
            # Same KL penalty as omt/solver.py's cvx branch.
            objective = objective + reg * cvx.sum(
                cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :]))
        problem = cvx.Problem(cvx.Minimize(objective),
                              [x @ np.ones(K_t) == alpha_s,
                               x.T @ np.ones(K_s) == alpha_t])
        problem.solve(solver=getattr(cvx, cvx_solver), max_iters=max_iter,
                      eps=stop_thr)
        if x.value is None:
            raise RuntimeError(f"cvxpy failed: status={problem.status}")
        w = np.clip(np.asarray(x.value), 0.0, None)
        w /= w.sum()
        n_iter = problem.solver_stats.num_iters
        print(f"  cvxpy status: {problem.status}", flush=True)
    else:
        raise ValueError(f"unknown method: {method}")
    solve_time = time.time() - t0
    print(f"  solved in {solve_time:.1f}s", flush=True)
    return w, {"solve_time_s": solve_time, "n_iter": n_iter}


def gmm_pushforward(X_t, gmm_s, gmm_t, omt_w, eps=1e-6):
    """T(x) = (x - mu_i) * sqrt(var_j / var_i) + mu_j, j = argmax_j omt_w[i, j]."""
    mu_s, var_s = diag_params(gmm_s)
    mu_t, var_t = diag_params(gmm_t)
    X_np = X_t.cpu().numpy()
    comp_i = gmm_s.predict(X_np)                       # most likely source component
    comp_j = np.argmax(omt_w, axis=1)[comp_i]          # matched target component
    dev = X_t.device
    mu_s_t = torch.as_tensor(mu_s, dtype=torch.float32, device=dev)
    mu_t_t = torch.as_tensor(mu_t, dtype=torch.float32, device=dev)
    var_s_t = torch.as_tensor(var_s, dtype=torch.float32, device=dev)
    var_t_t = torch.as_tensor(var_t, dtype=torch.float32, device=dev)
    i_t = torch.as_tensor(comp_i, dtype=torch.long, device=dev)
    j_t = torch.as_tensor(comp_j, dtype=torch.long, device=dev)
    scale = torch.sqrt(var_t_t[j_t] / (var_s_t[i_t] + eps))
    return (X_t - mu_s_t[i_t]) * scale + mu_t_t[j_t], comp_i, comp_j


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="approximate samples per side; rounded down to a whole "
                             "number of images per class")
    parser.add_argument("--per-class", type=int, default=None,
                        help="images per class per side; overrides --n-pairs")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="use a random subset of this many classes")
    parser.add_argument("--n-components", type=int, default=200,
                        help="mixture components per side (Ks = Kt)")
    parser.add_argument("--cov-type", default="diag", choices=["diag", "full"])
    parser.add_argument("--method", default="w2",
                        choices=["w2", "sinkhorn", "cvx"],
                        help="component-level OT solver (omt/solver.py's three); "
                             "default w2 = exact, non-entropic (ot.emd)")
    parser.add_argument("--reg", type=float, default=0.0,
                        help="entropic reg (sinkhorn) / KL weight (cvx); ignored "
                             "by w2, which is unregularised")
    parser.add_argument("--cost-eps", type=float, default=0.0,
                        help="entropic-Gaussian eps in the component cost; 0 (the "
                             "default) = exact Bures-Wasserstein W2^2")
    parser.add_argument("--cvx-solver", default="SCS")
    parser.add_argument("--gmm-max-samples", type=int, default=100000,
                        help="cap on points used for the EM refinement")
    parser.add_argument("--max-iter", type=int, default=int(1e6))
    parser.add_argument("--stop-thr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb", default=EMB_PATH)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--save-gmm", default=None,
                        help="optional path to pickle the fitted GMMs + coupling")
    parser.add_argument("--load-gmm", default=None,
                        help="reuse GMMs pickled by an earlier run (skip KMeans/EM); "
                             "must be the same split/seed/K")
    parser.add_argument("--out", default=None, help="path for the JSON report")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"On device: {device}", flush=True)
    torch.manual_seed(args.seed)

    # 1-2. Embeddings, class-stratified random split into source / target.
    X, class_X, Y, class_Y = load_source_target(args.emb, args.n_pairs, args.seed,
                                                per_class=args.per_class,
                                                n_classes_subset=args.n_classes)
    n_side = X.shape[0]
    n_classes = len(np.unique(class_X))
    per_class = n_side // n_classes
    X_np, Y_np = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    X_t = torch.as_tensor(X, device=device)
    Y_t = torch.as_tensor(Y, device=device)

    # ResNet50 classifier head, used to label the transported samples.
    clf = load_classifier(RESNET_PATH, X.shape[1], device)

    # 3. Fit the two mixtures on this very split (or reuse a cached fit).
    if args.load_gmm:
        print(f"Loading cached GMMs from {args.load_gmm} ...", flush=True)
        with open(args.load_gmm, "rb") as f:
            cached = pickle.load(f)
        gmm_s, gmm_t = cached["gmm_s"], cached["gmm_t"]
        if gmm_s.means_.shape[0] != args.n_components:
            raise ValueError(f"cached GMM has {gmm_s.means_.shape[0]} components, "
                             f"--n-components is {args.n_components}")
        t_s = t_t = {"time_kmeans_s": 0.0, "time_gmm_s": 0.0, "n_iter": 0}
    else:
        gmm_s, t_s = fit_gmm(X_np, args.n_components, args.cov_type, args.seed,
                             args.gmm_max_samples, label="source")
        gmm_t, t_t = fit_gmm(Y_np, args.n_components, args.cov_type, args.seed,
                             args.gmm_max_samples, label="target")

    mu_s, var_s = diag_params(gmm_s)
    mu_t, var_t = diag_params(gmm_t)

    print("Computing component-level cost matrix ...", flush=True)
    C_comp = component_cost(mu_s, var_s, mu_t, var_t, args.cost_eps, batch_size=100)
    C_comp_norm = (C_comp / C_comp.max()).astype(np.float64)

    omt_w, solve_info = solve_component_ot(
        gmm_s.weights_, gmm_t.weights_, C_comp_norm, args.method, args.reg,
        args.max_iter, args.stop_thr, cvx_solver=args.cvx_solver)
    comp_cost = float(np.sum(omt_w * C_comp))
    print(f"  component transport cost: {comp_cost:.4f}", flush=True)

    # 4. Pushforward.
    X_hat, comp_i, comp_j = gmm_pushforward(X_t, gmm_s, gmm_t, omt_w)

    if args.save_gmm:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_gmm)), exist_ok=True)
        with open(args.save_gmm, "wb") as f:
            pickle.dump({"gmm_s": gmm_s, "gmm_t": gmm_t, "omt_w": omt_w}, f)
        print(f"Fitted GMMs + coupling saved to {args.save_gmm}", flush=True)

    # Cost summaries, same definitions as the other two scripts.
    displacement = float(torch.norm(X_t - X_hat, dim=1).mean())
    nn_idx, nn_sq = nearest_target(X_hat, Y_t)
    nn_dist = float(np.sqrt(nn_sq).mean())
    # GMM-OT has no sample-level plan; the match is the nearest target to T(x).
    match = nn_idx

    # 5a. Class preservation of the matching.
    matched_class_acc = float(np.mean(class_Y[match] == class_X))
    rng = np.random.default_rng(args.seed + 1)
    chance_acc = float(np.mean(class_Y[rng.permutation(len(class_Y))] == class_X))

    # 5b. Class preservation under the ResNet50 classifier head.
    X_hat_np = X_hat.cpu().numpy()
    pred_X, top5_X = predict(clf, X, device)
    pred_Y, _ = predict(clf, Y, device)
    pred_hat, top5_hat = predict(clf, X_hat_np, device)

    metrics = {
        "method": f"gmmot_{args.method}",
        "n_per_side": int(n_side),
        "n_classes": int(n_classes),
        "per_class_per_side": int(per_class),
        "n_components": int(args.n_components),
        "cov_type": args.cov_type,
        "dim": int(X.shape[1]),
        "reg": args.reg,
        "cost_eps": args.cost_eps,
        "seed": args.seed,
        # ── transport cost ──
        "transport_cost_plan": None,                          # no sample-level plan
        "component_transport_cost": comp_cost,
        "mean_displacement_L2": displacement,
        "mean_displacement_L2_per_dim": displacement / X.shape[1],
        "mean_dist_to_nearest_target": nn_dist,
        # ── class preservation of the alignment ──
        "matched_class_accuracy": matched_class_acc,
        "nearest_neighbour_class_accuracy": matched_class_acc,  # match == NN here
        "plan_weighted_class_accuracy": None,
        "random_matching_class_accuracy": chance_acc,
        # ── ResNet50 classifier on the transported points ──
        "clf_top1_source": float(np.mean(pred_X == class_X)),
        "clf_top1_target": float(np.mean(pred_Y == class_Y)),
        "clf_top1_transported": float(np.mean(pred_hat == class_X)),
        "clf_top5_source": topk_hit(top5_X, class_X),
        "clf_top5_transported": topk_hit(top5_hat, class_X),
        "label_preservation_top1": float(np.mean(pred_hat == pred_X)),
        "label_preservation_top5": topk_hit(top5_hat, pred_X),
        "clf_agreement_with_matched_target": float(np.mean(pred_hat == pred_Y[match])),
        "n_distinct_targets_matched": int(len(np.unique(match))),
        # ── mixture / timing detail ──
        "n_target_components_used": int(len(np.unique(np.argmax(omt_w, axis=1)))),
        "n_source_components_occupied": int(len(np.unique(comp_i))),
        "gmm_time_source_s": t_s["time_kmeans_s"] + t_s["time_gmm_s"],
        "gmm_time_target_s": t_t["time_kmeans_s"] + t_t["time_gmm_s"],
        "gmm_em_iters": t_s["n_iter"] + t_t["n_iter"],
    }
    metrics.update({f"ot_{k}": v for k, v in solve_info.items()})

    print("\n" + "=" * 68)
    print(f"GMM-OT alignment of ImageNet embeddings  |  solver={args.method}  "
          f"K={args.n_components}")
    print(f"n={n_side} per side ({per_class}/class x {n_classes} classes)")
    print("=" * 68)
    print("-- transport cost")
    print(f"  component transport cost            : {comp_cost:.4f}")
    print(f"  mean ||x - T(x)||                   : {displacement:.4f} "
          f"({displacement / X.shape[1]:.3e} per dim)")
    print(f"  mean ||T(x) - nearest y||           : {nn_dist:.4f}")
    print("-- class preservation of the matching")
    print(f"  matched-pair class accuracy (NN)    : {matched_class_acc:.4f}")
    print(f"  random-matching baseline (chance)    : {chance_acc:.4f}")
    print("-- ResNet50 classifier head on transported samples")
    print(f"  top-1 on source X                   : {metrics['clf_top1_source']:.4f}")
    print(f"  top-1 on target Y                   : {metrics['clf_top1_target']:.4f}")
    print(f"  top-1 on T(X) vs true source class  : {metrics['clf_top1_transported']:.4f}")
    print(f"  top-5 on T(X) vs true source class  : {metrics['clf_top5_transported']:.4f}")
    print(f"  label preservation clf(T(x))==clf(x): {metrics['label_preservation_top1']:.4f}")
    print(f"  clf(T(x)) == clf(matched y)         : "
          f"{metrics['clf_agreement_with_matched_target']:.4f}")
    print(f"  distinct target images matched      : "
          f"{metrics['n_distinct_targets_matched']} / {n_side}")
    print(f"  target components used              : "
          f"{metrics['n_target_components_used']} / {args.n_components}")
    print("=" * 68 + "\n")

    out_path = args.out or os.path.join(
        OUT_DIR, f"imagenet_gmmot_{args.method}_n{n_side}_c{n_classes}"
                 f"_K{args.n_components}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
