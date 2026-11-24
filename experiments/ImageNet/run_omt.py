#!/usr/bin/env python3
"""
Optimal-transport alignment of ImageNet ResNet50 embeddings, with class-preservation.

Pipeline
--------
1. Load the ResNet50 embeddings (<embeddings>, 2048-d,
   produced by embedding.py) together with their ground-truth class ids.
2. Randomly split a subsample into a *source* set X and a *target* set Y,
   stratified by class: every class puts the same number of (distinct) images in
   X as in Y.  The halves are disjoint, share an identical class histogram, and
   the ground-truth alignment is therefore class-preserving.
3. Solve an OT problem between X and Y in the latent space.  The three
   sample-level solvers are the ones omt/solver.py offers:
     --method sinkhorn : entropic OT, Bregman projections (default, GPU)
     --method emd      : exact (unregularised) OT, network simplex ('w2' there)
     --method cvx      : the conic programme min <C,P> + reg*KL(P||ab^T) solved
                         with cvxpy/SCS -- exact but O(n^2) variables, so slow;
                         pair it with --n-classes / --per-class to size down
     --method gmmot    : GMM-OT, i.e. OT between the mixture components fitted in
                         gmm.py (<model_dir>/gmm_1000.pkl), the point map being
                         the per-component Gaussian (diagonal) transport map.
4. Report the transport cost *and* how well the alignment preserves the class id:
     - matched-pair accuracy: class of x vs class of its transported partner in Y,
     - classifier accuracy on the transported points T(x), obtained by pushing
       T(x) through the frozen ResNet50 classifier head (`fc`) loaded from
       <resnet_weights>.

Usage
-----
    python run_omt.py --n-pairs 10000 --method sinkhorn --reg 0.01
    python run_omt.py --per-class 5   --method emd        # 5 per class per side
    python run_omt.py --n-classes 200 --per-class 2 --method cvx
    python run_omt.py --n-pairs 5000  --method gmmot      # needs gmm_1000.pkl

Output
------
    <results_dir>/imagenet_omt_<method>_n<n>.json   (all reported metrics)
"""

import argparse
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import ot
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import PATHS, add_code_dirs
add_code_dirs()

# All data / model / output locations come from config.toml (+ config.local.toml).
EMB_PATH = PATHS["embeddings"]
RESNET_PATH = PATHS["resnet_weights"]
GMM_PATH = PATHS["gmm"]
MODEL_DIR = PATHS["model_dir"]
OUT_DIR = PATHS["results_dir"]


# ── Data ─────────────────────────────────────────────────────────────────────
def load_source_target(emb_path, n_pairs, seed, per_class=None, n_classes_subset=None,
                       label_key="class_idx"):
    """Class-stratified random split of the embeddings into a source and target half.

    Every class contributes exactly `per_class` *distinct* images to the source
    and `per_class` other distinct images to the target, so X and Y are disjoint
    sets of images with identical class histograms (uniform over the classes).
    Which images go where -- and which of the two halves each image lands in --
    is random.

    `per_class` is taken from the argument if given, else n_pairs // n_classes.

    Returns (X, class_X, Y, class_Y), both of shape (per_class * n_classes, d).
    """
    with h5py.File(emb_path, "r") as f:
        n_total, dim = f["X"].shape
        print(f"Reading class labels for {n_total:,} images ...", flush=True)
        all_labels = np.asarray(f["obs"][label_key][:], dtype=np.int64)
        classes, counts = np.unique(all_labels, return_counts=True)

        rng = np.random.default_rng(seed)
        if n_classes_subset is not None:
            if n_classes_subset > len(classes):
                raise ValueError(f"--n-classes {n_classes_subset} > {len(classes)} "
                                 f"available classes")
            keep = np.sort(rng.choice(len(classes), size=n_classes_subset, replace=False))
            classes, counts = classes[keep], counts[keep]
            print(f"Restricting to {n_classes_subset} randomly chosen classes", flush=True)
        n_classes = len(classes)

        if per_class is None:
            per_class = n_pairs // n_classes
        if per_class < 1:
            raise ValueError(
                f"--n-pairs {n_pairs} over {n_classes} classes gives < 1 sample per "
                f"class per side; use --n-pairs >= {n_classes} or set --per-class"
            )
        need = 2 * per_class                      # per_class for source + per_class for target
        short = classes[counts < need]
        if len(short) > 0:
            raise ValueError(
                f"{len(short)} class(es) have fewer than {need} images "
                f"(smallest: {counts.min()}); reduce --per-class to "
                f"{counts.min() // 2}"
            )

        src_rows, tgt_rows = [], []
        for c in classes:
            pool = np.flatnonzero(all_labels == c)
            take = rng.choice(pool, size=need, replace=False)   # distinct images
            src_rows.append(take[:per_class])                   # random half -> source
            tgt_rows.append(take[per_class:])                   # other half -> target
        src_rows = np.concatenate(src_rows)
        tgt_rows = np.concatenate(tgt_rows)

        n_side = per_class * n_classes
        print(f"Stratified split: {per_class} image(s) per class per side, "
              f"{n_classes} classes -> {n_side:,} per side "
              f"({2 * n_side:,} / {n_total:,} images, {dim}-d)", flush=True)

        # h5py fancy indexing needs increasing, unique indices -> read sorted, then
        # scatter back to the original (random) order.
        t0 = time.time()

        # Large splits: one sequential read of the whole matrix is far faster than
        # h5py fancy indexing with hundreds of thousands of scattered rows.
        full = f["X"][:] if 2 * per_class * n_classes > 50000 else None

        def read_rows(rows):
            if full is not None:
                return np.asarray(full[rows], dtype=np.float32)
            order = np.argsort(rows)
            emb = np.empty((len(rows), dim), dtype=np.float32)
            emb[order] = f["X"][rows[order]]
            return emb

        X = read_rows(src_rows)
        Y = read_rows(tgt_rows)
        print(f"  read in {time.time() - t0:.1f}s", flush=True)

    class_X = all_labels[src_rows]
    class_Y = all_labels[tgt_rows]

    # Shuffle each side so row order carries no class structure.
    pX, pY = rng.permutation(len(class_X)), rng.permutation(len(class_Y))
    X, class_X = X[pX], class_X[pX]
    Y, class_Y = Y[pY], class_Y[pY]

    assert len(np.intersect1d(src_rows, tgt_rows)) == 0, "source/target overlap"
    hX = np.bincount(class_X, minlength=n_classes)
    hY = np.bincount(class_Y, minlength=n_classes)
    assert np.array_equal(hX, hY), "class histograms differ between source and target"

    print(f"source X: {X.shape} ({len(np.unique(class_X))} classes) | "
          f"target Y: {Y.shape} ({len(np.unique(class_Y))} classes) | "
          f"identical class histograms, {per_class} per class each", flush=True)
    return X, class_X, Y, class_Y


# ── ResNet50 classification head ─────────────────────────────────────────────
def load_classifier(resnet_path, dim, device):
    """The frozen ResNet50 `fc` layer: 2048-d embedding -> 1000 ImageNet logits.

    The embeddings were produced with `model.fc = Identity()`, so this layer is
    exactly the classifier that ImageNet training left on top of them.
    """
    state = torch.load(resnet_path, map_location="cpu", weights_only=True)
    W, b = state["fc.weight"], state["fc.bias"]
    if W.shape[1] != dim:
        raise ValueError(f"fc expects {W.shape[1]}-d input, embeddings are {dim}-d")

    clf = torch.nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        clf.weight.copy_(W)
        clf.bias.copy_(b)
    clf.eval().to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    print(f"ResNet50 classifier head: {tuple(W.shape)} (1000 ImageNet classes)", flush=True)
    return clf


@torch.no_grad()
def predict(clf, Z, device, batch_size=8192, topk=5):
    """Top-1 and top-k ImageNet predictions for a batch of embeddings."""
    top1, topk_idx = [], []
    for start in range(0, Z.shape[0], batch_size):
        z = torch.as_tensor(Z[start:start + batch_size], dtype=torch.float32, device=device)
        logits = clf(z)
        top1.append(logits.argmax(dim=1).cpu())
        topk_idx.append(logits.topk(topk, dim=1).indices.cpu())
    return torch.cat(top1).numpy(), torch.cat(topk_idx).numpy()


# ── Cost matrix ──────────────────────────────────────────────────────────────
def squared_cost_matrix(X_t, Y_t, chunk=2048):
    """Squared Euclidean cost matrix ||x_i - y_j||^2, computed chunk-wise."""
    n, m = X_t.shape[0], Y_t.shape[0]
    C = torch.empty((n, m), dtype=torch.float32, device=X_t.device)
    y_sq = (Y_t * Y_t).sum(dim=1)
    for start in range(0, n, chunk):
        xb = X_t[start:start + chunk]
        C[start:start + chunk] = (
            (xb * xb).sum(dim=1, keepdim=True) + y_sq[None, :] - 2.0 * (xb @ Y_t.T)
        )
    return C.clamp_(min=0.0)


# ── OT solvers ───────────────────────────────────────────────────────────────
def solve_sample_ot(X_t, Y_t, method, reg, max_iter, stop_thr, cvx_solver="SCS"):
    """Sample-level OT between the two point clouds. Returns (plan, C, info).

    The three solvers mirror the ones offered by omt/solver.py:
      sinkhorn : entropic regularisation, Bregman projections (GPU)
      emd      : exact OT, network simplex  (= 'w2' in omt/solver.py)
      cvx      : the LP/KL programme solved with a conic solver (SCS via cvxpy),
                 i.e. min <C,P> + reg * KL(P || a b^T) s.t. P1 = a, P^T 1 = b
    """
    n, m = X_t.shape[0], Y_t.shape[0]
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=X_t.device)
    b = torch.full((m,), 1.0 / m, dtype=torch.float32, device=Y_t.device)

    print(f"Building {n} x {m} cost matrix ...", flush=True)
    C = squared_cost_matrix(X_t, Y_t)
    C_norm = C / C.max()

    t0 = time.time()
    if method == "sinkhorn":
        print(f"Solving entropic OT (reg={reg}, log-domain) ...", flush=True)
        P, log = ot.bregman.sinkhorn(
            a, b, C_norm, reg, method="sinkhorn_log",
            numItermax=max_iter, stopThr=stop_thr, log=True,
        )
        n_iter = int(log.get("niter", max_iter))
    elif method == "emd":
        print("Solving exact OT (network simplex, CPU) ...", flush=True)
        P = ot.emd(
            a.double().cpu().numpy(), b.double().cpu().numpy(),
            C_norm.double().cpu().numpy(), numItermax=max_iter,
        )
        P = torch.as_tensor(P, dtype=torch.float32, device=C.device)
        n_iter = None
    elif method == "cvx":
        import cvxpy as cvx

        print(f"Solving OT as a conic programme ({cvx_solver}, reg={reg}) "
              f"-- {n * m:,} variables, this is the slow one ...", flush=True)
        a_np = a.double().cpu().numpy()
        b_np = b.double().cpu().numpy()
        C_np = C_norm.double().cpu().numpy()

        x = cvx.Variable((n, m), nonneg=True)
        objective = cvx.sum(cvx.multiply(C_np, x))
        if reg > 0:
            # Same KL penalty as omt/solver.py (0 reg -> plain LP).
            objective = objective + reg * cvx.sum(
                cvx.kl_div(x, a_np[:, None] @ b_np[None, :]))
        problem = cvx.Problem(cvx.Minimize(objective),
                              [x @ np.ones(m) == a_np, x.T @ np.ones(n) == b_np])
        problem.solve(solver=getattr(cvx, cvx_solver), max_iters=max_iter,
                      eps=stop_thr)
        if x.value is None:
            raise RuntimeError(f"cvxpy failed: status={problem.status}")
        P_np = np.asarray(x.value)
        P_np = np.clip(P_np, 0.0, None)
        P_np /= P_np.sum()
        P = torch.as_tensor(P_np, dtype=torch.float32, device=C.device)
        n_iter = problem.solver_stats.num_iters
        print(f"  cvxpy status: {problem.status}", flush=True)
    else:
        raise ValueError(f"unknown sample-level method: {method}")
    solve_time = time.time() - t0
    print(f"  solved in {solve_time:.1f}s", flush=True)

    info = {"solve_time_s": solve_time, "n_iter": n_iter,
            "cost_max": float(C.max()), "marginal_violation":
            float((P.sum(dim=1) - a).abs().max())}
    return P, C, info


def barycentric_map(P, Y_t):
    """T(x_i) = sum_j (P_ij / sum_j P_ij) y_j -- barycentric projection of the plan."""
    row_mass = P.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return (P @ Y_t) / row_mass


def _sample_categorical_rows(probs, rng):
    """Vectorised per-row categorical sampling (inverse-CDF).

    probs : (n, K) each row a probability vector.  Returns (n,) sampled indices.
    Equivalent to [rng.choice(K, p=probs[i]) for i in range(n)] but vectorised.
    """
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0                                # guard against fp drift
    r = rng.random((probs.shape[0], 1))
    return (r < cdf).argmax(axis=1)                 # first bin whose cdf exceeds r


def solve_omt(X_t, gmm_s_path, gmm_t_path, eps_gmm, method, eps_w,
              max_iter, stop_thr, cvx_solver="SCS", seed=0):
    """OMT: entropic-Gaussian OT between the *saved* diagonal mixtures.

    This is the real OMT of omt/solver.py, not sample-level OT and not Bures/W2
    GMM-OT.  The distinction:
      * GMM-OT  -> Bures-Wasserstein W2^2 between components (eps = 0).
      * OMT     -> the closed-form *entropic*-Gaussian transport distance
                   (omt/functions.py, eps_gmm > 0).

    The mixtures gmm_s / gmm_t are the ones fitted once by fit_gmm.py
    (GPU KMeans -> diagonal Gaussian params).  We only *load* them here; there is
    no GMM fitting in this script.

    Cost.  The component cost is the entropic-Gaussian distance
    `omt.functions.vectorized_eogt_dist(mu_s, mu_t, Sigma_s, Sigma_t, eps_gmm_mat)`.
    That routine forms `Sigma_s @ Sigma_t` and eigendecomposes it for every
    component pair -- with full d x d covariances at d=2048, K=1000 the pairwise
    tensor alone is ~268 TB, so it cannot run.  Our covariances are *diagonal*
    (that is how the mixtures were fit), and for diagonal covariances the eogt
    formula collapses exactly to `compute_distance_diagonal` in the same module:
    eig(AB) -> elementwise A*B, trace -> sum over the diagonal, every other term
    identical.  We therefore call that diagonal specialization -- same closed
    form, same numbers, tractable memory -- with the (K_s, K_t) `eps_gmm_mat`
    built exactly as omt/solver.py does.

    Weight coupling.  omt_w solves the same programme as gmmomt_vectorized:
      cvx      : min <C, w> + eps_w * KL(w || a b^T)   (cvxpy/SCS)
      sinkhorn : entropic OT between the component weights (eps_w capped at 0.1)

    Transport.  Diagonal Gaussian pushforward, argmax variant of
    functions.points_transport: x |-> (x - mu_i) * sqrt(var_j / var_i) + mu_j,
    with i the most likely source component of x and j = argmax_j omt_w[i, j].
    (points_transport instead *samples* i and j; argmax concentrates mass on the
    dominant target and scores higher on label preservation.)
    """
    from functions import compute_distance_diagonal   # omt/functions.py

    def load_gmm(path):
        print(f"Loading GMM from {path} ...", flush=True)
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["gmm"] if isinstance(d, dict) and "gmm" in d else d

    t0 = time.time()
    gmm_s = load_gmm(gmm_s_path)
    gmm_t = load_gmm(gmm_t_path)

    def diag_params(gmm):
        cov = gmm.covariances_
        if cov.ndim == 3:                       # full matrices stored -> take diagonal
            cov = np.ascontiguousarray(np.diagonal(cov, axis1=1, axis2=2))
        return gmm.means_, cov

    mu_s, var_s = diag_params(gmm_s)
    mu_t, var_t = diag_params(gmm_t)
    alpha_s = np.asarray(gmm_s.weights_, dtype=np.float64)
    alpha_t = np.asarray(gmm_t.weights_, dtype=np.float64)
    alpha_s = alpha_s / alpha_s.sum()
    alpha_t = alpha_t / alpha_t.sum()
    K_s, d = mu_s.shape
    K_t = mu_t.shape[0]
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"({K_s} x {K_t} components, d={d})", flush=True)

    # eps_gmm_mat -- the (K_s, K_t) matrix of entropic-Gaussian eps, built exactly
    # as omt/solver.py's gmmomt_vectorized does before calling vectorized_eogt_dist.
    eps_gmm_mat = np.full(K_s * K_t, eps_gmm).reshape(K_s, K_t)

    print(f"Computing entropic-Gaussian component cost "
          f"(eps_gmm={eps_gmm}, diagonal closed form) ...", flush=True)
    t0 = time.time()
    C_comp, _, _ = compute_distance_diagonal(mu_s, var_s, mu_t, var_t,
                                             eps=float(eps_gmm_mat[0, 0]),
                                             batch_size=100)
    C_comp = np.asarray(C_comp, dtype=np.float64)
    print(f"  cost in {time.time() - t0:.1f}s "
          f"(min={C_comp.min():.3f}, max={C_comp.max():.3f})", flush=True)
    # Normalise for the solve (as solve_sample_ot / solve_gmm_ot do); report the
    # transport cost on the raw entropic-Gaussian distances.
    C_norm = C_comp / C_comp.max()

    t0 = time.time()
    if method == "sinkhorn":
        reg = float(min(eps_w, 0.1))            # omt/solver.py caps eps_w at 0.1
        print(f"Solving component OT (sinkhorn, eps_w={reg}) ...", flush=True)
        omt_w, log = ot.bregman.sinkhorn(alpha_s, alpha_t, C_norm, reg,
                                         numItermax=max_iter, stopThr=stop_thr,
                                         log=True)
        n_iter = int(log.get("niter", max_iter))
    elif method == "cvx":
        import cvxpy as cvx

        print(f"Solving component OT (cvx/{cvx_solver}, eps_w={eps_w}) -- "
              f"{K_s * K_t:,} variables ...", flush=True)
        x = cvx.Variable((K_s, K_t), nonneg=True)
        objective = cvx.sum(cvx.multiply(C_norm, x))
        if eps_w > 0:                           # same KL penalty as gmmomt_vectorized
            objective = objective + eps_w * cvx.sum(
                cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :]))
        problem = cvx.Problem(cvx.Minimize(objective),
                              [x @ np.ones(K_t) == alpha_s,
                               x.T @ np.ones(K_s) == alpha_t])
        problem.solve(solver=getattr(cvx, cvx_solver), max_iters=max_iter,
                      eps=stop_thr)
        if x.value is None:
            raise RuntimeError(f"cvxpy failed: status={problem.status}")
        omt_w = np.clip(np.asarray(x.value), 0.0, None)
        omt_w /= omt_w.sum()
        n_iter = problem.solver_stats.num_iters
        print(f"  cvxpy status: {problem.status}", flush=True)
    else:
        raise ValueError(f"omt weight solver must be 'sinkhorn' or 'cvx', got {method}")
    solve_time = time.time() - t0
    comp_cost = float(np.sum(omt_w * C_comp))
    print(f"  solved in {solve_time:.1f}s | component transport cost: {comp_cost:.4f}",
          flush=True)

    # Point-level pushforward T(x) = (x - mu_i) * sqrt(var_j / var_i) + mu_j.
    # Argmax (greedy) variant of omt/functions.py:points_transport: rather than
    # *sampling* the source/target components (predict_proba / omt_w[i,:]/sum),
    # each x is assigned its most likely source component i and the single most
    # strongly coupled target component j = argmax_j omt_w[i, j].  This
    # concentrates mass on the dominant (usually class-correct) target and scores
    # higher on label preservation than the stochastic map, at the cost of not
    # conserving the target mixture marginal.
    eps = 1e-6
    X_np = X_t.cpu().numpy()
    comp_i = gmm_s.predict(X_np)                          # (n,) source component
    comp_j = np.argmax(omt_w, axis=1)[comp_i]             # (n,) matched target component
    dev = X_t.device
    mu_s_t = torch.as_tensor(mu_s, dtype=torch.float32, device=dev)
    mu_t_t = torch.as_tensor(mu_t, dtype=torch.float32, device=dev)
    var_s_t = torch.as_tensor(var_s, dtype=torch.float32, device=dev)
    var_t_t = torch.as_tensor(var_t, dtype=torch.float32, device=dev)
    i_t = torch.as_tensor(comp_i, dtype=torch.long, device=dev)
    j_t = torch.as_tensor(comp_j, dtype=torch.long, device=dev)
    scale = torch.sqrt(var_t_t[j_t] / (var_s_t[i_t] + eps))
    X_hat = (X_t - mu_s_t[i_t]) * scale + mu_t_t[j_t]

    info = {"solve_time_s": solve_time, "n_iter": n_iter,
            "eps_gmm": float(eps_gmm), "eps_w": float(eps_w),
            "weight_solver": method,
            "n_components_source": int(K_s),
            "n_components_target": int(K_t),
            "n_target_components_used": int(len(np.unique(comp_j))),
            "component_transport_cost": comp_cost}
    return X_hat, info


def solve_gmm_ot(X_t, gmm_path, reg, max_iter, stop_thr):
    """GMM-OT: OT between mixture components, then the per-pair Gaussian map.

    The mixtures are the ones fitted in gmm.py.  Each 1source point is
    assigned to its most likely source component i, that component is mapped to
    the target component j carrying most of the coupling mass of row i, and the
    point is pushed forward by the diagonal Gaussian transport map between them.
    """
    from functions import compute_distance_diagonal  # omt/functions.py

    print(f"Loading GMMs from {gmm_path} (large file, may take a while) ...", flush=True)
    t0 = time.time()
    with open(gmm_path, "rb") as f:
        gmm_dict = pickle.load(f)
    gmm_s, gmm_t = gmm_dict["gmm_s"], gmm_dict["gmm_t"]
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"({gmm_s.means_.shape[0]} x {gmm_t.means_.shape[0]} components)", flush=True)

    def diag_params(gmm):
        cov = gmm.covariances_
        if cov.ndim == 3:                       # full matrices stored -> take diagonal
            # np.diagonal returns a read-only view; copy so torch can wrap it.
            cov = np.ascontiguousarray(np.diagonal(cov, axis1=1, axis2=2))
        return gmm.means_, cov

    mu_s, var_s = diag_params(gmm_s)
    mu_t, var_t = diag_params(gmm_t)

    print("Computing component-level cost matrix ...", flush=True)
    C_comp, _, _ = compute_distance_diagonal(mu_s, var_s, mu_t, var_t,
                                             eps=1e-2, batch_size=100)
    C_comp_norm = C_comp / C_comp.max()

    print("Solving OT between mixture components (sinkhorn) ...", flush=True)
    t0 = time.time()
    omt_w, log = ot.bregman.sinkhorn(gmm_s.weights_, gmm_t.weights_,
                                     C_comp_norm.astype(np.float64), reg,
                                     numItermax=max_iter, stopThr=stop_thr, log=True)
    solve_time = time.time() - t0
    comp_cost = float(np.sum(omt_w * C_comp))
    print(f"  solved in {solve_time:.1f}s | component transport cost: {comp_cost:.4f}",
          flush=True)

    # Point-level pushforward T(x) = (x - mu_i) * sqrt(var_j / var_i) + mu_j.
    eps = 1e-6
    X_np = X_t.cpu().numpy()
    comp_i = gmm_s.predict(X_np)                          # (n,)
    comp_j = np.argmax(omt_w, axis=1)[comp_i]             # (n,) matched target component
    dev = X_t.device
    mu_s_t = torch.as_tensor(mu_s, dtype=torch.float32, device=dev)
    mu_t_t = torch.as_tensor(mu_t, dtype=torch.float32, device=dev)
    var_s_t = torch.as_tensor(var_s, dtype=torch.float32, device=dev)
    var_t_t = torch.as_tensor(var_t, dtype=torch.float32, device=dev)
    i_t = torch.as_tensor(comp_i, dtype=torch.long, device=dev)
    j_t = torch.as_tensor(comp_j, dtype=torch.long, device=dev)
    scale = torch.sqrt(var_t_t[j_t] / (var_s_t[i_t] + eps))
    X_hat = (X_t - mu_s_t[i_t]) * scale + mu_t_t[j_t]

    info = {"solve_time_s": solve_time,
            "n_iter": int(log.get("niter", max_iter)),
            "n_components_source": int(mu_s.shape[0]),
            "n_components_target": int(mu_t.shape[0]),
            "n_target_components_used": int(len(np.unique(np.argmax(omt_w, axis=1)))),
            "component_transport_cost": comp_cost}
    return X_hat, info


# ── Matching / metrics ───────────────────────────────────────────────────────
def nearest_target(X_hat, Y_t, chunk=2048):
    """Index (and squared distance) of the nearest target point for each T(x)."""
    idx, dist = [], []
    y_sq = (Y_t * Y_t).sum(dim=1)
    for start in range(0, X_hat.shape[0], chunk):
        xb = X_hat[start:start + chunk]
        d = (xb * xb).sum(dim=1, keepdim=True) + y_sq[None, :] - 2.0 * (xb @ Y_t.T)
        best = d.min(dim=1)
        idx.append(best.indices.cpu())
        dist.append(best.values.clamp(min=0.0).cpu())
    return torch.cat(idx).numpy(), torch.cat(dist).numpy()


def plan_weighted_class_accuracy(P, class_X, class_Y, chunk=2048):
    """sum_ij P_ij * 1[class(x_i) == class(y_j)] -- soft, mass-weighted agreement."""
    cy = torch.as_tensor(class_Y, device=P.device)
    total = 0.0
    for start in range(0, P.shape[0], chunk):
        cx = torch.as_tensor(class_X[start:start + chunk], device=P.device)
        same = (cx[:, None] == cy[None, :]).to(P.dtype)
        total += float((P[start:start + chunk] * same).sum())
    return total / float(P.sum())


def topk_hit(topk_idx, labels):
    return float(np.mean([labels[i] in topk_idx[i] for i in range(len(labels))]))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-pairs", type=int, default=10000,
                        help="approximate samples per side; rounded down to a whole "
                             "number of images per class (n_pairs // n_classes)")
    parser.add_argument("--per-class", type=int, default=None,
                        help="images per class per side; overrides --n-pairs")
    parser.add_argument("--method", default="sinkhorn",
                        choices=["sinkhorn", "emd", "cvx", "gmmot", "omt"],
                        help="OT solver: sample-level entropic (sinkhorn) / exact "
                             "(emd) / conic programme (cvx), GMM-OT (gmmot), or the "
                             "entropic-Gaussian OMT of omt/solver.py (omt)")
    parser.add_argument("--omt-solver", default="sinkhorn",
                        choices=["sinkhorn", "cvx"],
                        help="weight solver for --method omt (mirrors omt/solver.py)")
    parser.add_argument("--eps-gmm", type=float, default=0.01,
                        help="entropic-Gaussian regularisation for the OMT component "
                             "cost (vectorized_eogt_dist eps_gmm_mat)")
    parser.add_argument("--eps-w", type=float, default=0.01,
                        help="weight-coupling regularisation for --method omt "
                             "(entropic reg for sinkhorn / KL weight for cvx)")
    parser.add_argument("--gmm-s", default=None,
                        help="saved source mixture for --method omt "
                             "(default <model_dir>/gmm_s_n<n>_c<n_classes>_K<K>.pkl)")
    parser.add_argument("--gmm-t", default=None,
                        help="saved target mixture for --method omt "
                             "(default <model_dir>/gmm_t_n<n>_c<n_classes>_K<K>.pkl)")
    parser.add_argument("--n-components", type=int, default=1000,
                        help="K in the saved-mixture filename for --method omt")
    parser.add_argument("--reg", type=float, default=0.01,
                        help="regularisation: entropic for sinkhorn/gmmot, "
                             "KL weight for cvx (0 -> plain LP)")
    parser.add_argument("--cvx-solver", default="SCS",
                        help="cvxpy backend for --method cvx (SCS, ECOS, CLARABEL, ...)")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="use a random subset of this many classes (keeps the "
                             "split stratified; useful to size down for --method cvx)")
    parser.add_argument("--max-iter", type=int, default=int(1e6))
    parser.add_argument("--stop-thr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb", default=EMB_PATH)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
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
    X_t = torch.as_tensor(X, device=device)
    Y_t = torch.as_tensor(Y, device=device)

    # ResNet50 classifier head, used to label the transported samples.
    clf = load_classifier(RESNET_PATH, X.shape[1], device)

    # 3. Optimal transport in the latent space.
    P = None
    if args.method in ("sinkhorn", "emd", "cvx"):
        P, C, ot_info = solve_sample_ot(X_t, Y_t, args.method, args.reg,
                                        args.max_iter, args.stop_thr,
                                        cvx_solver=args.cvx_solver)
        X_hat = barycentric_map(P, Y_t)
        transport_cost = float((P * C).sum())               # <P, ||x-y||^2>
        hard_match = P.argmax(dim=1).cpu().numpy()          # dominant partner in Y
        soft_class_acc = plan_weighted_class_accuracy(P, class_X, class_Y)
        del C
    elif args.method == "omt":
        tag = f"n{n_side}_c{n_classes}_K{args.n_components}"
        gmm_s_path = args.gmm_s or os.path.join(MODEL_DIR, f"gmm_s_{tag}.pkl")
        gmm_t_path = args.gmm_t or os.path.join(MODEL_DIR, f"gmm_t_{tag}.pkl")
        X_hat, ot_info = solve_omt(X_t, gmm_s_path, gmm_t_path, args.eps_gmm,
                                   args.omt_solver, args.eps_w,
                                   args.max_iter, args.stop_thr,
                                   cvx_solver=args.cvx_solver, seed=args.seed)
        transport_cost = None
        hard_match, soft_class_acc = None, None
    else:
        X_hat, ot_info = solve_gmm_ot(X_t, GMM_PATH, args.reg,
                                      args.max_iter, args.stop_thr)
        transport_cost = None
        hard_match, soft_class_acc = None, None

    torch.cuda.empty_cache() if device.type == "cuda" else None

    # Displacement of the map and the closest actual target image for each T(x).
    displacement = float(torch.norm(X_t - X_hat, dim=1).mean())
    nn_idx, nn_sq = nearest_target(X_hat, Y_t)
    nn_dist = float(np.sqrt(nn_sq).mean())
    if hard_match is None:
        hard_match = nn_idx                                 # GMM-OT: match = nearest T(x)

    # 4a. Class preservation of the matching (source class vs matched target class).
    matched_class_acc = float(np.mean(class_Y[hard_match] == class_X))
    nn_class_acc = float(np.mean(class_Y[nn_idx] == class_X))
    rng = np.random.default_rng(args.seed + 1)
    chance_acc = float(np.mean(class_Y[rng.permutation(len(class_Y))] == class_X))

    # 4b. Class preservation under the ResNet50 classifier head.
    X_hat_np = X_hat.cpu().numpy()
    pred_X, top5_X = predict(clf, X, device)
    pred_Y, _ = predict(clf, Y, device)
    pred_hat, top5_hat = predict(clf, X_hat_np, device)

    metrics = {
        "method": args.method,
        "n_per_side": int(n_side),
        "n_classes": int(n_classes),
        "per_class_per_side": int(per_class),
        "dim": int(X.shape[1]),
        "reg": args.reg,
        "seed": args.seed,
        # ── transport cost ──
        "transport_cost_plan": transport_cost,                # <P, C>, sample-level only
        "mean_displacement_L2": displacement,                 # ||x - T(x)||
        "mean_displacement_L2_per_dim": displacement / X.shape[1],
        "mean_dist_to_nearest_target": nn_dist,               # ||T(x) - y_NN||
        # ── class preservation of the alignment ──
        "matched_class_accuracy": matched_class_acc,          # class(x) == class(matched y)
        "nearest_neighbour_class_accuracy": nn_class_acc,
        "plan_weighted_class_accuracy": soft_class_acc,
        "random_matching_class_accuracy": chance_acc,         # chance baseline
        # ── ResNet50 classifier on the transported points ──
        "clf_top1_source": float(np.mean(pred_X == class_X)),
        "clf_top1_target": float(np.mean(pred_Y == class_Y)),
        "clf_top1_transported": float(np.mean(pred_hat == class_X)),
        "clf_top5_source": topk_hit(top5_X, class_X),
        "clf_top5_transported": topk_hit(top5_hat, class_X),
        "label_preservation_top1": float(np.mean(pred_hat == pred_X)),
        "label_preservation_top5": topk_hit(top5_hat, pred_X),
        "clf_agreement_with_matched_target": float(np.mean(pred_hat == pred_Y[hard_match])),
        "n_distinct_targets_matched": int(len(np.unique(hard_match))),
    }
    metrics.update({f"ot_{k}": v for k, v in ot_info.items()})

    print("\n" + "=" * 68)
    print(f"OT alignment of ImageNet embeddings  |  method={args.method}  "
          f"n={n_side} per side ({per_class}/class x {n_classes} classes)")
    print("=" * 68)
    print("-- transport cost")
    if transport_cost is not None:
        print(f"  <P, C> (squared Euclidean)          : {transport_cost:.4f}")
    print(f"  mean ||x - T(x)||                   : {displacement:.4f} "
          f"({displacement / X.shape[1]:.3e} per dim)")
    print(f"  mean ||T(x) - nearest y||           : {nn_dist:.4f}")
    print("-- class preservation of the matching")
    print(f"  matched-pair class accuracy         : {matched_class_acc:.4f}")
    print(f"  nearest-neighbour class accuracy    : {nn_class_acc:.4f}")
    if soft_class_acc is not None:
        print(f"  plan-mass-weighted class accuracy   : {soft_class_acc:.4f}")
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
    print("=" * 68 + "\n")

    if args.method == "omt":
        default_out = os.path.join(
            OUT_DIR, f"imagenet_omt_{args.omt_solver}_n{n_side}_c{n_classes}"
                     f"_K{args.n_components}.json")
    else:
        default_out = os.path.join(
            OUT_DIR, f"imagenet_omt_{args.method}_n{n_side}_c{n_classes}.json")
    out_path = args.out or default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {out_path}", flush=True)


if __name__ == "__main__":
    main()