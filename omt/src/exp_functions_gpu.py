"""
GPU-accelerated exponential mixture fitting using PyTorch.
Drop-in replacement for exp_fitting() from exp_functions.py.

Key optimization: M-step is batched across ALL dimensions simultaneously,
eliminating the D-loop bottleneck for high-dimensional data.
"""
import torch
import numpy as np
import time
from scipy.special import erf, erfinv


def _to_device(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.float64)
    return x.to(device=device, dtype=torch.float64)


# -------------------------------------------------------
# GPU log-PDFs (vectorized over N samples, K components)
# -------------------------------------------------------
def _log_pdf_asym_exp_nd_gpu(X, b, c_rate, d_rate):
    """X: (N,D), b/c_rate/d_rate: (D,). Returns (N,)."""
    Z = 1.0 / c_rate + 1.0 / d_rate
    diff = X - b
    log_p = -torch.log(Z) + torch.where(diff >= 0, -c_rate * diff, d_rate * diff)
    return log_p.sum(dim=1)


def _log_pdf_gaussian_nd_gpu(X, mu, c_rate, d_rate):
    """X: (N,D), mu/c_rate/d_rate: (D,). Returns (N,)."""
    log_a = torch.log(2.0 * c_rate * d_rate) - 0.5 * np.log(np.pi) - torch.log(c_rate + d_rate)
    diff = X - mu
    log_p = log_a + torch.where(diff >= 0, -(diff ** 2) * (c_rate ** 2), -(diff ** 2) * (d_rate ** 2))
    return log_p.sum(dim=1)


def _log_pdf_all_components_gpu(X, comp_types, locs, c_rates, d_rates, weights):
    """Compute log R for all components at once. Returns (N, K)."""
    N = X.shape[0]
    K = len(comp_types)
    device = X.device
    log_R = torch.zeros(N, K, device=device, dtype=X.dtype)
    for k in range(K):
        lw = torch.log(weights[k] + 1e-300)
        if comp_types[k] == 'G':
            log_R[:, k] = lw + _log_pdf_gaussian_nd_gpu(X, locs[k], c_rates[k], d_rates[k])
        else:
            log_R[:, k] = lw + _log_pdf_asym_exp_nd_gpu(X, locs[k], c_rates[k], d_rates[k])
    return log_R


# -------------------------------------------------------
# BATCHED profile scan: all D dimensions at once
# -------------------------------------------------------
def _batched_profile_scan_gpu(X_masked, r_masked, N_k_eff):
    """
    Profile scan for ALL dimensions simultaneously.
    X_masked: (M, D) - masked points
    r_masked: (M,) - responsibilities
    Returns: locs (D,), c_rates (D,), d_rates (D,)
    """
    M, D = X_masked.shape
    device = X_masked.device
    dtype = X_masked.dtype

    if M < 3:
        return (X_masked[M // 2].clone(),
                torch.ones(D, device=device, dtype=dtype),
                torch.ones(D, device=device, dtype=dtype))

    # Sort each dimension independently: (M, D)
    order = torch.argsort(X_masked, dim=0)  # (M, D) indices
    X_sorted = torch.gather(X_masked, 0, order)  # (M, D)
    r_sorted = r_masked[order]  # (M, D) - broadcast responsibilities per dim sort order

    # Cumulative sums along sorted axis: (M+1, D)
    cum_rk = torch.zeros(M + 1, D, device=device, dtype=dtype)
    cum_rk[1:] = torch.cumsum(r_sorted, dim=0)

    cum_rkx = torch.zeros(M + 1, D, device=device, dtype=dtype)
    cum_rkx[1:] = torch.cumsum(r_sorted * X_sorted, dim=0)

    total_rX = cum_rkx[M]  # (D,)

    # Vectorized scan over split points: indices 1..M-1
    # W_L[i,d] = cum_rk[i+1, d], for i in 0..M-2
    W_L = cum_rk[1:M]  # (M-1, D)
    W_R = N_k_eff - W_L  # (M-1, D)
    sum_rX_L = cum_rkx[1:M]  # (M-1, D)
    sum_rX_R = total_rX.unsqueeze(0) - sum_rX_L  # (M-1, D)

    b_vals = X_sorted[1:M]  # (M-1, D)
    WS_R = sum_rX_R - W_R * b_vals  # (M-1, D)
    WS_L = W_L * b_vals - sum_rX_L  # (M-1, D)

    valid = (WS_R > 1e-10) & (WS_L > 1e-10) & (W_L > 1e-10) & (W_R > 1e-10)
    denom2 = W_L * WS_R + W_R * WS_L
    valid2 = valid & (denom2 > 1e-10)

    log_obj = torch.full((M - 1, D), -float('inf'), device=device, dtype=dtype)
    log_obj[valid2] = (torch.log(W_L[valid2] * W_R[valid2]) - torch.log(denom2[valid2]))

    # Best split per dimension
    best_idx = torch.argmax(log_obj, dim=0)  # (D,)
    d_idx = torch.arange(D, device=device)

    b_scan = b_vals[best_idx, d_idx]  # (D,)
    c_scan = torch.clamp(W_R[best_idx, d_idx] / WS_R[best_idx, d_idx].clamp(min=1e-10), min=1e-3)
    d_scan = torch.clamp(W_L[best_idx, d_idx] / WS_L[best_idx, d_idx].clamp(min=1e-10), min=1e-3)

    # Also try weighted median candidate (batched)
    cum_w_norm = cum_rk[1:] / (N_k_eff + 1e-300)  # (M, D)
    med_target = torch.full((D,), 0.5, device=device, dtype=dtype)
    med_idx = torch.zeros(D, device=device, dtype=torch.long)
    for d in range(D):
        med_idx[d] = torch.searchsorted(cum_w_norm[:, d].contiguous(), med_target[d])
    med_idx = torch.clamp(med_idx, 0, M - 1)
    b_median = X_sorted[med_idx, d_idx]  # (D,)

    # Evaluate LL for scan and median candidates, pick best per dim
    # Candidate 1: b_scan
    best_b, best_c, best_d = _eval_b_candidates_batched(
        X_masked, r_masked, N_k_eff,
        torch.stack([b_scan, b_median], dim=0))  # (2, D)

    return best_b, best_c, best_d


def _eval_b_candidates_batched(X_masked, r_masked, N_k_eff, candidates):
    """
    Evaluate multiple b candidates for all dimensions simultaneously.
    X_masked: (M, D), r_masked: (M,), candidates: (n_cands, D)
    Returns best (b, c, d) each of shape (D,)
    """
    M, D = X_masked.shape
    n_cands = candidates.shape[0]
    device = X_masked.device
    dtype = X_masked.dtype

    # (n_cands, M, D)
    diff = X_masked.unsqueeze(0) - candidates.unsqueeze(1)  # (n_cands, M, D)
    right = diff >= 0
    left = ~right

    r_exp = r_masked.unsqueeze(0).unsqueeze(2)  # (1, M, 1)
    wr = r_exp * right.float()  # (n_cands, M, D)
    wl = r_exp * left.float()

    W_R = wr.sum(dim=1)  # (n_cands, D)
    W_L = wl.sum(dim=1)

    S_R = (wr * diff).sum(dim=1)  # (n_cands, D)
    S_L = -(wl * diff).sum(dim=1)

    valid = (W_R > 1e-10) & (W_L > 1e-10) & (S_R > 1e-10) & (S_L > 1e-10)

    c_cand = torch.clamp(W_R / S_R.clamp(min=1e-10), min=1e-3)  # (n_cands, D)
    d_cand = torch.clamp(W_L / S_L.clamp(min=1e-10), min=1e-3)

    log_a = torch.log(c_cand * d_cand / (c_cand + d_cand) + 1e-300)  # (n_cands, D)

    # LL per candidate per dim: sum over M
    ll_per_point = r_exp * torch.where(diff >= 0, -c_cand.unsqueeze(1) * diff,
                                        d_cand.unsqueeze(1) * diff)  # (n_cands, M, D)
    ll = N_k_eff * log_a + ll_per_point.sum(dim=1)  # (n_cands, D)
    ll[~valid] = -float('inf')

    # Pick best per dim
    best_cand_idx = torch.argmax(ll, dim=0)  # (D,)
    d_idx = torch.arange(D, device=device)

    best_b = candidates[best_cand_idx, d_idx]
    best_c = c_cand[best_cand_idx, d_idx]
    best_d = d_cand[best_cand_idx, d_idx]

    return best_b, best_c, best_d


# -------------------------------------------------------
# Batched M-step for Exp component (ALL dims at once)
# -------------------------------------------------------
def _mstep_exp_batched_gpu(X, r_k, max_rate, K):
    """M-step for one Exp component, ALL dimensions batched."""
    N, D = X.shape
    device = X.device

    mask = r_k > (1e-2 / K)
    if mask.sum() < 5:
        mask = r_k > 0

    X_masked = X[mask]
    r_masked = r_k[mask]
    N_k_eff = r_masked.sum()

    locs_k, c_rates_k, d_rates_k = _batched_profile_scan_gpu(X_masked, r_masked, N_k_eff)
    c_rates_k = torch.clamp(c_rates_k, max=max_rate)
    d_rates_k = torch.clamp(d_rates_k, max=max_rate)

    return locs_k, c_rates_k, d_rates_k


# -------------------------------------------------------
# Batched M-step for Gaussian component (ALL dims at once)
# -------------------------------------------------------
def _mstep_gauss_batched_gpu(X, r_k, sorted_indices, max_rate):
    """M-step for one Gaussian component, all dims batched."""
    N, D = X.shape
    device = X.device
    dtype = X.dtype

    n_cands = 10
    qpos = torch.linspace(0.05, 0.95, n_cands, device=device, dtype=dtype)
    qidx = torch.clamp((qpos * N).long() - 1, 0, N - 1)

    # Get candidates for all dims: (n_cands, D)
    candidates = torch.zeros(n_cands, D, device=device, dtype=dtype)
    for d in range(D):
        sidx = sorted_indices[d]
        candidates[:, d] = X[sidx[qidx], d]

    # Vectorized: (n_cands, N, D)
    diff_all = X.unsqueeze(0) - candidates.unsqueeze(1)  # (n_cands, N, D)
    right_m = diff_all >= 0
    left_m = ~right_m

    r_exp = r_k.view(1, N, 1)  # (1, N, 1)
    wr = r_exp * right_m.float()  # (n_cands, N, D)
    wl = r_exp * left_m.float()

    Nr = wr.sum(dim=1)  # (n_cands, D)
    Nl = wl.sum(dim=1)

    valid_mask = (Nr > 1e-10) & (Nl > 1e-10)

    rms_r = torch.sqrt(torch.clamp((wr * diff_all**2).sum(dim=1) / Nr.clamp(min=1e-10), min=1e-6))
    rms_l = torch.sqrt(torch.clamp((wl * diff_all**2).sum(dim=1) / Nl.clamp(min=1e-10), min=1e-6))

    c_cand = 1.0 / (rms_r * np.sqrt(2.0))  # (n_cands, D)
    d_cand = 1.0 / (rms_l * np.sqrt(2.0))

    log_a = torch.log(2.0 * c_cand * d_cand) - 0.5 * np.log(np.pi) - torch.log(c_cand + d_cand)

    # LL: (n_cands, D)
    ll_per_point = r_exp * (log_a.unsqueeze(1) + torch.where(
        diff_all >= 0,
        -(diff_all**2) * (c_cand**2).unsqueeze(1),
        -(diff_all**2) * (d_cand**2).unsqueeze(1)))  # (n_cands, N, D)
    ll = ll_per_point.sum(dim=1)  # (n_cands, D)
    ll[~valid_mask] = -float('inf')

    # Best per dim
    best_idx = torch.argmax(ll, dim=0)  # (D,)
    d_idx = torch.arange(D, device=device)

    locs_k = candidates[best_idx, d_idx]
    c_rates_k = c_cand[best_idx, d_idx]
    d_rates_k = d_cand[best_idx, d_idx]

    return locs_k, c_rates_k, d_rates_k


# -------------------------------------------------------
# Main GPU EM loop (batched M-step)
# -------------------------------------------------------
def _run_em_loop_gpu(X, comp_types, locs, c_rates, d_rates, weights,
                     sorted_indices, n_iter, tol, verbose=False):
    N, D = X.shape
    K = len(comp_types)
    device = X.device

    global_std = X.std(dim=0)
    max_rate = 1.0 / (torch.clamp(global_std, min=1e-3) * 0.05)

    loglike_prev = torch.tensor(-float('inf'), device=device)

    for it in range(n_iter):
        # E-step
        log_R = _log_pdf_all_components_gpu(X, comp_types, locs, c_rates, d_rates, weights)
        log_total = torch.logsumexp(log_R, dim=1)
        loglike = log_total.sum()
        R = torch.exp(log_R - log_total.unsqueeze(1))

        if torch.abs(loglike - loglike_prev) < tol:
            if verbose:
                print(f"  EM converged at iteration {it}")
            break
        loglike_prev = loglike

        # M-step (batched across all dims)
        for k in range(K):
            r_k = R[:, k]
            N_k = r_k.sum()
            if N_k < 1e-10:
                continue

            if comp_types[k] == 'G':
                locs[k], c_rates[k], d_rates[k] = _mstep_gauss_batched_gpu(
                    X, r_k, sorted_indices, max_rate)
            else:
                locs[k], c_rates[k], d_rates[k] = _mstep_exp_batched_gpu(
                    X, r_k, max_rate, K)

        weights = R.sum(dim=0) / N

    # Compute BIC
    log_comp = _log_pdf_all_components_gpu(X, comp_types, locs, c_rates, d_rates, weights)
    loglike_final = torch.logsumexp(log_comp, dim=1).sum()
    n_params = (K - 1) + K * 3 * D
    bic = n_params * np.log(N) - 2 * loglike_final.item()

    return dict(comp_types=comp_types, locs=locs, c_rates=c_rates,
                d_rates=d_rates, weights=weights, bic=bic, loglike=loglike_final.item())


# -------------------------------------------------------
# Initialization helpers
# -------------------------------------------------------
def _profile_scan_init_gpu(sorted_pts):
    """Unweighted profile scan for initialization."""
    n = sorted_pts.shape[0]
    device = sorted_pts.device
    if n < 3:
        return sorted_pts[n // 2], torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

    prefix = torch.zeros(n + 1, device=device, dtype=sorted_pts.dtype)
    prefix[1:] = torch.cumsum(sorted_pts, dim=0)
    total_sum = prefix[n]

    N_L = torch.arange(1, n, device=device, dtype=sorted_pts.dtype)
    N_R = n - N_L
    sum_L = prefix[1:n]
    sum_R = total_sum - sum_L

    b_vals = sorted_pts[1:n]
    S_R = sum_R - N_R * b_vals
    S_L = N_L * b_vals - sum_L

    valid = (S_R > 1e-10) & (S_L > 1e-10)
    if not valid.any():
        j = n // 2
        return sorted_pts[j], torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

    denom = N_L * S_R + N_R * S_L
    log_obj = torch.full((n - 1,), -float('inf'), device=device, dtype=sorted_pts.dtype)
    log_obj[valid] = torch.log(N_L[valid] * N_R[valid]) - torch.log(denom[valid])

    best = torch.argmax(log_obj)
    b = sorted_pts[best + 1]
    c_rate = torch.clamp(N_R[best] / S_R[best], min=1e-3)
    d_rate = torch.clamp(N_L[best] / S_L[best], min=1e-3)
    return b, c_rate, d_rate


def _batched_init_cluster(X_cluster, device):
    """Initialize one cluster's parameters, all dims at once."""
    M, D = X_cluster.shape
    if M < 3:
        return (X_cluster.mean(dim=0),
                torch.ones(D, device=device, dtype=X_cluster.dtype),
                torch.ones(D, device=device, dtype=X_cluster.dtype))

    # Sort each dim
    X_sorted, _ = torch.sort(X_cluster, dim=0)  # (M, D)
    n = M

    prefix = torch.zeros(n + 1, D, device=device, dtype=X_cluster.dtype)
    prefix[1:] = torch.cumsum(X_sorted, dim=0)
    total_sum = prefix[n]  # (D,)

    N_L = torch.arange(1, n, device=device, dtype=X_cluster.dtype).unsqueeze(1)  # (n-1, 1)
    N_R = n - N_L

    sum_L = prefix[1:n]  # (n-1, D)
    sum_R = total_sum.unsqueeze(0) - sum_L

    b_vals = X_sorted[1:n]  # (n-1, D)
    S_R = sum_R - N_R * b_vals
    S_L = N_L * b_vals - sum_L

    valid = (S_R > 1e-10) & (S_L > 1e-10)
    denom = N_L * S_R + N_R * S_L
    log_obj = torch.full((n - 1, D), -float('inf'), device=device, dtype=X_cluster.dtype)
    valid2 = valid & (denom > 1e-10)
    log_obj[valid2] = torch.log(N_L.expand_as(S_R)[valid2] * N_R.expand_as(S_R)[valid2]) - torch.log(denom[valid2])

    best_idx = torch.argmax(log_obj, dim=0)  # (D,)
    d_idx = torch.arange(D, device=device)

    # Handle dims with no valid split
    any_valid = valid2.any(dim=0)  # (D,)
    b = torch.where(any_valid, b_vals[best_idx, d_idx], X_sorted[n // 2])
    c_rate = torch.where(any_valid,
                         torch.clamp(N_R.squeeze(1)[best_idx] / S_R[best_idx, d_idx].clamp(min=1e-10), min=1e-3),
                         torch.ones(D, device=device, dtype=X_cluster.dtype))
    d_rate = torch.where(any_valid,
                         torch.clamp(N_L.squeeze(1)[best_idx] / S_L[best_idx, d_idx].clamp(min=1e-10), min=1e-3),
                         torch.ones(D, device=device, dtype=X_cluster.dtype))

    return b, c_rate, d_rate


# -------------------------------------------------------
# GPU exp_fitting entry point
# -------------------------------------------------------
def exp_fitting_gpu(X, K, n_iter=500, tol=1e-8, device=None, verbose=True):
    """
    GPU-accelerated exp-only mixture fitting.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor, shape (N, D)
    K : int, number of components
    n_iter : int, max EM iterations
    tol : float, convergence tolerance
    device : str or torch.device, default 'cuda' if available
    verbose : bool

    Returns
    -------
    dict compatible with exp_fitting() output
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X).to(device=device, dtype=torch.float64)
    else:
        X_t = X.to(device=device, dtype=torch.float64)

    N, D = X_t.shape
    if verbose:
        print(f"[GPU] Fitting Exp-only K={K}, N={N}, D={D}, device={device}")

    t0 = time.time()

    # Initialize with K-Means (on CPU, then transfer)
    from sklearn.cluster import KMeans
    X_np = X_t.cpu().numpy() if X_t.is_cuda else X_t.numpy()
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X_np)
    labels_np = km.labels_
    labels = torch.from_numpy(labels_np).to(device)

    # Initialize parameters (batched per cluster)
    comp_types = ['A'] * K
    locs = torch.zeros(K, D, device=device, dtype=torch.float64)
    c_rates = torch.zeros(K, D, device=device, dtype=torch.float64)
    d_rates = torch.zeros(K, D, device=device, dtype=torch.float64)

    for k in range(K):
        pts = X_t[labels == k]
        locs[k], c_rates[k], d_rates[k] = _batched_init_cluster(pts, device)

    weights = torch.ones(K, device=device, dtype=torch.float64) / K
    sorted_indices = [torch.argsort(X_t[:, d]) for d in range(D)]

    t_init = time.time()
    if verbose:
        print(f"  Initialization: {t_init - t0:.2f}s")

    # Run EM
    result = _run_em_loop_gpu(X_t, comp_types, locs, c_rates, d_rates,
                              weights, sorted_indices, n_iter, tol, verbose=verbose)

    t_em = time.time()
    if verbose:
        K_g = sum(1 for t in result['comp_types'] if t == 'G')
        print(f"  EM done: {K_g}G+{K-K_g}A  BIC={result['bic']:.1f}")
        print(f"  EM time: {t_em - t_init:.2f}s")
        print(f"  Total time: {t_em - t0:.2f}s")

    # Format output (convert back to numpy for compatibility)
    types = ['Exp' if t == 'A' else 'Gaussian' for t in result['comp_types']]
    out = dict(
        K=K,
        types=types,
        weights=result['weights'].cpu().numpy(),
        means=result['locs'].cpu().numpy(),
        c_rates=result['c_rates'].cpu().numpy(),
        d_rates=result['d_rates'].cpu().numpy(),
        bic=result['bic'],
        loglike=result['loglike'],
        n_iter=n_iter,
    )
    return out


# -------------------------------------------------------
# GPU W2 cost matrix
# -------------------------------------------------------
def vectorized_W2_cost_matrix_gpu(mean_s, c_rates_s, d_rates_s, types_s,
                                  mean_t, c_rates_t, d_rates_t, types_t,
                                  device=None):
    """GPU version of vectorized_W2_cost_matrix."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean_s = _to_device(mean_s, device)
    mean_t = _to_device(mean_t, device)
    c_rates_s = _to_device(c_rates_s, device)
    c_rates_t = _to_device(c_rates_t, device)
    d_rates_s = _to_device(d_rates_s, device)
    d_rates_t = _to_device(d_rates_t, device)

    K_s, D = mean_s.shape
    K_t = mean_t.shape[0]

    b_i = mean_s[:, None, :]
    b_j = mean_t[None, :, :]
    c_i = c_rates_s[:, None, :]
    c_j = c_rates_t[None, :, :]
    d_i = d_rates_s[:, None, :]
    d_j = d_rates_t[None, :, :]

    is_exp_s = torch.tensor([t == 'Exp' for t in types_s], device=device)
    is_exp_t = torch.tensor([t == 'Exp' for t in types_t], device=device)

    ee = _W2_exp_gpu(b_i, c_i, d_i, b_j, c_j, d_j)
    gg = _W2_gaussian_gpu(b_i, c_i, d_i, b_j, c_j, d_j)
    eg = _W2_exp_gauss_gpu(b_i, c_i, d_i, b_j, c_j, d_j)
    ge = _W2_exp_gauss_gpu(b_j, c_j, d_j, b_i, c_i, d_i)

    m_ee = (is_exp_s[:, None] & is_exp_t[None, :]).float()
    m_gg = ((~is_exp_s)[:, None] & (~is_exp_t)[None, :]).float()
    m_eg = (is_exp_s[:, None] & (~is_exp_t)[None, :]).float()
    m_ge = ((~is_exp_s)[:, None] & is_exp_t[None, :]).float()

    C = (m_ee * ee.sum(dim=-1) + m_gg * gg.sum(dim=-1) +
         m_eg * eg.sum(dim=-1) + m_ge * ge.sum(dim=-1))

    return C.cpu().numpy()


def _W2_exp_gpu(b_i, c_i, d_i, b_j, c_j, d_j):
    diff_b = b_i - b_j
    cross_const = 2.0 - (np.pi**2 / 6.0)
    w2_cc = diff_b**2 + 2*diff_b*(1/c_i - 1/c_j) + 2*(1/c_i - 1/c_j)**2
    w2_cd = diff_b**2 + 2*diff_b*(1/c_i + 1/d_j) + 2/c_i**2 + 2/d_j**2 + 2/(c_i*d_j)*cross_const
    w2_dc = diff_b**2 - 2*diff_b*(1/d_i + 1/c_j) + 2/d_i**2 + 2/c_j**2 + 2/(d_i*c_j)*cross_const
    w2_dd = diff_b**2 - 2*diff_b*(1/d_i - 1/d_j) + 2*(1/d_i - 1/d_j)**2
    return w2_cc + w2_cd + w2_dc + w2_dd


def _W2_gaussian_gpu(b_i, c_i, d_i, b_j, c_j, d_j):
    diff_b = b_i - b_j
    k1, k2, k3 = 1.0/np.sqrt(np.pi), 0.5, 0.0812625
    w2_cc = diff_b**2 + 2*k1*diff_b*(1/c_i - 1/c_j) + k2*(1/c_i - 1/c_j)**2
    w2_cd = diff_b**2 + 2*k1*diff_b*(1/c_i + 1/d_j) + k2/c_i**2 + k2/d_j**2 + 2*(k3/(c_i*d_j))
    w2_dc = diff_b**2 - 2*k1*diff_b*(1/d_i + 1/c_j) + k2/d_i**2 + k2/c_j**2 + 2*(k3/(d_i*c_j))
    w2_dd = diff_b**2 - 2*k1*diff_b*(1/d_i - 1/d_j) + k2*(1/d_i - 1/d_j)**2
    return w2_cc + w2_cd + w2_dc + w2_dd


def _W2_exp_gauss_gpu(b_e, c_e, d_e, b_g, c_g, d_g):
    diff_b = b_e - b_g
    kg_m, kg_v, ke_v = 1.0/np.sqrt(np.pi), 0.5, 2.0
    k_eg1, k_eg2 = -0.981344, -0.242783
    w2_rc_rc = diff_b**2 + 2*diff_b*(1/c_e - kg_m/c_g) + ke_v/c_e**2 + kg_v/c_g**2 + 2/(c_e*c_g)*k_eg1
    w2_rc_ld = diff_b**2 + 2*diff_b*(1/c_e + kg_m/d_g) + ke_v/c_e**2 + kg_v/d_g**2 - 2/(c_e*d_g)*k_eg1
    w2_ld_rc = diff_b**2 + 2*diff_b*(-1/d_e - kg_m/c_g) + ke_v/d_e**2 + kg_v/c_g**2 - 2/(d_e*c_g)*k_eg2
    w2_ld_ld = diff_b**2 + 2*diff_b*(-1/d_e + kg_m/d_g) + ke_v/d_e**2 + kg_v/d_g**2 + 2/(d_e*d_g)*k_eg2
    return w2_rc_rc + w2_rc_ld + w2_ld_rc + w2_ld_ld


# -------------------------------------------------------
# GPU transport maps
# -------------------------------------------------------
def _T_exp_exp_gpu(x, bi, ci, di, bj, cj, dj):
    """Vectorized transport Exp->Exp. x: (N, D), params: (D,) each."""
    ai = (ci * di) / (ci + di)
    aj = (cj * dj) / (cj + dj)

    F = torch.where(x >= bi,
                    1.0 - (ai / ci) * torch.exp(-ci * (x - bi)),
                    (ai / di) * torch.exp(di * (x - bi)))

    F_bj = aj / dj
    target_right = F >= F_bj

    arg_right = torch.clamp(cj * (1.0 - F) / aj, min=1e-300)
    arg_left = torch.clamp(dj * F / aj, min=1e-300)

    y = torch.where(target_right,
                    bj - (1.0 / cj) * torch.log(arg_right),
                    bj + (1.0 / dj) * torch.log(arg_left))
    return y


def _T_gauss_gauss_gpu(x, bi, ci, di, bj, cj, dj):
    """Vectorized transport Gauss->Gauss."""
    ai = (2.0 * ci * di) / (np.sqrt(np.pi) * (ci + di))
    aj = (2.0 * cj * dj) / (np.sqrt(np.pi) * (cj + dj))

    sqrt_pi_ai_half = np.sqrt(np.pi) * ai / 2.0
    F = torch.where(x >= bi,
                    sqrt_pi_ai_half * (1.0/di + (1.0/ci) * torch.erf(ci * (x - bi))),
                    sqrt_pi_ai_half * (1.0/di + (1.0/di) * torch.erf(di * (x - bi))))

    F_bj = np.sqrt(np.pi) * aj / (2.0 * dj)
    target_right = F >= F_bj

    arg_right = torch.clamp((2.0 * cj) / (np.sqrt(np.pi) * aj) * F - cj / dj, -1.0 + 1e-12, 1.0 - 1e-12)
    arg_left = torch.clamp((2.0 * dj) / (np.sqrt(np.pi) * aj) * F - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)

    y = torch.where(target_right,
                    bj + (1.0 / cj) * torch.erfinv(arg_right),
                    bj + (1.0 / dj) * torch.erfinv(arg_left))
    return y


def _T_exp_gauss_gpu(x, bi, ci, di, bj, cj, dj):
    """Vectorized transport Exp->Gauss."""
    ai = (ci * di) / (ci + di)
    aj = (2.0 * cj * dj) / (np.sqrt(np.pi) * (cj + dj))

    F = torch.where(x >= bi,
                    1.0 - (ai / ci) * torch.exp(-ci * (x - bi)),
                    (ai / di) * torch.exp(di * (x - bi)))

    F_bj = np.sqrt(np.pi) * aj / (2.0 * dj)
    target_right = F >= F_bj

    arg_right = torch.clamp((2.0 * cj) / (np.sqrt(np.pi) * aj) * F - cj / dj, -1.0 + 1e-12, 1.0 - 1e-12)
    arg_left = torch.clamp((2.0 * dj) / (np.sqrt(np.pi) * aj) * F - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)

    y = torch.where(target_right,
                    bj + (1.0 / cj) * torch.erfinv(arg_right),
                    bj + (1.0 / dj) * torch.erfinv(arg_left))
    return y


def _T_gauss_exp_gpu(x, bi, ci, di, bj, cj, dj):
    """Vectorized transport Gauss->Exp."""
    ai = (2.0 * ci * di) / (np.sqrt(np.pi) * (ci + di))
    aj = (cj * dj) / (cj + dj)

    sqrt_pi_ai_half = np.sqrt(np.pi) * ai / 2.0
    F = torch.where(x >= bi,
                    sqrt_pi_ai_half * (1.0/di + (1.0/ci) * torch.erf(ci * (x - bi))),
                    sqrt_pi_ai_half * (1.0/di + (1.0/di) * torch.erf(di * (x - bi))))

    F_bj = aj / dj
    target_right = F >= F_bj

    arg_right = torch.clamp(cj * (1.0 - F) / aj, min=1e-300)
    arg_left = torch.clamp(dj * F / aj, min=1e-300)

    y = torch.where(target_right,
                    bj - (1.0 / cj) * torch.log(arg_right),
                    bj + (1.0 / dj) * torch.log(arg_left))
    return y


# -------------------------------------------------------
# GPU points transport
# -------------------------------------------------------
def points_transport_gpu(x, weights, solver, device=None):
    """
    GPU-accelerated point transport.
    x: (N, D) np.ndarray or torch.Tensor
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mm_s = solver['mm_s']
    mm_t = solver['mm_t']
    types_s = mm_s['types']
    types_t = mm_t['types']
    K_0 = mm_s['K']
    K_1 = mm_t['K']

    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x).to(device=device, dtype=torch.float64)
    else:
        x_t = x.to(device=device, dtype=torch.float64)

    w_t = _to_device(weights, device)
    N, D = x_t.shape

    b_s = _to_device(mm_s['means'], device)
    c_s = _to_device(mm_s['c_rates'], device)
    d_s = _to_device(mm_s['d_rates'], device)
    b_t = _to_device(mm_t['means'], device)
    c_t = _to_device(mm_t['c_rates'], device)
    d_t = _to_device(mm_t['d_rates'], device)

    # Assign each point to nearest source component
    dists = torch.cdist(x_t, b_s)  # (N, K_0)
    exp_i = torch.argmin(dists, dim=1)  # (N,)

    # Sample target component for each point
    w_rows = w_t[exp_i]  # (N, K_1)
    w_rows = w_rows / w_rows.sum(dim=1, keepdim=True)
    exp_j = torch.multinomial(w_rows.float(), 1).squeeze(1)  # (N,)

    # Transport all points (vectorized per (i,j) pair)
    all_exp_s = all(t in ('Exp', 'A') for t in types_s)
    all_exp_t = all(t in ('Exp', 'A') for t in types_t)

    T_x = torch.zeros_like(x_t)

    if all_exp_s and all_exp_t:
        n_sigmas = 3.0
        clamp_upper = b_t + n_sigmas / c_t
        clamp_lower = b_t - n_sigmas / d_t

        for i in range(K_0):
            for j in range(K_1):
                mask = (exp_i == i) & (exp_j == j)
                if not mask.any():
                    continue
                pts = x_t[mask]  # (M, D)
                transported = _T_exp_exp_gpu(pts, b_s[i], c_s[i], d_s[i], b_t[j], c_t[j], d_t[j])
                transported = torch.clamp(transported, clamp_lower[j], clamp_upper[j])
                T_x[mask] = transported
    else:
        n_sigmas = 3.0
        clamp_upper_t = b_t + n_sigmas / c_t
        clamp_lower_t = b_t - n_sigmas / d_t

        for i in range(K_0):
            for j in range(K_1):
                mask = (exp_i == i) & (exp_j == j)
                if not mask.any():
                    continue
                pts = x_t[mask]

                is_exp_s = types_s[i] in ('Exp', 'A')
                is_exp_t = types_t[j] in ('Exp', 'A')

                if is_exp_s and is_exp_t:
                    transported = _T_exp_exp_gpu(pts, b_s[i], c_s[i], d_s[i], b_t[j], c_t[j], d_t[j])
                elif (not is_exp_s) and (not is_exp_t):
                    transported = _T_gauss_gauss_gpu(pts, b_s[i], c_s[i], d_s[i], b_t[j], c_t[j], d_t[j])
                elif is_exp_s and (not is_exp_t):
                    transported = _T_exp_gauss_gpu(pts, b_s[i], c_s[i], d_s[i], b_t[j], c_t[j], d_t[j])
                else:
                    transported = _T_gauss_exp_gpu(pts, b_s[i], c_s[i], d_s[i], b_t[j], c_t[j], d_t[j])

                if is_exp_t:
                    transported = torch.clamp(transported, clamp_lower_t[j], clamp_upper_t[j])
                T_x[mask] = transported

    return T_x.cpu().numpy()


# -------------------------------------------------------
# Full GPU solver (drop-in replacement for omt())
# -------------------------------------------------------
def omt_gpu(
        data_source,
        data_target,
        n_components=(10, 30),
        eps_gmm=0.1,
        eps_w=0.0,
        method='cvx',
        fitting_mode_source='exp',
        fitting_mode_target='exp',
        cov_type='diag',
        max_iter=100000,
        n_iter_em=500,
        stop_thr=1e-6,
        verbose=True,
        device=None,
        ):
    """
    GPU-accelerated OMT solver. Drop-in for general_solver.omt() with exp fitting.
    Fits source/target on GPU, then solves transport on CPU (cvxpy/POT).
    """
    import cvxpy as cvx
    import ot

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    K0, K1 = n_components

    t0 = time.time()

    # Fit source
    if fitting_mode_source == 'exp':
        mm_s = exp_fitting_gpu(data_source, K=K0, n_iter=n_iter_em, device=device, verbose=verbose)
    else:
        from general_solver import gmm_fitting
        mm_s = gmm_fitting(data_source, K0, covariance_type=cov_type, verbose=verbose)

    # Fit target
    if fitting_mode_target == 'exp':
        mm_t = exp_fitting_gpu(data_target, K=K1, n_iter=n_iter_em, device=device, verbose=verbose)
    else:
        from general_solver import gmm_fitting
        mm_t = gmm_fitting(data_target, K1, covariance_type=cov_type, verbose=verbose)

    t1 = time.time()
    if verbose:
        print(f"\nFitting time: {t1-t0:.2f}s")

    # Compute cost matrix on GPU
    C = vectorized_W2_cost_matrix_gpu(
        mm_s['means'], mm_s['c_rates'], mm_s['d_rates'], mm_s['types'],
        mm_t['means'], mm_t['c_rates'], mm_t['d_rates'], mm_t['types'],
        device=device)

    # Solve OT (on CPU, small problem)
    alpha_s = mm_s['weights']
    alpha_t = mm_t['weights']

    # Sanitize cost matrix
    if np.any(np.isnan(C)) or np.any(np.isinf(C)):
        C_max = np.nanmax(C[np.isfinite(C)]) if np.any(np.isfinite(C)) else 1e6
        C = np.nan_to_num(C, nan=C_max, posinf=C_max, neginf=0.0)
        if verbose:
            print(f"  Warning: cost matrix had inf/nan, clamped to {C_max:.2f}")

    if method == 'cvx':
        x = cvx.Variable((K0, K1), nonneg=True)
        A = np.ones(K1)
        B = np.ones(K0)
        constraints = [x @ A == alpha_s, x.T @ B == alpha_t]
        if eps_w > 0:
            objective = cvx.Minimize(cvx.sum(cvx.multiply(C, x)) +
                                     eps_w * cvx.sum(cvx.kl_div(x, alpha_s[:, None] @ alpha_t[None, :])))
        else:
            objective = cvx.Minimize(cvx.sum(cvx.multiply(C, x)))
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver=cvx.SCS, max_iters=int(max_iter), eps=stop_thr)
        omt_w = x.value
        if omt_w is None:
            raise RuntimeError(
                f"CVX solver failed (status: {problem.status}). "
                f"Cost matrix range: [{C.min():.2f}, {C.max():.2f}]. "
                f"Try increasing eps_w or using method='w2'.")
        omt_w /= np.sum(omt_w)
        total_cost = problem.value
        n_iter_ot = problem.solver_stats.num_iters
    elif method == 'sinkhorn':
        omt_w, log = ot.bregman.sinkhorn(alpha_s, alpha_t, C, max(eps_w, 0.1),
                                          numItermax=int(max_iter), stopThr=stop_thr, log=True)
        total_cost = np.sum(omt_w * C)
        n_iter_ot = log['niter']
    else:
        omt_w = ot.emd(alpha_s, alpha_t, C)
        total_cost = np.sum(omt_w * C)
        n_iter_ot = 0

    t2 = time.time()
    if verbose:
        print(f"Transport cost: {total_cost:.4f}")
        print(f"OT solve time: {t2-t1:.2f}s")
        print(f"Total time: {t2-t0:.2f}s")

    solver_dict = {
        'method': method,
        'total_cost': total_cost,
        'omt_time_taken': t2 - t1,
        'mm_time_taken': t1 - t0,
        'time': t2 - t0,
        'n_iter': n_iter_ot,
        'n_total_iter': mm_s.get('n_iter', 0) + mm_t.get('n_iter', 0) + n_iter_ot,
        'mm_s': mm_s,
        'mm_t': mm_t,
        'fitting_mode_source': fitting_mode_source,
        'fitting_mode_target': fitting_mode_target,
    }

    return omt_w, solver_dict
