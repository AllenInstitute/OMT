import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.signal import argrelextrema
from scipy.special import logsumexp
from scipy.special import erf, erfinv
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt



class DoubleExponential:
    """Eq. 11 with p=q=1. Supports scalar (1-D) and vector (disjoint multi-D)."""
    def __init__(self, b, c, d):
        self.b = np.asarray(b, dtype=float).squeeze()
        self.c = np.asarray(c, dtype=float).squeeze()
        self.d = np.asarray(d, dtype=float).squeeze()
        self.a = (self.c * self.d) / (self.c + self.d)
        self.ndim = self.b.ndim

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        per_dim = np.where(x >= self.b,
                           self.a * np.exp(-(x - self.b) * self.c),
                           self.a * np.exp((x - self.b) * self.d))
        if self.ndim > 0:
            return np.prod(per_dim, axis=-1)
        return per_dim

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= self.b,
                        1 - (self.a / self.c) * np.exp(-(x - self.b) * self.c),
                        (self.a / self.d) * np.exp((x - self.b) * self.d))

    def inv_cdf(self, u):
        u = np.asarray(u, dtype=float)
        threshold = self.a / self.d
        return np.where(u >= threshold,
                        self.b - (1/self.c) * np.log(self.c * (1 - u) / self.a),
                        self.b + (1/self.d) * np.log(self.d * u / self.a))

class GaussianLike:
    """Eq. 11 with p=q=2. Supports scalar (1-D) and vector (disjoint multi-D)."""
    def __init__(self, b, c, d):
        self.b = np.asarray(b, dtype=float).squeeze()
        self.c = np.asarray(c, dtype=float).squeeze()
        self.d = np.asarray(d, dtype=float).squeeze()
        self.a = (2 * self.c * self.d) / (np.sqrt(np.pi) * (self.c + self.d))
        self.ndim = self.b.ndim

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        per_dim = np.where(x >= self.b,
                           self.a * np.exp(-(x - self.b)**2 * self.c**2),
                           self.a * np.exp(-(x - self.b)**2 * self.d**2))
        if self.ndim > 0:
            return np.prod(per_dim, axis=-1)
        return per_dim

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        const = np.sqrt(np.pi) * self.a / 2
        return np.where(x >= self.b,
                        const * (1/self.d + (1/self.c) * erf((x - self.b) * self.c)),
                        const * (1/self.d + (1/self.d) * erf((x - self.b) * self.d)))

    def inv_cdf(self, u):
        u = np.asarray(u, dtype=float)
        threshold = (np.sqrt(np.pi) * self.a) / (2 * self.d)
        const_val = (np.sqrt(np.pi) * self.a)
        return np.where(u >= threshold,
                        self.b + (1/self.c) * erfinv((2 * self.c / const_val) * u - self.c / self.d),
                        self.b + (1/self.d) * erfinv((2 * self.d / const_val) * u - 1))

# --- Transport Map Function ---

def transport_map(x, source_dist, target_dist):
    """T_ij(x) = F_j^{-1}(F_i(x)). For disjoint multi-D, operates per-dimension."""
    u = source_dist.cdf(x)
    return target_dist.inv_cdf(u)

# --- Visualization Function ---

def plot_omt_coupling(source_mix, target_mix, weights, title="OMT Coupling", eps=.1, n_grid=400):
    """
    Plots a coupling map similar to the OMT figure.
    source_mix/target_mix: List of component distributions.
    weights: Matrix of shape (len(source_mix), len(target_mix)).
    """
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.1, wspace=0.1)

    ax_joint = fig.add_subplot(gs[1:, 1:])
    ax_src = fig.add_subplot(gs[1:, 0], sharey=ax_joint)
    ax_tgt = fig.add_subplot(gs[0, 1:], sharex=ax_joint)

    # Auto-compute plotting ranges from distribution support.
    # y-axis = source, x-axis = target; use separate ranges for each.
    probe = np.linspace(-50, 50, 2000)
    all_src = sum(dist.pdf(probe) for dist in source_mix)
    all_tgt = sum(dist.pdf(probe) for dist in target_mix)
    src_thr = 1e-3 * all_src.ravel().max()
    tgt_thr = 1e-3 * all_tgt.ravel().max()
    src_support = probe[all_src.ravel() > src_thr]
    tgt_support = probe[all_tgt.ravel() > tgt_thr]
    y_lo, y_hi = src_support.min() - 1, src_support.max() + 1
    x_lo, x_hi = tgt_support.min() - 1, tgt_support.max() + 1

    y_range = np.linspace(y_lo, y_hi, n_grid)
    x_range = np.linspace(x_lo, x_hi, n_grid)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate Marginals
    src_pdf = np.sum([w_i * dist.pdf(y_range) for w_i, dist in zip(weights.sum(1), source_mix)], axis=0)
    tgt_pdf = np.sum([w_j * dist.pdf(x_range) for w_j, dist in zip(weights.sum(0), target_mix)], axis=0)

    # Joint density: coupling concentrated on T_ij(y).
    # π_{ij}(x,y) = w_{ij} · f_i(y) · δ(x − T_ij(y)), smoothed with a
    # normalized Gaussian kernel so that marginals are preserved.
    joint_pdf = np.zeros_like(X)
    norm_kernel = 1.0 / (eps * np.sqrt(2.0 * np.pi))
    for i, s_dist in enumerate(source_mix):
        for j, t_dist in enumerate(target_mix):
            if weights[i, j] < 1e-12:
                continue
            mapped_x = transport_map(y_range, s_dist, t_dist).ravel()
            src_density = s_dist.pdf(y_range).ravel()
            valid = np.isfinite(mapped_x) & (mapped_x >= x_lo) & (mapped_x <= x_hi)
            mapped_x = np.where(valid, mapped_x, 0.0)
            coeff = np.where(valid, weights[i, j] * src_density * norm_kernel, 0.0)
            joint_pdf += coeff[:, None] * \
                         np.exp(-0.5 * ((x_range[None, :] - mapped_x[:, None]) / eps)**2)

    # Plotting
    ax_src.fill_betweenx(y_range, 0, src_pdf, color='#ff715b', alpha=0.8)
    ax_tgt.fill_between(x_range, 0, tgt_pdf, color='#29335C', alpha=0.8)
    jmax = joint_pdf.max()
    if jmax > 0:
        levels = np.concatenate([[0], np.geomspace(jmax * 1e-3, jmax, 25)])
    else:
        levels = 20
    ax_joint.contourf(X, Y, joint_pdf, cmap='YlOrBr', levels=levels)

    ax_joint.invert_yaxis()

    # Clean up axes
    ax_src.invert_xaxis()
    ax_src.set_axis_off()
    ax_tgt.set_axis_off()
    ax_joint.grid(False)
    ax_joint.tick_params(axis='x', labelbottom=False) 
    ax_joint.spines['top'].set_visible(False)
    ax_joint.spines['right'].set_visible(False)
    ax_joint.spines['left'].set_visible(False)
    ax_joint.spines['bottom'].set_visible(False)
    ax_joint.tick_params(axis='y', labelleft=False)  
    ax_joint.tick_params(axis='x', labelbottom=False)
    for tick in ax_joint.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax_joint.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    plt.suptitle(title, y=0.95, fontsize=14)
    plt.show()
    

# -------------------------------------------------------
# 1-D PDFs
# -------------------------------------------------------
def asym_exp_pdf(x, b, c_rate, d_rate):
    """1-D asymmetric exponential PDF (scalar or array x)."""
    Z = 1.0 / c_rate + 1.0 / d_rate
    diff = x - b
    return (1.0 / Z) * np.where(diff >= 0,
                                 np.exp(-c_rate * diff),
                                 np.exp( d_rate * diff))


# -------------------------------------------------------
# Multi-D log-PDFs  (diagonal covariance)
# -------------------------------------------------------
def _log_pdf_gaussian_nd(X, mu, c_rate, d_rate):
    """GaussianLike log-PDF, multi-D diagonal.

    Matches GaussianLike class: a*exp(-(x-b)^2 * c^2) for x>=b, d^2 for x<b.
    X: (N,D), mu/c_rate/d_rate: (D,).  Returns (N,).
    """
    log_a = np.log(2.0 * c_rate * d_rate) - 0.5 * np.log(np.pi) - np.log(c_rate + d_rate)
    diff = X - mu                                           # (N,D)
    log_p = log_a + np.where(diff >= 0,
                             -(diff ** 2) * (c_rate ** 2),
                             -(diff ** 2) * (d_rate ** 2))
    return np.sum(log_p, axis=1)                            # (N,)


def _log_pdf_asym_exp_nd(X, b, c_rate, d_rate):
    """X: (N,D), b/c_rate/d_rate: (D,).  Returns (N,)."""
    Z = 1.0 / c_rate + 1.0 / d_rate          # (D,)
    diff = X - b                               # (N,D)
    log_p = -np.log(Z) + np.where(diff >= 0,
                                   -c_rate * diff,
                                    d_rate * diff)
    return np.sum(log_p, axis=1)               # (N,)


# -------------------------------------------------------
# 1-D profile-likelihood scans  (unchanged)
# -------------------------------------------------------
def _profile_scan_unweighted(sorted_pts):
    n = len(sorted_pts)
    prefix = np.concatenate(([0.0], np.cumsum(sorted_pts)))
    total_sum = prefix[n]

    N_L = np.arange(1, n, dtype=np.float64)
    N_R = n - N_L
    sum_L = prefix[1:n]
    sum_R = total_sum - sum_L

    b_vals = sorted_pts[1:n]
    S_R = sum_R - N_R * b_vals
    S_L = N_L * b_vals - sum_L

    valid = (S_R > 1e-10) & (S_L > 1e-10)
    if not valid.any():
        j = n // 2
        b = sorted_pts[j]
        c_rate = d_rate = 1.0
        ll = np.sum(np.log(asym_exp_pdf(sorted_pts, b, c_rate, d_rate) + 1e-300))
        return b, c_rate, d_rate, ll

    denom = N_L * S_R + N_R * S_L
    log_obj = np.full(n - 1, -np.inf)
    log_obj[valid] = np.log(N_L[valid] * N_R[valid]) - np.log(denom[valid])

    best = np.argmax(log_obj)
    b = sorted_pts[best + 1]
    c_rate = max(N_R[best] / S_R[best], 1e-3)
    d_rate = max(N_L[best] / S_L[best], 1e-3)
    ll = n * log_obj[best] - n
    return b, c_rate, d_rate, ll


def _profile_scan_weighted(X_sorted, r_k_sorted, N_k):
    n = len(X_sorted)
    cum_rk  = np.concatenate(([0.0], np.cumsum(r_k_sorted)))
    cum_rkx = np.concatenate(([0.0], np.cumsum(r_k_sorted * X_sorted)))
    total_rX = cum_rkx[n]

    W_L = cum_rk[1:n]
    W_R = N_k - W_L
    sum_rX_L = cum_rkx[1:n]
    sum_rX_R = total_rX - sum_rX_L

    b_vals = X_sorted[1:n]
    WS_R = sum_rX_R - W_R * b_vals
    WS_L = W_L * b_vals - sum_rX_L

    valid = (WS_R > 1e-10) & (WS_L > 1e-10) & (W_L > 1e-10) & (W_R > 1e-10)
    if not valid.any():
        return X_sorted[n // 2], 1.0, 1.0

    denom2 = W_L * WS_R + W_R * WS_L
    log_obj = np.full(n - 1, -np.inf)
    valid2 = valid & (denom2 > 1e-10)
    log_obj[valid2] = np.log(W_L[valid2] * W_R[valid2]) - np.log(denom2[valid2])

    best = np.argmax(log_obj)
    b = X_sorted[best + 1]
    c_rate = max(W_R[best] / WS_R[best], 1e-3)
    d_rate = max(W_L[best] / WS_L[best], 1e-3)
    return b, c_rate, d_rate


# -------------------------------------------------------
# 1-D fit helpers  (for initial per-cluster type selection)
# -------------------------------------------------------
def _fit_gaussian_at_b(pts, b):
    """Fit c and d for a fixed b, return (b, c, d, log-likelihood)."""
    right = pts[pts >= b] - b
    left  = b - pts[pts < b]
    rms_r = max(np.sqrt((right ** 2).mean()) if len(right) > 0 else 1e-3, 1e-3)
    rms_l = max(np.sqrt((left  ** 2).mean()) if len(left)  > 0 else 1e-3, 1e-3)
    c = max(1.0 / (rms_r * np.sqrt(2.0)), 1e-3)
    d = max(1.0 / (rms_l * np.sqrt(2.0)), 1e-3)
    log_a = np.log(2.0 * c * d) - 0.5 * np.log(np.pi) - np.log(c + d)
    diff = pts - b
    ll = np.sum(log_a + np.where(diff >= 0, -(diff ** 2) * c ** 2,
                                              -(diff ** 2) * d ** 2))
    return b, c, d, ll


def fit_gaussian(pts):
    """Fit a 1-D asymmetric GaussianLike via profile scan over b.

    GaussianLike PDF: a * exp(-(x-b)^2 * c^2)  for x>=b
                      a * exp(-(x-b)^2 * d^2)  for x<b
    MLE for c given right-side data: c = 1 / sqrt(2 * mean((x_i - b)^2))
    """
    # Profile scan: try many candidate b values, pick best LL
    candidates = np.quantile(pts, np.linspace(0.05, 0.95, 50))
    best_ll = -np.inf
    best = None
    for b_cand in candidates:
        b, c, d, ll = _fit_gaussian_at_b(pts, b_cand)
        if ll > best_ll:
            best_ll = ll
            best = (b, c, d)

    # Refine with iteration from best candidate
    mu = best[0]
    for _ in range(20):
        right = pts[pts >= mu] - mu
        left  = mu - pts[pts < mu]
        rms_r = max(np.sqrt((right ** 2).mean()) if len(right) > 0 else 1e-3, 1e-3)
        rms_l = max(np.sqrt((left  ** 2).mean()) if len(left)  > 0 else 1e-3, 1e-3)
        mu_new = pts.mean() + (1.0 / np.sqrt(np.pi)) * (rms_r - rms_l)
        if abs(mu_new - mu) < 1e-8:
            break
        mu = mu_new

    b, c, d, ll = _fit_gaussian_at_b(pts, mu)
    # Also check the scan winner in case iteration moved away
    _, _, _, ll_scan = _fit_gaussian_at_b(pts, best[0])
    if ll_scan > ll:
        b, c, d, ll = best[0], best[1], best[2], ll_scan

    bic = 3 * np.log(len(pts)) - 2 * ll   # 3 params: b, c, d
    return {"mu": b, "c_rate": c, "d_rate": d, "bic": bic}


def fit_asym_exp(pts):
    sorted_pts = np.sort(pts)
    b, c_rate, d_rate, _ = _profile_scan_unweighted(sorted_pts)
    loglike = np.sum(np.log(asym_exp_pdf(pts, b, c_rate, d_rate) + 1e-300))
    bic = 3 * np.log(len(pts)) - 2 * loglike
    return {"b": b, "c_rate": c_rate, "d_rate": d_rate, "bic": bic}


# -------------------------------------------------------
# 1-D KDE mode-finding for initialization
# -------------------------------------------------------
def init_clusters(X_1d, K=None, bw=None):
    if K is None:
        K = 1
    kde = gaussian_kde(X_1d, bw_method=bw)
    n_smp = len(X_1d)
    x_grid = np.linspace(X_1d.min(), X_1d.max(), n_smp)
    density = kde(x_grid)
    mode_idx = argrelextrema(density, np.greater,
                             order=max(int(0.05 * n_smp / K), 1))[0]

    if len(mode_idx) == 0:
        centers = np.quantile(X_1d, np.linspace(0, 1, K + 2)[1:-1])
    elif len(mode_idx) < K:
        modes = x_grid[mode_idx]
        needed = K - len(modes)
        cands = np.quantile(X_1d, np.linspace(0, 1, 10 * K + 2)[1:-1])
        dists = np.min(np.abs(cands[:, None] - modes[None, :]), axis=1)
        extra = cands[np.argsort(dists)[-needed:]]
        centers = np.sort(np.concatenate([modes, extra]))
    else:
        if len(mode_idx) > K:
            mode_idx = mode_idx[np.argsort(density[mode_idx])[::-1][:K]]
        centers = np.sort(x_grid[mode_idx])

    labels = np.argmin(np.abs(X_1d[:, None] - centers[None, :]), axis=1)
    return labels, centers


# -------------------------------------------------------
# Per-cluster type selection  (sum of per-dim BICs)
# -------------------------------------------------------
def _select_component_types(X, labels, K, min_pts=10):
    N, D = X.shape
    comp_types = []
    locs    = np.zeros((K, D))
    c_rates = np.zeros((K, D))
    d_rates = np.zeros((K, D))

    for k in range(K):
        pts = X[labels == k]
        n_k = len(pts)

        if n_k < min_pts:
            comp_types.append('G')
            locs[k]    = pts.mean(axis=0)
            sig = np.maximum(pts.std(axis=0), 1e-3)
            c_rates[k] = 1.0 / sig
            d_rates[k] = 1.0 / sig
            continue

        bic_g_total = 0.0
        bic_a_total = 0.0
        g_mu = np.zeros(D);  g_c = np.zeros(D);  g_d = np.zeros(D)
        a_b  = np.zeros(D);  a_c = np.zeros(D);  a_d = np.zeros(D)

        for d in range(D):
            g = fit_gaussian(pts[:, d])
            a = fit_asym_exp(pts[:, d])
            bic_g_total += g['bic']
            bic_a_total += a['bic']
            g_mu[d] = g['mu'];  g_c[d] = g['c_rate'];  g_d[d] = g['d_rate']
            a_b[d]  = a['b'];   a_c[d] = a['c_rate'];  a_d[d] = a['d_rate']

        print(f"  Cluster {k+1} (n={n_k}): "
              f"sumBIC(G)={bic_g_total:.1f}  sumBIC(A)={bic_a_total:.1f}")

        if bic_g_total <= bic_a_total:
            comp_types.append('G')
            locs[k] = g_mu;  c_rates[k] = g_c;  d_rates[k] = g_d
        else:
            comp_types.append('A')
            locs[k] = a_b;  c_rates[k] = a_c;  d_rates[k] = a_d

    return comp_types, locs, c_rates, d_rates


# -------------------------------------------------------
# Overall mixture BIC  (multi-D)
# -------------------------------------------------------
def compute_mixture_bic(X, comp_types, locs, c_rates, d_rates, weights):
    N, D = X.shape
    K = len(comp_types)

    log_comp = np.zeros((N, K))
    for k in range(K):
        lw = np.log(weights[k] + 1e-300)
        if comp_types[k] == 'G':
            log_comp[:, k] = lw + _log_pdf_gaussian_nd(X, locs[k],
                                                        c_rates[k], d_rates[k])
        else:
            log_comp[:, k] = lw + _log_pdf_asym_exp_nd(X, locs[k],
                                                        c_rates[k], d_rates[k])

    loglike = logsumexp(log_comp, axis=1).sum()
    # Both G and A have 3 params per dim (loc, c_rate, d_rate)
    n_params = (K - 1) + K * 3 * D
    bic  = n_params * np.log(N) - 2 * loglike
    return bic, loglike


# -------------------------------------------------------
# Multi-D EM loop
# -------------------------------------------------------
def _run_em_loop(X, comp_types, locs, c_rates, d_rates, weights,
                 sorted_indices, n_iter, tol):
    N, D = X.shape
    K = len(comp_types)
    comp_types = list(comp_types)
    locs    = locs.copy()
    c_rates = c_rates.copy(); d_rates = d_rates.copy()
    weights = weights.copy()

    loglike_prev = -np.inf
    for _ in range(n_iter):
        # ── E-step (log-space) ──
        log_R = np.zeros((N, K))
        for k in range(K):
            lw = np.log(weights[k] + 1e-300)
            if comp_types[k] == 'G':
                log_R[:, k] = lw + _log_pdf_gaussian_nd(X, locs[k],
                                                         c_rates[k], d_rates[k])
            else:
                log_R[:, k] = lw + _log_pdf_asym_exp_nd(X, locs[k],
                                                         c_rates[k], d_rates[k])
        log_total = logsumexp(log_R, axis=1)
        loglike   = log_total.sum()
        R = np.exp(log_R - log_total[:, None])

        if abs(loglike - loglike_prev) < tol:
            break
        loglike_prev = loglike

        # ── M-step ──
        for k in range(K):
            r_k = R[:, k]
            N_k = r_k.sum()
            if N_k < 1e-10:
                continue

            if comp_types[k] == 'G':
                # Asymmetric GaussianLike M-step (per dimension profile scan)
                for d in range(D):
                    x_d = X[:, d]
                    r_d = r_k
                    # Weighted profile scan over candidate b values
                    weighted_q = lambda q: x_d[np.argsort(x_d)[
                        max(0, min(int(q * N) - 1, N - 1))]]
                    candidates = np.array([weighted_q(q)
                                           for q in np.linspace(0.05, 0.95, 30)])
                    best_ll_d = -np.inf
                    best_b_d = locs[k, d]
                    best_c_d = c_rates[k, d]
                    best_d_d = d_rates[k, d]
                    for b_cand in candidates:
                        diff_d = x_d - b_cand
                        right_m = diff_d >= 0
                        left_m = ~right_m
                        wr = r_d * right_m
                        wl = r_d * left_m
                        Nr = wr.sum()
                        Nl = wl.sum()
                        if Nr < 1e-10 or Nl < 1e-10:
                            continue
                        rms_r = np.sqrt((wr * diff_d**2).sum() / Nr)
                        rms_l = np.sqrt((wl * diff_d**2).sum() / Nl)
                        rms_r = max(rms_r, 1e-3)
                        rms_l = max(rms_l, 1e-3)
                        c_cand = 1.0 / (rms_r * np.sqrt(2.0))
                        d_cand = 1.0 / (rms_l * np.sqrt(2.0))
                        log_a = (np.log(2.0 * c_cand * d_cand)
                                 - 0.5 * np.log(np.pi)
                                 - np.log(c_cand + d_cand))
                        ll_d = (r_d * (log_a + np.where(
                            diff_d >= 0,
                            -(diff_d ** 2) * c_cand ** 2,
                            -(diff_d ** 2) * d_cand ** 2))).sum()
                        if ll_d > best_ll_d:
                            best_ll_d = ll_d
                            best_b_d = b_cand
                            best_c_d = c_cand
                            best_d_d = d_cand
                    locs[k, d] = best_b_d
                    c_rates[k, d] = best_c_d
                    d_rates[k, d] = best_d_d
            else:
                # Per-dimension profile scan for asymmetric exponential
                for d in range(D):
                    sidx       = sorted_indices[d]
                    r_sorted   = r_k[sidx]
                    X_d_sorted = X[sidx, d]
                    b, c, dd   = _profile_scan_weighted(X_d_sorted, r_sorted, N_k)
                    locs[k, d]    = b
                    c_rates[k, d] = c
                    d_rates[k, d] = dd

        weights = R.sum(axis=0) / N

    bic, ll = compute_mixture_bic(X, comp_types, locs,
                                  c_rates, d_rates, weights)
    return dict(comp_types=comp_types, locs=locs,
                c_rates=c_rates, d_rates=d_rates, weights=weights,
                bic=bic, loglike=ll)


# -------------------------------------------------------
# Type swap search  (swap-and-rerun, multi-D)
# -------------------------------------------------------
def _try_type_swaps(X, result, sorted_indices, n_iter, tol):
    comp_types = list(result['comp_types'])
    locs    = result['locs'].copy()
    c_rates = result['c_rates'].copy(); d_rates = result['d_rates'].copy()
    weights = result['weights'].copy()
    current_bic = result['bic']
    K = len(comp_types)

    improved = True
    while improved:
        improved = False
        for k in range(K):
            tr_types = list(comp_types)
            tr_locs  = locs.copy()
            tr_c     = c_rates.copy(); tr_d = d_rates.copy()

            # Just flip the type; c_rates/d_rates carry over as initial guess
            tr_types[k] = 'A' if comp_types[k] == 'G' else 'G'

            trial = _run_em_loop(X, tr_types, tr_locs, tr_c, tr_d,
                                 weights.copy(), sorted_indices, n_iter, tol)

            if trial['bic'] < current_bic - 1.0:
                old, new = comp_types[k], tr_types[k]
                print(f"    Swap comp {k+1} {old}→{new}: "
                      f"BIC {current_bic:.1f} → {trial['bic']:.1f}")
                comp_types = trial['comp_types']
                locs    = trial['locs']
                c_rates = trial['c_rates']; d_rates = trial['d_rates']
                weights = trial['weights']
                current_bic = trial['bic']
                improved = True
                break

    bic, ll = compute_mixture_bic(X, comp_types, locs,
                                  c_rates, d_rates, weights)
    return dict(comp_types=comp_types, locs=locs,
                c_rates=c_rates, d_rates=d_rates, weights=weights,
                bic=bic, loglike=ll)


# -------------------------------------------------------
# Fixed-K EM  (multi-D)
# -------------------------------------------------------
def _em_fixed_K(X, K, n_iter, bw, tol):
    N, D = X.shape

    # KDE init on dimension with largest variance
    init_dim = int(np.argmax(X.std(axis=0)))
    labels, _ = init_clusters(X[:, init_dim], K, bw)
    K = len(set(labels))

    comp_types, locs, c_rates, d_rates = \
        _select_component_types(X, labels, K)
    weights = np.ones(K) / K

    sorted_indices = [np.argsort(X[:, d]) for d in range(D)]

    result = _run_em_loop(X, comp_types, locs, c_rates, d_rates,
                          weights, sorted_indices, n_iter, tol)
    K_g = sum(1 for t in result['comp_types'] if t == 'G')
    print(f"  EM done: {K_g}G+{K-K_g}A  BIC={result['bic']:.1f}")

    print("  Checking type swaps...")
    result = _try_type_swaps(X, result, sorted_indices, n_iter, tol)
    return result


# -------------------------------------------------------
# Format internal result → user-facing dict
# -------------------------------------------------------
def _format_output(res, D):
    K = len(res['comp_types'])
    types = ['Gaussian' if t == 'G' else 'Exp' for t in res['comp_types']]
    return dict(K=K, types=types, weights=res['weights'],
                means=res['locs'],
                c_rates=res['c_rates'], d_rates=res['d_rates'],
                bic=res['bic'], loglike=res['loglike'])


def _print_result(res, D):
    K  = len(res['comp_types'])
    K_g = sum(1 for t in res['comp_types'] if t == 'G')
    print(f"\n─── Result: {K_g} Gaussian + {K-K_g} Exp  BIC={res['bic']:.1f} ───")
    for k in range(K):
        t = res['comp_types'][k]
        w = res['weights'][k]
        c_r = res['c_rates'][k]
        d_r = res['d_rates'][k]
        label = 'Gaussian' if t == 'G' else 'Exp'
        if D == 1:
            print(f"  {label} {k+1}: w={w:.3f}  b={res['locs'][k,0]:.3f}"
                  f"  c={c_r[0]:.3f}(1/c={1/c_r[0]:.3f})"
                  f"  d={d_r[0]:.3f}(1/d={1/d_r[0]:.3f})")
        else:
            print(f"  {label} {k+1}: w={w:.3f}  b={np.round(res['locs'][k],3)}"
                  f"  c={np.round(c_r,3)}  d={np.round(d_r,3)}")


# -------------------------------------------------------
# Main entry
# -------------------------------------------------------
def em_mixture(X, K=None, max_K=None, n_iter=500, bw=0.25, tol=1e-8):
    """Fit a mixture of Gaussians and asymmetric exponentials.

    Parameters
    ----------
    X : array (N,) or (N, D)
        Data matrix.  1-D arrays are reshaped to (N, 1).
    K : int, optional
        Fixed number of components.
    max_K : int, optional
        If K is None, search K=1..max_K and pick lowest BIC.
    n_iter, bw, tol : EM hyper-parameters.

    Returns
    -------
    dict with keys:
        K        – number of components
        types    – list of K strings ("Gaussian" or "Exp")
        weights  – (K,) mixture weights
        means    – (K, D) component centres (b)
        c_rates  – (K, D) right-side rates (c)
        d_rates  – (K, D) left-side rates (d)
        bic, loglike
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N, D = X.shape

    if K is not None:
        print(f"Fitting K={K}, N={N}, D={D}")
        res = _em_fixed_K(X, K, n_iter, bw, tol)
        _print_result(res, D)
        return _format_output(res, D)

    if max_K is None:
        max_K = 8
    best_res = None
    best_bic = np.inf
    all_bics = []

    for k_try in range(1, max_K + 1):
        print(f"\n{'='*50}\nTrying K={k_try}\n{'='*50}")
        res = _em_fixed_K(X, k_try, n_iter, bw, tol)
        print(f"  K={k_try}: BIC={res['bic']:.1f}")
        all_bics.append((k_try, res['bic']))
        if res['bic'] < best_bic:
            best_bic = res['bic']
            best_res = res

    print(f"\nBest K={len(best_res['comp_types'])}  BIC={best_bic:.1f}")
    print(f"BIC per K: {all_bics}")
    _print_result(best_res, D)
    return _format_output(best_res, D)


# -------------------------------------------------------
# Evaluate & Plot  (1-D only, D=1)
# -------------------------------------------------------
def eval_mixture(x_grid, params):
    K = params['K']
    w = params['weights']
    total = np.zeros_like(x_grid)
    comps = []
    for k in range(K):
        b      = params['means'][k, 0]
        c_rate = params['c_rates'][k, 0]
        d_rate = params['d_rates'][k, 0]
        if params['types'][k] == 'Gaussian':
            comp = w[k] * GaussianLike(b, c_rate, d_rate).pdf(x_grid)
            comps.append(('Gaussian', k, comp))
        else:
            comp = w[k] * DoubleExponential(b, c_rate, d_rate).pdf(x_grid)
            comps.append(('Exp', k, comp))
        total += comp
    return total, comps


def plot_mixture(x_grid, y_orig, params):
    assert params['means'].shape[1] == 1, "plot_mixture only supports D=1"
    K     = params['K']
    w     = params['weights']
    K_g   = sum(1 for t in params['types'] if t == 'Gaussian')
    K_a   = K - K_g
    y_norm       = y_orig / (y_orig.sum() * (x_grid[1] - x_grid[0]))
    total, comps = eval_mixture(x_grid, params)
    colors_g = plt.cm.Blues(np.linspace(0.5, 0.9,  max(K_g, 1)))
    colors_e = plt.cm.Greens(np.linspace(0.5, 0.9, max(K_a, 1)))
    fig, ax  = plt.subplots(1, figsize=(10, 6))
    ax.plot(x_grid, y_norm, 'k-',  lw=2, label='Original (normalized)')
    ax.plot(x_grid, total,  'r--', lw=2, label='Fitted mixture')
    gi, ai = 0, 0
    for (kind, k, comp) in comps:
        b      = params['means'][k, 0]
        c_rate = params['c_rates'][k, 0]
        d_rate = params['d_rates'][k, 0]
        if kind == 'Gaussian':
            lbl = (f"Gaussian {gi+1}  w={w[k]:.2f}  b={b:.2f}"
                   f"  c={c_rate:.2f}  d={d_rate:.2f}")
            ax.plot(x_grid, comp, lw=1.5, ls=':', color=colors_g[gi], label=lbl)
            gi += 1
        else:
            lbl = (f"Exp {ai+1}  w={w[k]:.2f}  b={b:.2f}"
                   f"  c={c_rate:.2f}  d={d_rate:.2f}")
            ax.plot(x_grid, comp, lw=1.5, ls=':', color=colors_e[ai], label=lbl)
            ai += 1
    ax.set_title(f'Fitted Mixture: {K_g} Gaussian + {K_a} Exp  [BIC-selected types]')
    ax.set_xlabel('x'); ax.set_ylabel('Density')
    ax.legend(fontsize=8); ax.grid(True)
    plt.tight_layout(); plt.show()


def plot_mixture_nd(X, params, dim_names=None):
    """Plot per-dimension marginal density: data histogram + fitted marginal."""
    K = params['K']
    D = params['means'].shape[1]
    w = params['weights']
    K_g = sum(1 for t in params['types'] if t == 'Gaussian')
    K_a = K - K_g

    if dim_names is None:
        dim_names = [f"dim {d}" for d in range(D)]

    colors_g = plt.cm.Blues(np.linspace(0.5, 0.9, max(K_g, 1)))
    colors_e = plt.cm.Greens(np.linspace(0.5, 0.9, max(K_a, 1)))

    fig, axes = plt.subplots(1, D, figsize=(6 * D, 5), squeeze=False)
    axes = axes[0]

    for d in range(D):
        ax = axes[d]
        x_d = X[:, d]
        ax.hist(x_d, bins=120, density=True, alpha=0.35, color='gray',
                label='Data')

        x_grid = np.linspace(x_d.min() - 1, x_d.max() + 1, 500)
        total = np.zeros_like(x_grid)
        gi, ai = 0, 0

        for k in range(K):
            b_d = params['means'][k, d]
            c_d = params['c_rates'][k, d]
            d_d = params['d_rates'][k, d]
            if params['types'][k] == 'Gaussian':
                comp = w[k] * GaussianLike(b_d, c_d, d_d).pdf(x_grid)
                lbl = f"G{gi+1} w={w[k]:.2f} b={b_d:.1f} c={c_d:.2f} d={d_d:.2f}"
                ax.plot(x_grid, comp, lw=1.5, ls=':', color=colors_g[gi],
                        label=lbl)
                gi += 1
            else:
                comp = w[k] * DoubleExponential(b_d, c_d, d_d).pdf(x_grid)
                lbl = f"E{ai+1} w={w[k]:.2f} b={b_d:.1f} c={c_d:.2f} d={d_d:.2f}"
                ax.plot(x_grid, comp, lw=1.5, ls=':', color=colors_e[ai],
                        label=lbl)
                ai += 1
            total += comp

        ax.plot(x_grid, total, 'r--', lw=2, label='Mixture')
        ax.set_title(f'{dim_names[d]}')
        ax.set_xlabel('x'); ax.set_ylabel('Density')
        ax.legend(fontsize=7); ax.grid(True)

    fig.suptitle(f'Marginal Densities: {K_g} Gaussian + {K_a} Exp  (K={K})',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('mixture_nd.png', dpi=150, bbox_inches='tight')
    plt.show()


# ===============================================================
# Test
# ===============================================================
if __name__ == '__main__':
    # x = np.linspace(-20, 20, 1000)
    # a = [.1, .5, 1, .2, .75]
    # sigma_a = [1, 1, 1, .05, .5]
    # sigma_b = [1, .1, 1, .5, .5]
    # c = [-8, -5, 5, 0, 10]
    # p = [2, 1, 1, 1, 2]
    # q = [2, 1, 1, 1, 2]

    # y = np.zeros_like(x)
    # for i in range(5):
    #     y += (a[i] * np.exp(-((x - c[i]) / sigma_a[i])**p[i])) * np.heaviside(x - c[i], 1) \
    #        + (a[i] * np.exp(-((c[i] - x) / sigma_b[i])**q[i])) * np.heaviside(c[i] - x, 1)

    # y_prob    = y / y.sum()
    # x_samples = np.random.choice(x, size=10000, p=y_prob)

    # plt.figure(); plt.plot(x, y); plt.show()

    # ── 1-D test ──
    src_g = [GaussianLike(b=1, c=1.0, d=2.0), GaussianLike(b=6, c=1.0, d=0.5)]
    x = np.linspace(-20, 20, 1000)
    pdf_x = np.concatenate([src_g[0].pdf(x), src_g[1].pdf(x)])
    y = pdf_x / (pdf_x.sum())

    x = np.concatenate([np.linspace(-20, 20, 1000), np.linspace(-20, 20, 1000)])
    x_samples = np.random.choice(x, size=10000, p=y)

    params = em_mixture(x_samples, K=2, n_iter=100, bw=0.1)
    print("\n─── Final 1-D Parameters ───")
    for k in range(params['K']):
        t = params['types'][k]
        w = params['weights'][k]
        cr = params['c_rates'][k, 0]; dr = params['d_rates'][k, 0]
        print(f"  {t} {k+1}: w={w:.3f}  b={params['means'][k,0]:.3f}"
              f"  c={cr:.3f}  d={dr:.3f}")

    plot_mixture(x, y, params)
    # ── 2-D test (same 1-D signal duplicated + noise in dim 2) ──
    print("\n\n" + "="*60)
    print("2-D test")
    print("="*60)
    X_2d = np.column_stack([x_samples,
                            x_samples + np.random.randn(len(x_samples)) * 0.5])
    params_2d = em_mixture(X_2d, K=5, n_iter=100, bw=0.1)
    print("\n─── Final 2-D Parameters ───")
    for k in range(params_2d['K']):
        t = params_2d['types'][k]
        w = params_2d['weights'][k]
        print(f"  {t} {k+1}: w={w:.3f}  b={np.round(params_2d['means'][k],3)}"
              f"  c={np.round(params_2d['c_rates'][k],3)}"
              f"  d={np.round(params_2d['d_rates'][k],3)}")

    # ── 3-D test ──
    # Dim 0: original 5-component 1-D signal
    # Dim 1: shifted centres + different asymmetries
    # Dim 2: independent Gaussian clusters at distinct locations
    print("\n\n" + "="*60)
    print("3-D test")
    print("="*60)
    np.random.seed(42)
    N3 = 12000

    # Ground truth: 5 components, each generating 3-D samples independently
    #   comp 0: Gaussian       — centres [−8, 0, 5],   σ=[1, 0.8, 1.2]
    #   comp 1: AsymExp        — centres [−5, 3, −2],  1/c=[1, 0.5, 0.8], 1/d=[0.1, 0.3, 1.5]
    #   comp 2: AsymExp (sym)  — centres [ 5, −4, 8],  1/c=[1, 1, 1],     1/d=[1, 1, 1]
    #   comp 3: AsymExp (asym) — centres [ 0, 6, −6],  1/c=[0.05, 0.1, 0.2], 1/d=[0.5, 0.8, 1.0]
    #   comp 4: Gaussian       — centres [10, −7, 0],  σ=[0.5, 0.6, 0.4]
    true_w = np.array([0.10, 0.25, 0.30, 0.10, 0.25])
    true_types = ['G', 'A', 'A', 'A', 'G']
    true_means = np.array([[-8,  0,  5],
                           [-5,  3, -2],
                           [ 5, -4,  8],
                           [ 0,  6, -6],
                           [10, -7,  0]], dtype=float)
    true_sigma_g = np.array([[1.0, 0.8, 1.2],   # comp 0
                             [0,   0,   0  ],
                             [0,   0,   0  ],
                             [0,   0,   0  ],
                             [0.5, 0.6, 0.4]])   # comp 4
    true_c_inv = np.array([[0,    0,   0  ],
                           [1.0, 0.5, 0.8],      # comp 1
                           [1.0, 1.0, 1.0],      # comp 2
                           [0.05,0.1, 0.2],       # comp 3
                           [0,   0,   0  ]])
    true_d_inv = np.array([[0,    0,   0  ],
                           [0.1, 0.3, 1.5],      # comp 1
                           [1.0, 1.0, 1.0],      # comp 2
                           [0.5, 0.8, 1.0],      # comp 3
                           [0,   0,   0  ]])

    # Sample from the true mixture
    comp_ids = np.random.choice(5, size=N3, p=true_w)
    X_3d = np.zeros((N3, 3))
    for i in range(5):
        mask = comp_ids == i
        ni = mask.sum()
        if true_types[i] == 'G':
            for d in range(3):
                X_3d[mask, d] = np.random.normal(true_means[i, d],
                                                  true_sigma_g[i, d], ni)
        else:
            for d in range(3):
                c_rate = 1.0 / true_c_inv[i, d]
                d_rate = 1.0 / true_d_inv[i, d]
                u = np.random.rand(ni)
                # inverse CDF of asymmetric exponential centred at b
                b = true_means[i, d]
                p_right = d_rate / (c_rate + d_rate)
                right = u >= (1 - p_right)
                samp = np.where(right,
                                b - np.log(1 - (u - (1 - p_right)) / p_right) / c_rate,
                                b + np.log(u / (1 - p_right)) / d_rate)
                X_3d[mask, d] = samp

    print(f"  Generated {N3} samples in 3-D")
    print(f"  True: {true_types}, weights={true_w}")

    params_3d = em_mixture(X_3d, K=5, n_iter=200, bw=0.15)

    print("\n─── Final 3-D Parameters ───")
    for k in range(params_3d['K']):
        t = params_3d['types'][k]
        w = params_3d['weights'][k]
        m = np.round(params_3d['means'][k], 3)
        cr = np.round(params_3d['c_rates'][k], 3)
        dr = np.round(params_3d['d_rates'][k], 3)
        print(f"  {t} {k+1}: w={w:.3f}  b={m}  c={cr}  d={dr}")

    plot_mixture_nd(X_3d, params_3d, dim_names=["Dim 0", "Dim 1", "Dim 2"])
