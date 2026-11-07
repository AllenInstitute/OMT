import numpy as np
from scipy.stats import norm, gaussian_kde
from scipy.signal import argrelextrema
from scipy.special import logsumexp
from scipy.special import erf, erfinv
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.special import spence
from functions import vectorized_geometric_mean

class DoubleExponential:
    """Eq. 11 with p=q=1. Supports scalar (1-D) and vector (disjoint multi-D)."""
    def __init__(self, b, c, d):
        self.b = np.asarray(b, dtype=float).squeeze()
        self.c = np.asarray(c, dtype=float).squeeze()
        self.d = np.asarray(d, dtype=float).squeeze()
        self.a = (self.c * self.d) / (self.c + self.d)
        self.ndim = self.b.ndim  # 0=scalar, 1=disjoint multi-D

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
        self.ndim = self.b.ndim  # 0=scalar, 1=disjoint multi-D

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

def plot_omt_coupling(source_mix, target_mix, weights, title="OMT Coupling", eps=.1, n_grid=400, probe=np.linspace(-10, 10, 1000)):
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


def plot_omt_coupling_nd(source_params, target_params, weights,
                         title="OMT Coupling", eps=.1, n_grid=400,
                         probe=np.linspace(-20, 20, 1000)):
    """
    Per-dimension coupling plots for disjoint multi-D mixtures.

    Parameters
    ----------
    source_params / target_params : dict
        Output of em_mixture (keys: types, weights, means, c_rates, d_rates).
    weights : (K_s, K_t) array — transport weight matrix.
    """
    means_s = np.atleast_2d(source_params['means'])
    means_t = np.atleast_2d(target_params['means'])
    D = means_s.shape[1]
    K_s = means_s.shape[0]
    K_t = means_t.shape[0]
    types_s = source_params['types']
    types_t = target_params['types']
    c_s = np.atleast_2d(source_params['c_rates'])
    d_s = np.atleast_2d(source_params['d_rates'])
    c_t = np.atleast_2d(target_params['c_rates'])
    d_t = np.atleast_2d(target_params['d_rates'])

    for dim in range(D):
        src_comps = []
        for k in range(K_s):
            cls = DoubleExponential if types_s[k] == 'Exp' else GaussianLike
            src_comps.append(cls(b=means_s[k, dim], c=c_s[k, dim], d=d_s[k, dim]))
        tgt_comps = []
        for k in range(K_t):
            cls = DoubleExponential if types_t[k] == 'Exp' else GaussianLike
            tgt_comps.append(cls(b=means_t[k, dim], c=c_t[k, dim], d=d_t[k, dim]))

        plot_omt_coupling(src_comps, tgt_comps, weights,
                          title=f"{title} — dim {dim+1}/{D}",
                          eps=eps, n_grid=n_grid, probe=probe)


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


def _weighted_ll_at_b(b, X_sorted, r_k_sorted):
    """Compute weighted log-likelihood of asym exp at a given b."""
    diff = X_sorted - b
    right = diff >= 0
    left = ~right
    wr = r_k_sorted * right
    wl = r_k_sorted * left
    W_R = wr.sum()
    W_L = wl.sum()
    if W_R < 1e-10 or W_L < 1e-10:
        return -np.inf, 1.0, 1.0
    S_R = (wr * diff).sum()
    S_L = -(wl * diff).sum()
    if S_R < 1e-10 or S_L < 1e-10:
        return -np.inf, 1.0, 1.0
    c_rate = max(W_R / S_R, 1e-3)
    d_rate = max(W_L / S_L, 1e-3)
    N_k = r_k_sorted.sum()
    log_a = np.log(c_rate * d_rate / (c_rate + d_rate) + 1e-300)
    ll = N_k * log_a + (r_k_sorted * np.where(diff >= 0, -c_rate * diff, d_rate * diff)).sum()
    return ll, c_rate, d_rate


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
    b_scan = X_sorted[best + 1]
    c_scan = max(W_R[best] / WS_R[best], 1e-3)
    d_scan = max(W_L[best] / WS_L[best], 1e-3)

    # Also try the weighted median — the MLE for b in a Laplace-like model
    cum_w_norm = cum_rk[1:] / (N_k + 1e-300)
    med_idx = np.searchsorted(cum_w_norm, 0.5)
    med_idx = np.clip(med_idx, 0, n - 1)
    b_median = X_sorted[med_idx]

    # Also try weighted quantiles near the boundary (mode region)
    candidates = [b_scan, b_median]
    for q in (0.1, 0.25, 0.75, 0.9):
        qi = np.searchsorted(cum_w_norm, q)
        qi = np.clip(qi, 0, n - 1)
        candidates.append(X_sorted[qi])

    # Evaluate actual weighted LL for each candidate and pick the best
    best_ll = -np.inf
    best_b, best_c, best_d = b_scan, c_scan, d_scan
    for b_cand in candidates:
        ll, c_cand, d_cand = _weighted_ll_at_b(b_cand, X_sorted, r_k_sorted)
        if ll > best_ll:
            best_ll = ll
            best_b, best_c, best_d = b_cand, c_cand, d_cand

    return best_b, best_c, best_d


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

    # Cap rates: max rate = 1 / (min_spread per dim), prevents degenerate narrow tails
    global_std = np.std(X, axis=0)
    max_rate = 1.0 / (np.maximum(global_std, 1e-3) * 0.05)  # ~20/std

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
                for d in range(D):
                    x_d = X[:, d]
                    r_d = r_k
                    sidx = sorted_indices[d]
                    qpos = np.linspace(0.05, 0.95, 10)
                    qidx = np.clip((qpos * N).astype(int) - 1, 0, N - 1)
                    candidates = x_d[sidx[qidx]]
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
                        Nr = wr.sum(); Nl = wl.sum()
                        if Nr < 1e-10 or Nl < 1e-10:
                            continue
                        rms_r = max(np.sqrt((wr * diff_d**2).sum() / Nr), 1e-3)
                        rms_l = max(np.sqrt((wl * diff_d**2).sum() / Nl), 1e-3)
                        c_cand = 1.0 / (rms_r * np.sqrt(2.0))
                        d_cand = 1.0 / (rms_l * np.sqrt(2.0))
                        log_a = (np.log(2.0 * c_cand * d_cand)
                                 - 0.5 * np.log(np.pi)
                                 - np.log(c_cand + d_cand))
                        ll_d = (r_d * (log_a + np.where(
                            diff_d >= 0,
                            -(diff_d**2) * c_cand**2,
                            -(diff_d**2) * d_cand**2))).sum()
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
                # Filter to points with significant responsibility to avoid
                # noise from irrelevant points pulling b away from the mode
                mask = r_k > (1e-2 / K)
                if mask.sum() < 5:
                    mask = r_k > 0
                for d in range(D):
                    x_d_masked = X[mask, d]
                    r_masked = r_k[mask]
                    order = np.argsort(x_d_masked)
                    X_d_sorted = x_d_masked[order]
                    r_sorted = r_masked[order]
                    N_k_eff = r_masked.sum()
                    b, c, dd   = _profile_scan_weighted(X_d_sorted, r_sorted, N_k_eff)
                    locs[k, d]    = b
                    c_rates[k, d] = min(c, max_rate[d])
                    d_rates[k, d] = min(dd, max_rate[d])

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

            swap_iters = min(n_iter, max(10, 50 // max(K // 10, 1)))
            trial = _run_em_loop(X, tr_types, tr_locs, tr_c, tr_d,
                                 weights.copy(), sorted_indices, swap_iters, tol)

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

    if D > 1:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
        labels = km.labels_
    else:
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


def _exp_fixed_K(X, K, n_iter, bw, tol):
    N, D = X.shape

    # Use K-Means for multi-dimensional initialization instead of single-dim KDE
    if D > 1:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
        labels = km.labels_
    else:
        init_dim = int(np.argmax(X.std(axis=0)))
        labels, _ = init_clusters(X[:, init_dim], K, bw)
    K = len(set(labels))

    comp_types = ['A'] * K
    locs = np.zeros((K, D))
    c_rates = np.zeros((K, D))
    d_rates = np.zeros((K, D))

    for k in range(K):
        pts = X[labels == k]
        if len(pts) < 3:
            locs[k] = pts.mean(axis=0)
            c_rates[k] = 1.0
            d_rates[k] = 1.0
            continue
        for d in range(D):
            res = fit_asym_exp(pts[:, d])
            locs[k, d] = res['b']
            c_rates[k, d] = res['c_rate']
            d_rates[k, d] = res['d_rate']

    weights = np.ones(K) / K
    sorted_indices = [np.argsort(X[:, d]) for d in range(D)]

    result = _run_em_loop(X, comp_types, locs, c_rates, d_rates,
                          weights, sorted_indices, n_iter, tol)
    print(f"  EM done: {K}A  BIC={result['bic']:.1f}")
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


def exp_fitting(X, K=None, max_K=None, n_iter=500, bw=0.25, tol=1e-8):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N, D = X.shape

    if K is not None:
        print(f"Fitting Exp-only K={K}, N={N}, D={D}")
        res = _exp_fixed_K(X, K, n_iter, bw, tol)
        _print_result(res, D)
        out = _format_output(res, D)
        out['n_iter'] = n_iter
        return out

    if max_K is None:
        max_K = 8
    best_res = None
    best_bic = np.inf

    for k_try in range(1, max_K + 1):
        print(f"\n{'='*50}\nTrying K={k_try} (Exp-only)\n{'='*50}")
        res = _exp_fixed_K(X, k_try, n_iter, bw, tol)
        if res['bic'] < best_bic:
            best_bic = res['bic']
            best_res = res

    _print_result(best_res, D)
    out = _format_output(best_res, D)
    out['n_iter'] = n_iter
    return out


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
    


def W2TransportCalculator(b_s, c_s, d_s, type_s, b_t, c_t, d_t, type_t):
    """
    Compute W2^2 between two components. Accepts scalars or arrays.
    For disjoint multi-D (arrays), sums per-dimension W2^2 costs.
    c_s, d_s, c_t, d_t are diagonals of the scaling matrices C, D.
    """
    b_s = np.asarray(b_s, dtype=float).ravel()
    c_s = np.asarray(c_s, dtype=float).ravel()
    d_s = np.asarray(d_s, dtype=float).ravel()
    b_t = np.asarray(b_t, dtype=float).ravel()
    c_t = np.asarray(c_t, dtype=float).ravel()
    d_t = np.asarray(d_t, dtype=float).ravel()

    if type_s == 'Exp' and type_t == 'Exp':
        return np.sum(W2_exp(b_s, c_s, d_s, b_t, c_t, d_t))
    elif type_s == 'Gaussian' and type_t == 'Gaussian':
        return np.sum(W2_gaussian(b_s, c_s, d_s, b_t, c_t, d_t))
    elif type_s == 'Exp' and type_t == 'Gaussian':
        return np.sum(W2_exp_gauss(b_s, c_s, d_s, b_t, c_t, d_t))
    elif type_s == 'Gaussian' and type_t == 'Exp':
        return np.sum(W2_exp_gauss(b_t, c_t, d_t, b_s, c_s, d_s))
    else:
        raise ValueError("Unknown distribution types provided.")

def W2_exp(b_i, c_i, d_i, b_j, c_j, d_j):
    """Per-dimension W2^2 for Exp-Exp. Accepts scalars or arrays."""
    diff_b = b_i - b_j
    cross_const = 2.0 - (np.pi**2 / 6.0)

    w2_cc = diff_b**2 + 2*diff_b*(1/c_i - 1/c_j) + 2*(1/c_i - 1/c_j)**2
    w2_cd = diff_b**2 + 2*diff_b*(1/c_i + 1/d_j) + 2/c_i**2 + 2/d_j**2 + 2/(c_i*d_j)*cross_const
    w2_dc = diff_b**2 - 2*diff_b*(1/d_i + 1/c_j) + 2/d_i**2 + 2/c_j**2 + 2/(d_i*c_j)*cross_const
    w2_dd = diff_b**2 - 2*diff_b*(1/d_i - 1/d_j) + 2*(1/d_i - 1/d_j)**2

    return w2_cc + w2_cd + w2_dc + w2_dd

def W2_gaussian(b_i, c_i, d_i, b_j, c_j, d_j):
    """Per-dimension W2^2 for Gaussian-Gaussian. Accepts scalars or arrays."""
    diff_b = b_i - b_j
    k1, k2, k3 = 1.0/np.sqrt(np.pi), 0.5, 0.0812625

    w2_cc = diff_b**2 + 2*k1*diff_b*(1/c_i - 1/c_j) + k2*(1/c_i - 1/c_j)**2
    w2_cd = diff_b**2 + 2*k1*diff_b*(1/c_i + 1/d_j) + k2/c_i**2 + k2/d_j**2 + 2*(k3/(c_i*d_j))
    w2_dc = diff_b**2 - 2*k1*diff_b*(1/d_i + 1/c_j) + k2/d_i**2 + k2/c_j**2 + 2*(k3/(d_i*c_j))
    w2_dd = diff_b**2 - 2*k1*diff_b*(1/d_i - 1/d_j) + k2*(1/d_i - 1/d_j)**2

    return w2_cc + w2_cd + w2_dc + w2_dd

def W2_exp_gauss(b_e, c_e, d_e, b_g, c_g, d_g):
    """Per-dimension W2^2 for Exp-Gaussian. Accepts scalars or arrays."""
    diff_b = b_e - b_g
    kg_m, kg_v, ke_v = 1.0/np.sqrt(np.pi), 0.5, 2.0
    k_eg1, k_eg2 = -0.981344, -0.242783

    w2_rc_rc = diff_b**2 + 2*diff_b*(1/c_e - kg_m/c_g) + ke_v/c_e**2 + kg_v/c_g**2 + 2/(c_e*c_g)*k_eg1
    w2_rc_ld = diff_b**2 + 2*diff_b*(1/c_e + kg_m/d_g) + ke_v/c_e**2 + kg_v/d_g**2 - 2/(c_e*d_g)*k_eg1
    w2_ld_rc = diff_b**2 + 2*diff_b*(-1/d_e - kg_m/c_g) + ke_v/d_e**2 + kg_v/c_g**2 - 2/(d_e*c_g)*k_eg2
    w2_ld_ld = diff_b**2 + 2*diff_b*(-1/d_e + kg_m/d_g) + ke_v/d_e**2 + kg_v/d_g**2 + 2/(d_e*d_g)*k_eg2

    return w2_rc_rc + w2_rc_ld + w2_ld_rc + w2_ld_ld


def vectorized_W2_cost_matrix(mean_s, c_rates_s, d_rates_s, types_s,
                              mean_t, c_rates_t, d_rates_t, types_t):
    """
    Compute the full (K_s, K_t) W2^2 cost matrix without Python loops.
    For disjoint multi-D, sums per-dimension costs via broadcasting.

    Parameters
    ----------
    mean_s    : (K_s, D) array
    c_rates_s : (K_s, D) array   — diagonal of C matrices
    d_rates_s : (K_s, D) array   — diagonal of D matrices
    types_s   : list of str, length K_s  ('Exp' or 'Gaussian')
    mean_t    : (K_t, D) array
    c_rates_t : (K_t, D) array
    d_rates_t : (K_t, D) array
    types_t   : list of str, length K_t

    Returns
    -------
    C : (K_s, K_t) array of W2^2 costs
    """
    mean_s = np.atleast_2d(mean_s)
    mean_t = np.atleast_2d(mean_t)
    c_rates_s = np.atleast_2d(c_rates_s)
    c_rates_t = np.atleast_2d(c_rates_t)
    d_rates_s = np.atleast_2d(d_rates_s)
    d_rates_t = np.atleast_2d(d_rates_t)

    K_s, D = mean_s.shape
    K_t = mean_t.shape[0]

    # Broadcast to (K_s, K_t, D)
    b_i = mean_s[:, np.newaxis, :]      # (K_s, 1, D)
    b_j = mean_t[np.newaxis, :, :]      # (1, K_t, D)
    c_i = c_rates_s[:, np.newaxis, :]
    c_j = c_rates_t[np.newaxis, :, :]
    d_i = d_rates_s[:, np.newaxis, :]
    d_j = d_rates_t[np.newaxis, :, :]

    is_exp_s = np.array([t == 'Exp' for t in types_s])          # (K_s,)
    is_exp_t = np.array([t == 'Exp' for t in types_t])          # (K_t,)
    is_gau_s = ~is_exp_s
    is_gau_t = ~is_exp_t

    # Pre-compute all three kernel results — (K_s, K_t, D) each
    ee = W2_exp(b_i, c_i, d_i, b_j, c_j, d_j)
    gg = W2_gaussian(b_i, c_i, d_i, b_j, c_j, d_j)
    eg = W2_exp_gauss(b_i, c_i, d_i, b_j, c_j, d_j)
    ge = W2_exp_gauss(b_j, c_j, d_j, b_i, c_i, d_i)

    # Build (K_s, K_t) type masks
    m_ee = (is_exp_s[:, None] & is_exp_t[None, :])
    m_gg = (is_gau_s[:, None] & is_gau_t[None, :])
    m_eg = (is_exp_s[:, None] & is_gau_t[None, :])
    m_ge = (is_gau_s[:, None] & is_exp_t[None, :])

    # Sum across dimensions, then select via masks
    C = np.zeros((K_s, K_t))
    C += m_ee * ee.sum(axis=-1)
    C += m_gg * gg.sum(axis=-1)
    C += m_eg * eg.sum(axis=-1)
    C += m_ge * ge.sum(axis=-1)

    return C


# -------------------------------------------------------
# Closed-form 1-D transport maps (Eqs. 118, 120, 123, 124)
# -------------------------------------------------------

def _T_exp_exp(x, bi, ci, di, bj, cj, dj):
    """Eq. 118: transport map from Exp_i to Exp_j, per-dimension scalar arrays.
    Uses CDF-based case selection to guarantee continuity."""
    ai = (ci * di) / (ci + di)
    aj = (cj * dj) / (cj + dj)

    # Compute source CDF F_i(x) for all points
    F = np.empty_like(x)
    right = x >= bi
    left = ~right
    F[right] = 1.0 - (ai / ci) * np.exp(-ci * (x[right] - bi))
    F[left] = (ai / di) * np.exp(di * (x[left] - bi))

    # Target CDF at bj: CDF_j(bj) = aj/dj
    F_bj = aj / dj

    # Case split based on CDF value
    target_right = F >= F_bj
    target_left = ~target_right

    y = np.empty_like(x)

    # Target right branch: F_j^{-1}(F) = bj - (1/cj)*log(cj*(1-F)/aj)
    if target_right.any():
        arg = cj * (1.0 - F[target_right]) / aj
        arg = np.clip(arg, 1e-300, None)
        y[target_right] = bj - (1.0 / cj) * np.log(arg)

    # Target left branch: F_j^{-1}(F) = bj + (1/dj)*log(dj*F/aj)
    if target_left.any():
        arg = dj * F[target_left] / aj
        arg = np.clip(arg, 1e-300, None)
        y[target_left] = bj + (1.0 / dj) * np.log(arg)

    return y


def _T_gauss_gauss(x, bi, ci, di, bj, cj, dj):
    """Eq. 120: transport map from GaussianLike_i to GaussianLike_j, per-dimension.
    Uses CDF-based case selection to guarantee continuity."""
    ai = (2.0 * ci * di) / (np.sqrt(np.pi) * (ci + di))
    aj = (2.0 * cj * dj) / (np.sqrt(np.pi) * (cj + dj))

    # Compute source CDF F_i(x) for all points
    F = np.empty_like(x)
    right = x >= bi
    left = ~right
    F[right] = np.sqrt(np.pi) * ai / 2.0 * (1.0 / di + (1.0 / ci) * erf(ci * (x[right] - bi)))
    F[left] = np.sqrt(np.pi) * ai / 2.0 * (1.0 / di + (1.0 / di) * erf(di * (x[left] - bi)))

    # Target CDF at bj: CDF_j(bj) = sqrt(pi)*aj/(2*dj)
    F_bj = np.sqrt(np.pi) * aj / (2.0 * dj)

    # Case split based on CDF value
    target_right = F >= F_bj
    target_left = ~target_right

    y = np.empty_like(x)

    # Target right branch: F_j^{-1}(F) = bj + (1/cj)*erfinv((2*cj/(sqrt(pi)*aj))*F - cj/dj)
    if target_right.any():
        arg = (2.0 * cj) / (np.sqrt(np.pi) * aj) * F[target_right] - cj / dj
        arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
        y[target_right] = bj + (1.0 / cj) * erfinv(arg)

    # Target left branch: F_j^{-1}(F) = bj + (1/dj)*erfinv((2*dj/(sqrt(pi)*aj))*F - 1)
    if target_left.any():
        arg = (2.0 * dj) / (np.sqrt(np.pi) * aj) * F[target_left] - 1.0
        arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
        y[target_left] = bj + (1.0 / dj) * erfinv(arg)

    return y


def _T_exp_gauss(x, bi, ci, di, bj, cj, dj):
    """Eq. 123: transport map from Exp_i (source) to GaussianLike_j (target), per-dimension.
    Uses CDF-based case selection to avoid discontinuities from threshold clipping."""
    ai = (ci * di) / (ci + di)
    aj = (2.0 * cj * dj) / (np.sqrt(np.pi) * (cj + dj))

    # Compute source CDF F_i(x) for all points (DoubleExponential CDF)
    F = np.empty_like(x)
    right = x >= bi
    left = ~right
    F[right] = 1.0 - (ai / ci) * np.exp(-ci * (x[right] - bi))
    F[left] = (ai / di) * np.exp(di * (x[left] - bi))

    # Target CDF at bj: for GaussianLike, CDF_j(bj) = sqrt(pi)*aj/(2*dj)
    F_bj = np.sqrt(np.pi) * aj / (2.0 * dj)

    # Case split: target right (F >= F_bj) vs target left (F < F_bj)
    target_right = F >= F_bj
    target_left = ~target_right

    y = np.empty_like(x)

    # Target right branch: F_j^{-1}(F) = bj + (1/cj)*erfinv((2*cj/(sqrt(pi)*aj))*F - cj/dj)
    if target_right.any():
        arg = (2.0 * cj) / (np.sqrt(np.pi) * aj) * F[target_right] - cj / dj
        arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
        y[target_right] = bj + (1.0 / cj) * erfinv(arg)

    # Target left branch: F_j^{-1}(F) = bj + (1/dj)*erfinv((2*dj/(sqrt(pi)*aj))*F - 1)
    if target_left.any():
        arg = (2.0 * dj) / (np.sqrt(np.pi) * aj) * F[target_left] - 1.0
        arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)
        y[target_left] = bj + (1.0 / dj) * erfinv(arg)

    return y


def _T_gauss_exp(x, bi, ci, di, bj, cj, dj):
    """Eq. 124: transport map from GaussianLike_i (source) to Exp_j (target), per-dimension.
    Uses CDF-based case selection to avoid discontinuities from threshold clipping."""
    ai = (2.0 * ci * di) / (np.sqrt(np.pi) * (ci + di))
    aj = (cj * dj) / (cj + dj)

    # Compute source CDF F_i(x) for all points
    F = np.empty_like(x)
    right = x >= bi
    left = ~right
    F[right] = np.sqrt(np.pi) * ai / 2.0 * (1.0 / di + (1.0 / ci) * erf(ci * (x[right] - bi)))
    F[left] = np.sqrt(np.pi) * ai / 2.0 * (1.0 / di + (1.0 / di) * erf(di * (x[left] - bi)))

    # Target CDF at bj: CDF_j(bj) = aj/dj
    F_bj = aj / dj

    # Case split: target right (F >= F_bj) vs target left (F < F_bj)
    target_right = F >= F_bj
    target_left = ~target_right

    y = np.empty_like(x)

    # Target right branch: T(x) = bj - (1/cj) * log(cj*(1-F)/aj)
    if target_right.any():
        arg = cj * (1.0 - F[target_right]) / aj
        arg = np.clip(arg, 1e-300, None)
        y[target_right] = bj - (1.0 / cj) * np.log(arg)

    # Target left branch: T(x) = bj + (1/dj) * log(dj*F/aj)
    if target_left.any():
        arg = dj * F[target_left] / aj
        arg = np.clip(arg, 1e-300, None)
        y[target_left] = bj + (1.0 / dj) * np.log(arg)

    return y


def _T_component(x_d, type_s, bi, ci_s, di_s, type_t, bj, cj_t, dj_t):
    """Dispatch to the correct closed-form 1-D transport map based on component types."""
    is_exp_s = type_s in ('Exp', 'A')
    is_exp_t = type_t in ('Exp', 'A')

    if is_exp_s and is_exp_t:
        return _T_exp_exp(x_d, bi, ci_s, di_s, bj, cj_t, dj_t)
    elif (not is_exp_s) and (not is_exp_t):
        return _T_gauss_gauss(x_d, bi, ci_s, di_s, bj, cj_t, dj_t)
    elif is_exp_s and (not is_exp_t):
        return _T_exp_gauss(x_d, bi, ci_s, di_s, bj, cj_t, dj_t)
    else:
        return _T_gauss_exp(x_d, bi, ci_s, di_s, bj, cj_t, dj_t)


# -------------------------------------------------------
# OMT point transport (Eq. 9): T_omt(x) = sum_ij w~_ij(x) T_ij(x)
# -------------------------------------------------------
from collections import Counter
def points_transport(x, weights, solver):
    """
    Pushforward function using deterministic min-distance component selection.
    """
    mm_s = solver['mm_s']
    mm_t = solver['mm_t']

    types_s = solver['mm_s']['types']
    types_t = solver['mm_t']['types']

    K_0 = solver['mm_s']['K']
    K_1 = solver['mm_t']['K']
    T_x = []

    if all(t in ('Gaussian', 'A') for t in types_s) and all(t in ('Gaussian', 'A') for t in types_t):
        mu_0, cov_0 = mm_s['means'], mm_s["covariances"]
        mu_1, cov_1 = mm_t['means'], mm_t["covariances"]

        cov_0_vec = cov_0[:, np.newaxis, :, :]
        cov_1_vec = cov_1[np.newaxis, :, :, :]
        A = vectorized_geometric_mean(cov_0_vec, cov_1_vec)

        for n in range(x.shape[0]):
            point = x[n]

            # 1. Find source component i with min distance from point to mu_0[i]
            # Use squared euclidean distance for efficiency
            dist_s = np.sum((mu_0 - point)**2, axis=1)
            exp_i = np.argmin(dist_s)

            # 2. Find target component j
            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))

            # Transport calculation
            transported_point = np.dot(A[exp_i, exp_j, :, :], (point - mu_0[exp_i])) + mu_1[exp_j]
            T_x.append(transported_point)

    elif all(t in ('Exp', 'A') for t in types_s) and all(t in ('Exp', 'A') for t in types_t):
        b_s, c_s, d_s = mm_s['means'], mm_s['c_rates'], mm_s['d_rates']
        b_t, c_t, d_t = mm_t['means'], mm_t['c_rates'], mm_t['d_rates']

        # Clamp range: 3/rate covers ~95% of exp tail mass
        n_sigmas = 3.0
        clamp_upper = b_t + n_sigmas / c_t
        clamp_lower = b_t - n_sigmas / d_t

        for n in range(x.shape[0]):
            point = x[n]

            dist_s = np.sum((b_s - point)**2, axis=1)
            exp_i = np.argmin(dist_s)

            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))

            D = point.shape[0]
            transported_point = np.empty(D)
            for d in range(D):
                transported_point[d] = _T_exp_exp(
                    np.atleast_1d(point[d]),
                    b_s[exp_i][d], c_s[exp_i][d], d_s[exp_i][d],
                    b_t[exp_j][d], c_t[exp_j][d], d_t[exp_j][d],
                )[0]
                transported_point[d] = np.clip(
                    transported_point[d], clamp_lower[exp_j, d], clamp_upper[exp_j, d])
            T_x.append(transported_point)

    else:
        b_s, c_s, d_s = mm_s['means'], mm_s['c_rates'], mm_s['d_rates']
        b_t, c_t, d_t = mm_t['means'], mm_t['c_rates'], mm_t['d_rates']

        # Clamp range for exp target components
        n_sigmas = 3.0
        clamp_upper_t = b_t + n_sigmas / c_t
        clamp_lower_t = b_t - n_sigmas / d_t

        for n in range(x.shape[0]):
            point = x[n]

            dist_s = np.sum((b_s - point)**2, axis=1)
            exp_i = np.argmin(dist_s)

            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))

            D = point.shape[0]
            transported_point = np.empty(D)
            for d in range(D):
                transported_point[d] = _T_component(
                    np.atleast_1d(point[d]),
                    types_s[exp_i], b_s[exp_i][d], c_s[exp_i][d], d_s[exp_i][d],
                    types_t[exp_j], b_t[exp_j][d], c_t[exp_j][d], d_t[exp_j][d],
                )[0]
                # Clamp if target component is Exp
                if types_t[exp_j] in ('Exp', 'A'):
                    transported_point[d] = np.clip(
                        transported_point[d], clamp_lower_t[exp_j, d], clamp_upper_t[exp_j, d])
            T_x.append(transported_point)

    return np.array(T_x)
