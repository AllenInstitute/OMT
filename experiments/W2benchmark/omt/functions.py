import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import ot
from tqdm import tqdm
import torch
import pdb
from scipy.special import softmax
import scipy.stats as stats
from collections import Counter
from sklearn.metrics.pairwise import rbf_kernel
# from ignite.metrics import MaximumMeanDiscrepancy


def entropy(p):
    return np.sum(p * np.log(p))

def d_entropy(p):
    return np.log(p) + 1.

def inv_(A):
    return np.linalg.inv(A)

def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)

def batch_sqrtm(A):
    """
    Computes the square root of a stack of symmetric positive semi-definite matrices.
    Input A has shape (..., n, n).
    """
    # Use np.linalg.eigh for symmetric matrices; it's faster and more stable.
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Compute sqrt of eigenvalues, clip to avoid nan from small numerical errors.
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    
    # Reconstruct the matrix square root.
    # eigvecs @ diag(sqrt_eigvals) @ eigvecs.T
    # We use broadcasting and transpose to do this for the whole batch.
    return eigvecs @ np.apply_along_axis(np.diag, -1, sqrt_eigvals) @ np.transpose(eigvecs, axes=(*range(A.ndim - 2), -1, -2))


def vectorized_eogt_dist(mu_0, mu_1, sigma_0, sigma_1, eps):
    """
    A fully vectorized version of the eogt distance calculation for the multi-dimensional case.
    """
    # d is the feature dimension, taken from the last axis of the mean vectors.
    d = mu_0.shape[-1]
    # All operations are now on batches of matrices/vectors.
    # @ operator handles batch matrix multiplication.
    sigma_01 = sigma_0 @ sigma_1
    
    # np.linalg.eigvals is batch-aware.
    sigma_eig = np.linalg.eigvals(sigma_01)
    # Reshape eps for broadcasting with higher-dimensional arrays.
    eps_mat = eps[..., np.newaxis] # Shape: (K_s, K_t, 1)

    # pdb.set_trace()
    # try:
    eps_eig = np.sqrt(sigma_eig + eps_mat**2 / 4)
    # except:
    #     pdb.set_trace()

    # --- Calculate the final distance, using axis arguments for batch operations ---
    
    # Norm along the last dimension (d). Result shape: (K_s, K_t)
    dist_mu = np.linalg.norm(mu_0 - mu_1, ord=2, axis=-1)**2

    # Trace over the last two dimensions (d, d). Result shape: (K_s, K_t)
    tr_sigma_0 = np.trace(sigma_0, axis1=-2, axis2=-1)
    tr_sigma_1 = np.trace(sigma_1, axis1=-2, axis2=-1)
    
    # Sum over the last dimension (d). Result shape: (K_s, K_t)
    sum_eps_eig = np.sum(eps_eig, axis=-1)
    log_term = eps * np.sum(np.log(2*eps_eig + eps_mat), axis=-1)

    # Final distance calculation
    if eps[0,0] == 0:
        dist = (dist_mu + tr_sigma_0 + tr_sigma_1 - 2*sum_eps_eig)
    else: 
        dist = (dist_mu + tr_sigma_0 + tr_sigma_1 - 2*sum_eps_eig +
                log_term + d*eps*(1 - np.log(2*eps)))
    
    return dist, 1, 1

def eogt(mu_0, mu_1, sigma_0, sigma_1, eps):
    if np.isscalar(mu_0):
        eps_eig = np.sqrt((sigma_0 * sigma_1) + eps**2/4)
        dist = (mu_0 - mu_1)**2 + sigma_0 + sigma_1 - 2*eps_eig + eps*np.log(2*eps_eig + eps) + eps*(1- np.log(2*eps))
        D = np.sqrt(4*sigma_0 * sigma_1 + eps**2)
        cross_sigma = 0.5*(D - eps)
        pi_mu = np.array([mu_0, mu_1])
        pi_sigma = np.array([[sigma_0, cross_sigma], [cross_sigma, sigma_1]])
    else:
        d = mu_0.shape[0]
        eye_d = np.eye(d)
        sigma_0_sqrt = sqrtm(sigma_0)
        inv_sigma_0_sqrt = np.linalg.inv(sigma_0_sqrt)
        sigma_01 = sigma_0 @ sigma_1
        sigma_eig = np.linalg.eigvals(sigma_01)
        eps_eig = np.sqrt(sigma_eig + eps**2/4)
        dist = np.linalg.norm(mu_0 - mu_1, ord=2)**2 + np.trace(sigma_0) + np.trace(sigma_1) - 2*np.sum(eps_eig) + eps*np.sum(np.log(2*eps_eig + eps)) + d*eps*(1- np.log(2*eps))
        D = 4*sigma_0_sqrt @ sigma_1 @ sigma_0_sqrt + eps**2*eye_d
        D = sqrtm(D)
        cross_sigma = 0.5*(sigma_0_sqrt @ D @ inv_sigma_0_sqrt - eps*eye_d)
        pi_mu = np.squeeze(np.concatenate((mu_0, mu_1)))
        pi_sigma = np.zeros((2*d, 2*d))
        pi_sigma[:d, :d] = sigma_0
        pi_sigma[d:, d:] = sigma_1
        pi_sigma[:d, d:] = cross_sigma
        pi_sigma[d:, :d] = cross_sigma.T

    return dist, pi_mu, pi_sigma


def geometric_mean(cov1, cov2, eps=0.0):
    """
    Computes the geometric mean of two covariance matrices.

    Args:
        cov1 (np.ndarray): First covariance matrix.
        cov2 (np.ndarray): Second covariance matrix.
        eps (float): Small value to avoid singularities.

    Returns:
        np.ndarray: Geometric mean of the two covariance matrices.
    """
    if cov1.shape != cov2.shape:
        raise ValueError("Covariance matrices must have the same shape.")
    
    d = cov1.shape[0]
    eye_d = np.eye(d)
    cov1_sqrt = sqrtm(cov1)
    inv_cov1_sqrt = np.linalg.inv(cov1_sqrt)
    
    # D = 4*cov1_sqrt @ cov2 @ cov1_sqrt + eps**2*eye_d
    D = cov1_sqrt @ cov2 @ cov1_sqrt
    D = sqrtm(D)

    ent_geom_mean = 0.5*(cov1_sqrt @ D @ inv_cov1_sqrt - eps*eye_d)
    
    return inv_cov1_sqrt @ D @ inv_cov1_sqrt


def vectorized_geometric_mean(cov_0, cov_1):
    """
    Computes the geometric mean for all pairs of covariance matrices from two stacks.

    Args:
        cov_0 (np.ndarray): First stack of matrices, shape (K_0, 1, d, d).
        cov_1 (np.ndarray): Second stack of matrices, shape (1, K_1, d, d).

    Returns:
        np.ndarray: A stack of geometric means, shape (K_0, K_1, d, d).
    """
    # All functions are now batch-aware.
    cov_0_sqrt = batch_sqrtm(cov_0)
    inv_cov_0_sqrt = np.linalg.inv(cov_0_sqrt)
    
    # The @ operator handles batch matrix multiplication.
    # Broadcasting expands (K_0,1,d,d) and (1,K_1,d,d) to (K_0,K_1,d,d).
    D = cov_0_sqrt @ cov_1 @ cov_0_sqrt
    D_sqrt = batch_sqrtm(D)

    # Calculate the final result for all pairs at once.
    A = inv_cov_0_sqrt @ D_sqrt @ inv_cov_0_sqrt
    
    return A


def trans_interpolate(mu_0, mu_1, cov_0, cov_1, t, mode='linear'):
    mu_t = (1-t)*mu_0 + t*mu_1
    if mode == 'linear':
        cov_t = (1-t)*cov_0 + t*cov_1
    elif mode == 'geodesic':
        cov_t = geodesic_covariance_interpolation(cov_0, cov_1, t)
    else:
        raise ValueError("Invalid mode. Choose 'linear' or 'geodesic'.")
    
    return mu_t, cov_t


def matrix_sqrt(mat):
    # Eigen decomposition: mat = Q Λ Qᵀ
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 1e-12)  # prevent sqrt of zero or negative
    sqrt_vals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_vals) @ eigvecs.T


def matrix_inv_sqrt(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 1e-12)  # clamp small values
    inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
    return eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T


def geodesic_covariance_interpolation(sigma0, sigma1, t):
    sqrt_sigma0 = matrix_sqrt(sigma0)
    inv_sqrt_sigma0 = matrix_inv_sqrt(sigma0)
    
    middle = inv_sqrt_sigma0 @ sigma1 @ inv_sqrt_sigma0
    eigvals, eigvecs = np.linalg.eigh(middle)
    eigvals = np.maximum(eigvals, 1e-12)  # clamp to avoid complex numbers
    powered = eigvecs @ np.diag(eigvals ** t) @ eigvecs.T
    
    sigma_t = sqrt_sigma0 @ powered @ sqrt_sigma0
    return sigma_t



def generate_gmm(K=1, w=None, dim=1, n_smp=1e4, seed=0, mu_range=(0, 15), std_range=(0.1, 1), x_range=None):

    np.random.seed(seed)
    # Generate random means and standard deviations
    mean = np.random.uniform(mu_range[0], mu_range[1], size=(K, dim))
    std = np.random.uniform(std_range[0], std_range[1], size=(K, dim))
    
    if w is None:
        w = np.random.dirichlet(np.ones(K), size=1)[0]

    if x_range is None:
        x_range = (np.min(mean) - 5*np.max(std), np.max(mean) + 5*np.max(std))
        
    x = np.linspace(x_range[0], x_range[1], int(n_smp))
    
    gmm_prob = 0.
    for i in range(K):
        # gmm_prob += w[i] * ot.datasets.make_1D_gauss(n=n_smp, m=np.where(np.abs(x - mean[i]) < .1)[0][0], s=int(std[i]*n_smp/(max(x) - min(x))))
        gmm_prob += w[i] * norm.pdf(x, loc=mean[i], scale=std[i])
    
    gmm_prob /= gmm_prob.sum()
    
    return gmm_prob, x, mean, std**2, w


def omt_map(x, y, mu, sigma, w):
    
    X, Y = np.meshgrid(x, y)
    pos = np.vstack([X.ravel(), Y.ravel()]).T
    pi_omt = np.zeros(pos.shape[0])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i, j] < 1e-3:
                continue
            pi_ = multivariate_normal.pdf(pos, mean=mu[i, j], cov=sigma[i, j], allow_singular=False)
            w[i, j] = np.where(w[i, j] < 1e-6, 0, w[i, j])
            pi_ = np.where(pi_ < 1e-2, 0, pi_)
            pi_omt += w[i, j] * pi_

    pi_omt = pi_omt.reshape((x.shape[0], y.shape[0]))
    
    return pi_omt     


def gmm_pdf_scipy(x, weights, means, covariances):
    """
    Calculates the PDF of a GMM at points x.

    Args:
        x (np.ndarray): Array of points to evaluate the PDF at.
                        Shape (n_points, 2) or (2,) for a single point.
        weights (np.ndarray): Weights of the GMM components. Shape (n_components,).
                               Must sum to 1.
        means (np.ndarray): Means of the GMM components. Shape (n_components, 2).
        covariances (np.ndarray): Covariance matrices of the GMM components.
                                  Shape (n_components, 2, 2).

    Returns:
        np.ndarray: The PDF values evaluated at each point in x.
                    Shape (n_points,) or a scalar if x is a single point.
    """
  
    n_points = x.shape[0]
    n_components = len(weights)

    if means.shape != (n_components, 2):
        raise ValueError(f"Means shape mismatch. Expected ({n_components}, 2), got {means.shape}")
    if covariances.shape != (n_components, 2, 2):
        raise ValueError(f"Covariances shape mismatch. Expected ({n_components}, 2, 2), got {covariances.shape}")
    if not np.isclose(np.sum(weights), 1.0):
         print(f"Warning: Weights do not sum to 1 (sum={np.sum(weights)})")
        # You might want to normalize weights here, or raise an error depending on strictness
        # weights = weights / np.sum(weights)


    pdf_values = np.zeros(n_points)
    for k in range(n_components):
        try:
            # Calculate PDF of the k-th component for all points x
            component_pdf = multivariate_normal.pdf(x, mean=means[k], cov=covariances[k], allow_singular=False)
            # Add the weighted PDF to the total
            pdf_values += weights[k] * component_pdf
        except np.linalg.LinAlgError as e:
            print(f"Warning: Singular covariance matrix for component {k}. PDF contribution is 0. Error: {e}")
            # If allow_singular=True, scipy handles it, otherwise pdf() raises LinAlgError
            # If allow_singular=False (default), contribution is effectively 0 where PDF isn't infinite.
            pass # Contribution remains 0


    # If input x was a single point (shape (2,)), return a scalar
    if x.shape[0] == 1 and n_points == 1 and len(pdf_values) == 1:
         return pdf_values[0]
    else:
        return pdf_values



def sample_gmm_1d(weights, means, cov, n_samples, random_state=None):
    """
    Generates random samples from a 1-dimensional Gaussian Mixture Model (GMM).

    Args:
        n_samples (int): The total number of samples to generate.
        weights (array-like): Weights of the Gaussian components.
                               Shape (n_components,). Must sum to 1.0.
        means (array-like): Means of the Gaussian components.
                            Shape (n_components,).
        cov (array-like): Covariances of the Gaussian components.
                               Shape (n_components,). Must be > 0.
        random_state (int, np.random.RandomState instance, np.random.Generator instance or None, optional):
            Determines random number generation for reproducibility.

    Returns:
        tuple: A tuple containing:
            - samples (np.ndarray): The generated 1D samples, shape (n_samples,).
            - labels (np.ndarray): The component index (0 to n_components-1)
                                   from which each sample was generated,
                                   shape (n_samples,).

    Raises:
        ValueError: If input parameters or weights are invalid or inconsistent.
    """
    # --- Input Validation ---
    weights = np.asarray(weights)
    means = np.asarray(means)
    cov = np.asarray(cov)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Component weights must sum to 1.0")
    n_components = len(weights)
    if not (len(means) == n_components and len(cov) == n_components):
        raise ValueError("Lengths of weights, means, and std_devs must match.")
    if np.any(cov <= 0):
        raise ValueError("Standard deviations must be > 0.")

    # --- Setup Random State ---
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        raise TypeError("random_state must be int, numpy Generator/RandomState, or None")

    # --- Sampling ---
    # 1. Choose which component each sample will come from based on weights
    component_labels = rng.choice(n_components, size=n_samples, p=weights, replace=True)

    # 2. Generate samples from the chosen components
    samples = np.empty(n_samples, dtype=float)
    for k in range(n_components):
        # Find which samples need to be generated from component k
        indices_k = np.where(component_labels == k)[0]
        n_samples_k = len(indices_k)

        if n_samples_k > 0:
            # Generate samples for component k using its mean and std dev
            samples_k = rng.normal(loc=means[k], scale=np.sqrt(cov)[k], size=n_samples_k)
            # Place generated samples into the correct rows of the output array
            samples[indices_k] = samples_k

    return samples, component_labels
    

def sample_gmm_kdim(weights, means, covariances, n_samples, random_state=None):
    """
    Generates samples from a K-dimensional Gaussian Mixture Model (GMM).

    Args:
        n_samples (int): The total number of samples to generate.
        weights (array-like): The weights of the Gaussian components.
            Shape (n_components,). Must sum to 1.0.
        means (array-like): The mean vectors of the Gaussian components.
            Shape (n_components, K), where K is the dimensionality.
        covariances (array-like): The covariance matrices of the Gaussian components.
            Shape (n_components, K, K). Each matrix must be symmetric and
            positive semi-definite.
        random_state (int, np.random.RandomState instance or None, optional):
            Determines random number generation for reproducibility.
            Defaults to None.

    Returns:
        tuple: A tuple containing:
            - samples (np.ndarray): The generated samples, shape (n_samples, K).
            - labels (np.ndarray): The component index (0 to n_components-1)
              from which each sample was generated, shape (n_samples,).

    Raises:
        ValueError: If input shapes or values are inconsistent.
    """
    # --- Input Validation ---
    weights = np.asarray(weights) / np.sum(weights)
    means = np.asarray(means)
    covariances = np.asarray(covariances)


    n_components = len(weights)
    if not (means.shape[0] == n_components and covariances.shape[0] == n_components):
        raise ValueError("Mismatch between number of components in weights, means, and covariances.")

    if means.ndim != 2:
        raise ValueError("Means must be a 2D array (n_components, K).")
    K = means.shape[1] # Dimensionality

    # if covariances.ndim != 3 or covariances.shape[1:] != (K, K):
    #      raise ValueError(f"Covariances must be a 3D array (n_components, {K}, {K}).")

    # --- Setup Random State ---
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise TypeError("random_state must be int, np.random.RandomState, or None")

    # --- Sampling ---
    # 1. Choose which component each sample will come from based on weights
    component_choices = rng.choice(n_components, size=n_samples, p=weights, replace=True)

    # 2. Generate samples from the chosen components
    samples = np.empty((n_samples, K))

    for k in range(n_components):
        # Find which samples need to be generated from component k
        indices_k = np.where(component_choices == k)[0]
        n_samples_k = len(indices_k)

        if n_samples_k > 0:
            # Generate samples for component k
            samples_k = rng.multivariate_normal(mean=means[k], cov=covariances[k],size=n_samples_k)
            # Place generated samples into the correct rows of the output array
            samples[indices_k] = samples_k

    return samples, component_choices



def prob_gmm_kdim(weights, means, covariances, n_samples, samples, random_state=None):
    
    weights = np.asarray(weights) / np.sum(weights)
    means = np.asarray(means)
    covariances = np.asarray(covariances)


    n_components = len(weights)
    if not (means.shape[0] == n_components and covariances.shape[0] == n_components):
        raise ValueError("Mismatch between number of components in weights, means, and covariances.")

    if means.ndim != 2:
        raise ValueError("Means must be a 2D array (n_components, K).")
    K = means.shape[1] # Dimensionality

    # if covariances.ndim != 3 or covariances.shape[1:] != (K, K):
    #      raise ValueError(f"Covariances must be a 3D array (n_components, {K}, {K}).")

    # --- Setup Random State ---
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise TypeError("random_state must be int, np.random.RandomState, or None")

    
    # data_smp = np.linspace(data_range[0], data_range[1], n_samples)

    prob_smp = 0
    for k in range(n_components):
        # Generate samples for component k
        prob_smp += weights[k] * multivariate_normal(mean=means[k], cov=covariances[k], allow_singular=False).pdf(samples)

    return prob_smp



def map_smp(weights, means, covariances, n_samples, x, random_state=None):
    
    samples, _ = sample_gmm_kdim(weights, means, covariances, n_samples)
    
    dist = np.linalg.norm(samples - x, ord=2, axis=1)

    return samples[np.argmin(dist)]


def sample_gamma_gmm_mixture(gamma_shape,
                             gamma_scale,
                             n_gmm_components,
                             w_gmm=0.5,
                             n_samples=10000,
                             random_state=None):

    # --- Input Validation ---
    if gamma_shape <= 0 or gamma_scale <= 0:
        raise ValueError("Gamma shape and scale parameters must be > 0.")
    
    # --- Setup Random State ---
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        raise TypeError("random_state must be int, numpy Generator/RandomState, or None")

    # --- Determine Number of Samples from Each Component ---
    n_gmm = n_samples // 2
    n_gamma = n_samples // 2

    x_smp = np.linspace(0, 15, n_samples)
    # --- Sample from GMM Component ---
    gmm_prob = np.array([])
    gmm_samples = np.array([])
    if n_gmm > 0:
        _, _, gmm_mean, gmm_cov, gmm_w = generate_gmm(K=n_gmm_components, n_smp=n_gmm, x_range=(0, 10), seed=random_state)
        gmm_samples, _ = sample_gmm_1d(gmm_w, gmm_mean, gmm_cov, n_gmm, random_state=random_state)
        gmm_prob = 0.
        for mu, sigma, w in zip(gmm_mean, gmm_cov, gmm_w ):
            gmm_prob += w * norm.pdf(x_smp, loc=mu, scale=np.sqrt(sigma))
        gmm_prob /= gmm_prob.sum()

    # --- Sample from Gamma Component ---
    gamma_samples = np.array([])
    gamma_prob = np.array([])
    if n_gamma > 0:
        gamma_samples = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=n_gamma)
        gamma_prob = stats.gamma.pdf(x_smp, a=gamma_shape, scale=gamma_scale)

    # --- Combine and Shuffle ---
    mixture_samples = np.concatenate((gmm_samples, gamma_samples))
    rng.shuffle(mixture_samples)

    # --- Calculate the final mixture PDF ---
    mixture_prob = w_gmm * gmm_prob + (1.0 - w_gmm) * gamma_prob

    return mixture_samples, mixture_prob, x_smp



def fwd_map(
                weights,
                gmm_s, 
                gmm_t, 
                x,
                ):

    comp_ind = np.argmax(weights[gmm_s.predict(x), :], axis=1)
    x_push = np.array([gmm_t.means_[cp] for cp in comp_ind])
    
    # w_t = np.sum(weights, axis=0)
    
    # inputs = [
    #         (w_t[comp_ind[i]], gmm_t.means_[comp_ind[i]], gmm_t.covariances_[comp_ind[i]], x_push[i], 1000)
    #         for i in range(len(comp_ind))
    #         ]

    # def wrapper(args):
    #     return map_smp(*args)
    
    # map_results = list(map(wrapper, inputs))
    
    return x_push #, np.array(map_results)


def inv_transport(
                means, 
                covariances, 
                weights, 
                y,
                x_range,
                n_samples=1000,
                ):
    """
    Calculates the probability density function (PDF) of a 2D Gaussian Mixture Model (GMM)
    over a specified grid.

    Args:
        means (list of np.ndarray or list of lists): 
            A list of K means for the K Gaussian components. Each mean should be a 
            1D array or list of length 2 (e.g., [[mu1_x, mu1_y], [mu2_x, mu2_y], ...]).
        covariances (list of np.ndarray or list of lists): 
            A list of K covariance matrices (2x2) for the K Gaussian components
            (e.g., [[[cov1_xx, cov1_xy], [cov1_yx, cov1_yy]], ...]).
        weights (list of float): 
            A list of K weights for the K components. These should ideally sum to 1.
        x_min (float): Minimum value for the x-axis of the grid.
        x_max (float): Maximum value for the x-axis of the grid.
        y_min (float): Minimum value for the y-axis of the grid.
        y_max (float): Maximum value for the y-axis of the grid.
        num_points_x (int, optional): Number of points along the x-axis. Defaults to 100.
        num_points_y (int, optional): Number of points along the y-axis. Defaults to 100.

    Returns:
        tuple: (X_grid, Y_grid, PDF_values)
            X_grid (np.ndarray): A 2D array of x-coordinates of the grid points.
            Y_grid (np.ndarray): A 2D array of y-coordinates of the grid points.
            PDF_values (np.ndarray): A 2D array containing the GMM PDF values at each grid point.
    """

    # --- Input Validation (Basic) ---
    num_components = len(weights)
    if not (len(means) == num_components and len(covariances) == num_components):
        raise ValueError("The number of means, covariances, and weights must be the same.")
    
    if not np.isclose(np.sum(weights), 1.0):
        print("Warning: Component weights do not sum to 1. Normalizing weights.")
        weights = np.array(weights) / np.sum(weights)

    # --- Grid Creation ---
    x_coords = np.linspace(x_range[0], x_range[1], n_samples)

    # `pos` will be an array of shape (num_points_y, num_points_x, 2)
    # representing all (x,y) coordinate pairs on the grid.
    pos = np.hstack((x_coords, np.tile(y, (len(x_coords), 1))))

    # --- PDF Calculation ---
    # Initialize PDF_values to zeros, with the same shape as X_grid or Y_grid.
    PDF_values = np.zeros(n_samples, dtype=float)

    for k in range(num_components):
        mean_k = np.array(means[k])
        cov_k = np.array(covariances[k])
        weight_k = weights[k]

        # Calculate the PDF of the k-th Gaussian component over all points in `pos`.
        try:
            component_pdf = multivariate_normal(mean=mean_k, cov=cov_k, allow_singular=False).pdf(pos)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Singular covariance matrix for component {k}: {cov_k}. Error: {e}. "
                             "Consider adding a small jitter (e.g., np.eye(2)*1e-6) if near singular.")
        
        # Add the weighted PDF of the current component to the total GMM PDF.
        PDF_values += weight_k * component_pdf
        
    return PDF_values, x_coords



def points_transport_(x, weights, solver, direction="fwd", num_smp=1000):
    """
    Pushforward function for a given input x using the specified solver.

    Args:
        x (np.ndarray): Input data to be transformed.
        solver: The solver object that contains the transport map.
        eps (float): Small value to avoid singularities.

    Returns:
        np.ndarray: Transformed data after applying the pushforward function.
    """
    
    # Compute the transport map
    mu_0, cov_0 = solver["gmm_s"].means_, solver["gmm_s"].covariances_
    mu_1, cov_1 = solver["gmm_t"].means_, solver["gmm_t"].covariances_
    
    K_0 = weights.shape[0]
    K_1 = weights.shape[1]
    
    if direction == "fwd":
        # M = np.zeros((x.shape[0], x.shape[1], K_0, K_1))
        M = np.zeros((x.shape[0], x.shape[1], K_0))
        source_cmp = solver["gmm_s"].predict(x)
        m_ij = []
        for i in range(K_0):
            comps = np.random.choice(range(K_1), size=K_1*10, p=weights[i, :]/np.sum(weights[i, :]))
            exp_j = Counter(comps).most_common(1)[0][0]
            for j in range(K_1):
                A_ij = geometric_mean(cov_0[i], cov_1[j], eps=0.0)
                # M[:,:,i,j] = np.einsum('ij,kj->ki', A_ij, x - mu_0[i]) + mu_1[j]
                m_ij.append(np.einsum('ij,kj->ki', A_ij, x - mu_0[i]) + mu_1[j])
            
            M[:, :, i] = np.array(m_ij)[exp_j]
             
        # v = np.sum([np.multiply(weights[:, j], M[:, :, :, j]) for j in range(K_1)], axis=0)
        # T_x = np.divide(v, np.sum(weights, axis=1))
        # T_x = np.mean(T_x, axis=2)
        # T_x = np.mean(M, axis=2)
        T_x = M[range(x.shape[0]), :, source_cmp]
        
    elif direction == "inv":
        M = np.zeros((x.shape[0], x.shape[1], K_1, K_0))
        for i in range(K_1):
            for j in range(K_0):
                A_ij = geometric_mean(cov_1[i], cov_0[j], eps=0.0)
                M[:,:,i,j] = np.einsum('ij,kj->ki', A_ij, x - mu_1[i]) + mu_0[j]
                
        v = np.sum([np.multiply(weights[j, :].T, M[:, :, :, j]) for j in range(K_1)], axis=0)
        T_x = np.divide(v, np.sum(weights, axis=0))
        T_x = np.mean(T_x, axis=2)
        
    else:
        raise ValueError("Invalid direction. Use 'fwd' or 'inv'.")

    return T_x



def points_transport_(x, weights, solver, direction="fwd", num_smp=1000):
    """
    Pushforward function for a given input x using the specified solver.

    Args:
        x (np.ndarray): Input data to be transformed.
        solver: The solver object that contains the transport map.
        eps (float): Small value to avoid singularities.

    Returns:
        np.ndarray: Transformed data after applying the pushforward function.
    """
    
    # Compute the transport map
    mu_0, cov_0 = solver["gmm_s"].means_, solver["gmm_s"].covariances_
    mu_1, cov_1 = solver["gmm_t"].means_, solver["gmm_t"].covariances_
    
    K_0 = weights.shape[0]
    K_1 = weights.shape[1]
    T_x = []
    
    if direction == "fwd":
        for n in range(x.shape[0]):
            prob_s = solver["gmm_s"].predict_proba(x[n].reshape(1, -1))
            exp_i = np.random.choice(range(K_0), p=prob_s[0])
            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))
            A_ij = geometric_mean(cov_0[exp_i], cov_1[exp_j], eps=0.0)
            # M[:,:,i,j] = np.einsum(ij,kj->ki', A_ij, x - mu_0[i]) + mu_1[j]
            T_x.append(np.dot(A_ij, (x[n] - mu_0[exp_i]).T).T + mu_1[exp_j])
        
    elif direction == "inv":
       for n in range(x.shape[0]):
            prob_t = solver["gmm_t"].predict_proba(x[n].reshape(1, -1))
            exp_j = np.random.choice(range(K_1), p=prob_t[0])
            exp_i = np.random.choice(range(K_0), p=weights[:, exp_j]/np.sum(weights[:, exp_j]))
            A_ij = geometric_mean(cov_1[exp_j], cov_0[exp_i], eps=0.0)
            # M[:,:,i,j] = np.einsum(ij,kj->ki', A_ij, x - mu_0[i]) + mu_1[j]
            T_x.append(np.dot(A_ij, (x[n] - mu_1[exp_j]).T).T + mu_0[exp_i])
            
    else:
        raise ValueError("Invalid direction. Use 'fwd' or 'inv'.")

    return np.array(T_x)


def points_transport_vectorized(x, weights, solver, direction="fwd", num_smp=1000):
    mu_0, cov_0 = solver["gmm_s"].means_, solver["gmm_s"].covariances_
    mu_1, cov_1 = solver["gmm_t"].means_, solver["gmm_t"].covariances_
    
    N, D = x.shape
    K_0, K_1 = weights.shape

    T_x = np.zeros_like(x)

    if direction == "fwd":
        # Get source component probabilities for each sample
        prob_s = solver["gmm_s"].predict_proba(x)  # shape: (N, K_0)
        # Sample component index for each x[n]
        exp_i = np.array([np.random.choice(K_0, p=prob_s[n]) for n in range(N)])

        # Sample corresponding target components based on coupling weights
        weights_norm = weights[exp_i] / weights[exp_i].sum(axis=1, keepdims=True)
        exp_j = np.array([np.random.choice(K_1, p=weights_norm[n]) for n in range(N)])

        # Compute A_ij matrices for all (exp_i[n], exp_j[n]) pairs
        A_ij = np.stack([
            geometric_mean(cov_0[exp_i[n]], cov_1[exp_j[n]], eps=0.0)
            for n in range(N)
        ])  # shape: (N, D, D)

        # Compute transformed points: A_ij @ (x - mu_0[i]) + mu_1[j]
        x_shifted = x - mu_0[exp_i]  # shape: (N, D)
        T_x = np.einsum('nij,nj->ni', A_ij, x_shifted) + mu_1[exp_j]

    elif direction == "inv":
        prob_t = solver["gmm_t"].predict_proba(x)  # shape: (N, K_1)
        exp_j = np.array([np.random.choice(K_1, p=prob_t[n]) for n in range(N)])
        weights_norm = weights[:, exp_j].T / weights[:, exp_j].sum(axis=0)
        exp_i = np.array([np.random.choice(K_0, p=weights_norm[n]) for n in range(N)])
        A_ij = np.stack([
            geometric_mean(cov_1[exp_j[n]], cov_0[exp_i[n]], eps=0.0)
            for n in range(N)
        ])
        x_shifted = x - mu_1[exp_j]
        T_x = np.einsum('nij,nj->ni', A_ij, x_shifted) + mu_0[exp_i]

    else:
        raise ValueError("Invalid direction. Use 'fwd' or 'inv'.")

    return T_x



def mmd_rbf(X, Y, gamma=1.0):
    """
    Calculates the Maximum Mean Discrepancy (MMD) using the RBF kernel.

    Args:
        X (numpy.ndarray): Sample from distribution P, shape (n_samples1, n_features).
        Y (numpy.ndarray): Sample from distribution Q, shape (n_samples2, n_features).
        gamma (float): Kernel parameter for RBF kernel.

    Returns:
        float: MMD value.
    """
    kxx = rbf_kernel(X, X, gamma=gamma)
    kyy = rbf_kernel(Y, Y, gamma=gamma)
    kxy = rbf_kernel(X, Y, gamma=gamma)
    return np.mean(kxx) + np.mean(kyy) - 2 * np.mean(kxy)
    

def compute_mmd(X, Y, sigma=1.0, batch_size=5000, device='cpu'):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples using the RBF kernel.

    Args:
        X (numpy.ndarray): Sample from distribution P, shape (n_samples1, n_features).
        Y (numpy.ndarray): Sample from distribution Q, shape (n_samples2, n_features).
        gamma (float): Kernel parameter for RBF kernel.

    Returns:
        float: MMD value.
    """
    
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        return data.to(device)
    
    def rbf_kernel_torch(x, y, sigma):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * sigma ** 2))
     
    # X = to_tensor(X)
    # Y = to_tensor(Y)
    mmd_metric = []
    
    min_len = min(X.shape[0], Y.shape[0])

    # Compute K_xx
    for i in range(0, min_len, batch_size):
        end_idx = min(i + batch_size, min_len)
        x_batch = X[i:end_idx]
        y_batch = Y[i:end_idx]
        mmd_metric.append(mmd_rbf(x_batch, y_batch, sigma))

    mmd_metric = np.array(mmd_metric)
    return np.mean(mmd_metric)


def points_transport(x, weights, solver, direction="fwd", n_smp=100):
    """
    Pushforward function for a given input x using the specified solver.

    Args:
        x (np.ndarray): Input data to be transformed.
        solver: The solver object that contains the transport map.
        eps (float): Small value to avoid singularities.

    Returns:
        np.ndarray: Transformed data after applying the pushforward function.
    """
    
    # Compute the transport map
    mu_0, cov_0 = solver["gmm_s"].means_, solver["gmm_s"].covariances_
    mu_1, cov_1 = solver["gmm_t"].means_, solver["gmm_t"].covariances_

    K_0 = weights.shape[0]
    K_1 = weights.shape[1]
    T_x = []
    x_ = []
    
    if direction == "fwd":
        cov_0_vec = cov_0[:, np.newaxis, :, :]  # Shape -> (K_0, 1, d, d)
        cov_1_vec = cov_1[np.newaxis, :, :, :]  # Shape -> (1, K_1, d, d)
        A = vectorized_geometric_mean(cov_0_vec, cov_1_vec)
        
        for n in range(x.shape[0]):
            # for smp in range(10):
            prob_s = solver["gmm_s"].predict_proba(x[n].reshape(1, -1))
            prob_s[0] /= prob_s[0].sum()  # Normalize to ensure it's a valid probability distribution
            exp_i = np.random.choice(range(K_0), p=prob_s[0])
            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))         

            T_x.append(np.dot(A[exp_i, exp_j, :, :], (x[n] - mu_0[exp_i]).T).T + mu_1[exp_j])
            # print(f"Source component: {exp_i}, Target component: {exp_j}")
            
        
    elif direction == "inv":
        cov_1_vec = cov_1[:, np.newaxis, :, :]  # Shape -> (K_0, 1, d, d)
        cov_0_vec = cov_0[np.newaxis, :, :, :]  # Shape -> (1, K_1, d, d)
        A = vectorized_geometric_mean(cov_1_vec, cov_0_vec)
        for n in range(x.shape[0]):
            prob_t = solver["gmm_t"].predict_proba(x[n].reshape(1, -1))
            exp_j = np.random.choice(range(K_1), p=prob_t[0])
            exp_i = np.random.choice(range(K_0), p=weights[:, exp_j]/np.sum(weights[:, exp_j]))
            T_x.append(np.dot(A[exp_j, exp_i, :, :], (x[n] - mu_1[exp_j]).T).T + mu_0[exp_i])
            
    else:
        raise ValueError("Invalid direction. Use 'fwd' or 'inv'.")

    return np.array(T_x)


def dynamic_points_transport(x, weights, solver, geometry='linear', time_points=100):
    """
    Pushforward function for a given input x using the specified solver.

    Args:
        x (np.ndarray): Input data to be transformed.
        solver: The solver object that contains the transport map.
        eps (float): Small value to avoid singularities.

    Returns:
        np.ndarray: Transformed data after applying the pushforward function.
    """
    
    # Compute the transport map
    mu_0, cov_0 = solver["gmm_s"].means_, solver["gmm_s"].covariances_
    mu_1, cov_1 = solver["gmm_t"].means_, solver["gmm_t"].covariances_
    
    K_0 = weights.shape[0]
    K_1 = weights.shape[1]

    T_xt = np.zeros((time_points, x.shape[0], x.shape[1]))
    for i_t, t in enumerate(np.linspace(0, 1, time_points)):
        T_x = []
        A = np.zeros((K_0, K_1, x.shape[1], x.shape[1]))
        mu_t = np.zeros((K_0, K_1, x.shape[1]))
        for i in range(K_0):
            for j in range(K_1):
                mu_t[i, j, :], cov_t = trans_interpolate(mu_0[i], mu_1[j], cov_0[i], cov_1[j], t, geometry)
                A[i, j, :, :] = geometric_mean(cov_0[i], cov_t, eps=0.0)
        # for n in range(x.shape[0]):
        for n in tqdm(range(x.shape[0]), desc=f"samples - time points {i_t}", leave=True):
            # for smp in range(10):
            prob_s = solver["gmm_s"].predict_proba(x[n].reshape(1, -1))
            exp_i = np.random.choice(range(K_0), p=prob_s[0])
            exp_j = np.random.choice(range(K_1), p=weights[exp_i, :]/np.sum(weights[exp_i, :]))
            # M[:,:,i,j] = np.einsum(ij,kj->ki', A_ij, x - mu_0[i]) + mu_1[j]
            T_x.append(np.dot(A[exp_i, exp_j, :, :], (x[n] - mu_0[exp_i]).T).T + mu_t[exp_i, exp_j])
                    # x_.append(x[n])

        T_xt[i_t] = np.array(T_x)
  
    return T_xt



def transport_cost(omt_solver, weights):
    
    gmm_s = omt_solver["gmm_s"]
    gmm_t = omt_solver["gmm_t"]
    K_0 = weights.shape[0]
    K_1 = weights.shape[1]
    
    cost = 0
    for i in range(K_0):
        for j in range(K_1):
            cov_i = gmm_s.covariances_[i]
            cov_j = gmm_t.covariances_[j]
            mu_i = gmm_s.means_[i]
            mu_j = gmm_t.means_[j]
            w_ij = weights[i, j]
            cov_i_sqrt = sqrtm(cov_i)
            D = cov_i_sqrt @ cov_j @ cov_i_sqrt
            D = sqrtm(D)

            cost += w_ij * (np.trace(cov_i) + np.trace(cov_j) - 2*np.trace(D) + np.linalg.norm(mu_i - mu_j, ord=2)**2)
    
    return cost
