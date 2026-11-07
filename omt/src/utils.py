import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import ot
from scipy.stats import expon
from sklearn.datasets import make_s_curve
from scipy.special import softmax
import scipy.stats as stats
from sklearn.datasets import make_spd_matrix 


def generate_gmm(K=1, w=None, dim=1, n_smp=1e4, seed=0, mu_range=(0, 15), std_range=(0.1, 1), x_range=None, singular_cov=False):

    np.random.seed(seed)
    # Generate random means and standard deviations
    mean = np.random.uniform(mu_range[0], mu_range[1], size=(K, dim))
    std = np.random.uniform(std_range[0], std_range[1], size=(K, dim))
    
    if w is None:
        w = np.random.dirichlet(np.ones(K), size=1)[0]
    
    if singular_cov:
        std[1] /= 1e4
        w[1] /=10
        w /= sum(w)

    if x_range is None:
        x_range = (np.min(mean) - 5*np.max(std), np.max(mean) + 5*np.max(std))
        
    x = np.linspace(x_range[0], x_range[1], int(n_smp))
    if singular_cov:
        x_tmp = np.linspace(mean[1][0] - 1e-7, mean[1][0] + 1e-7, 1000)
        x = np.sort(np.concatenate((x, x_tmp)))
    
    gmm_prob = 0.
    for i in range(K):
        # gmm_prob += w[i] * ot.datasets.make_1D_gauss(n=n_smp, m=np.where(np.abs(x - mean[i]) < .1)[0][0], s=int(std[i]*n_smp/(max(x) - min(x))))
        gmm_prob += w[i] * norm.pdf(x, loc=mean[i], scale=std[i])
    
    gmm_prob /= gmm_prob.sum()
    cov = std**2
    
    gmm_samples, _ = sample_gmm_1d(w, mean, cov, n_smp)
    
    return gmm_samples, gmm_prob, x, mean, cov, w



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
        gmm_samples, _, _, gmm_mean, gmm_cov, gmm_w = generate_gmm(K=n_gmm_components, n_smp=n_gmm, x_range=(0, 10), seed=random_state)
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
    mixture_prob /= mixture_prob.sum()

    return mixture_samples, mixture_prob, x_smp



def swiss_roll(t_range=(1, 5), y_range=(0, 5), n_smp=10000, noise=0.1, dim=2):
    """
    Generate Swiss Roll data.

    Args:
        n (int): Number of samples.
        noise (float): Standard deviation of Gaussian noise added to the data.

    Returns:
        np.ndarray: Generated Swiss Roll data.
    """
    t = np.linspace(t_range[0] * np.pi, t_range[1] * np.pi, n_smp)

    # Calculate coordinates with spiral shape
    x = t * np.cos(t)
    z = t * np.sin(t)

    # Add noise
    x += noise * np.random.randn(n_smp)
    z += noise * np.random.randn(n_smp)

    if dim == 2:
        return np.vstack((x, z)).T
    elif dim == 3:
        # Generate 'Y' coordinates (height) - independent of the spiral, uniformly distributed
        y = np.random.uniform(y_range[0], y_range[1], n_smp)
        y += noise * np.random.randn(n_smp)
        return np.vstack((x, y, z)).T
    else:
        raise ValueError("dim must be 2 or 3!")
    


def RingSampler(radius, noise, n_samples):
    """
    Generates sample points forming a ring.

    Args:
        n_samples (int): The number of points to generate for the ring.
        radius (float): The radius of the ring.
        noise (float): The standard deviation of the Gaussian noise added
                       to the point coordinates.

    Returns:
        tuple: A tuple containing two numpy arrays (x_coords, y_coords)
               representing the coordinates of the sampled points.
    """
    # Generate random angles uniformly distributed between 0 and 2*pi
    angles = np.random.uniform(0, 2 * np.pi, n_samples)

    # Calculate ideal x and y coordinates on the circle
    x_ideal = radius * np.cos(angles)
    y_ideal = radius * np.sin(angles)

    # Add Gaussian noise to the coordinates
    # Generate noise for x and y from a normal distribution
    x_noise = np.random.normal(0, noise, n_samples)
    y_noise = np.random.normal(0, noise, n_samples)

    x_coords = x_ideal + x_noise
    y_coords = y_ideal + y_noise

    return x_coords, y_coords


def generate_s_curve_data(n_samples=250, noise=0.1, random_state=None):
    """
    Generates 2D data points lying on an S-curve shape.

    Uses scikit-learn's make_s_curve which generates 3D points,
    but we select only the 1st and 3rd dimensions for a 2D projection.

    Args:
        n_samples (int): The total number of points to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_state (int, optional): Determines random number generation for
                                       dataset creation. Pass an int for
                                       reproducible output across multiple
                                       function calls. Defaults to None.

    Returns:
        tuple: (x_coords, y_coords) numpy arrays of the generated points.
    """
    # make_s_curve generates points in 3D (X) and the manifold parameter (t)
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_state)

    # We'll use the 1st (index 0) and 3rd (index 2) dimensions for our 2D plot
    # X is shape (n_samples, 3)
    x_coords = X[:, 2]
    y_coords = X[:, 0]

    return np.vstack((x_coords, y_coords)).T


def generate_high_dim_gmm_samples(n_samples, n_features, n_components,
                                    covariance_type='full',
                                    means_range=(-5, 5),
                                    cov_scale_spherical=1.0,
                                    cov_scale_diag_max=1.0,
                                    reg_covar=1e-6,
                                    random_state=None):
    """
    Generates data samples from a randomly initialized high-dimensional
    Gaussian Mixture Model (GMM) and returns the samples along with the GMM parameters.

    Parameters
    ----------
    n_samples : int
        The total number of samples to generate.
    n_features : int
        The number of features (dimensionality) for each sample. This determines
        the "high-dimensional" aspect of the GMM.
    n_components : int
        The number of Gaussian mixture components.
    covariance_type : {'full', 'diag', 'spherical'}, default='full'
        The type of covariance parameters to use for the GMM components:
        - 'full': Each component has its own general covariance matrix.
                  These are generated using `sklearn.datasets.make_spd_matrix`.
        - 'diag': Each component has its own diagonal covariance matrix.
                  Diagonal elements are `random_values_in_[0, cov_scale_diag_max) + reg_covar`.
        - 'spherical': Each component has its own single variance (scaled identity matrix).
                       Variance is `random_value * cov_scale_spherical + reg_covar`.
    means_range : tuple (min, max), default=(-5, 5)
        The range [min, max) for randomly generating the mean vectors of the components.
        Each element of the mean vector is drawn uniformly from this range.
    cov_scale_spherical : float, default=1.0
        Scaling factor for the random part of the variance in 'spherical' covariance type.
    cov_scale_diag_max : float, default=1.0
        Upper bound for the random part of each diagonal element (variance) in 'diag'
        covariance type before regularization.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariances.
        This ensures that the covariance matrices are positive definite and well-conditioned,
        which is especially important in high dimensions.
    random_state : int, np.random.RandomState instance or None, default=None
        Determines random number generation for dataset creation.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated data samples.
    labels : ndarray of shape (n_samples,)
        The integer labels indicating the component membership of each sample.
    means : ndarray of shape (n_components, n_features)
        The mean vectors of each GMM component.
    covariances : ndarray
        The covariance matrices of each GMM component.
        - If covariance_type='full', shape is (n_components, n_features, n_features).
        - If covariance_type='diag', shape is (n_components, n_features, n_features).
        - If covariance_type='spherical', shape is (n_components, n_features, n_features).
    weights : ndarray of shape (n_components,)
        The mixing weights of each GMM component. These sum to 1.
    """
    if isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    elif random_state is None:
        rng = np.random.RandomState() # Fresh, non-reproducible state
    else: # Assuming it's already a RandomState object
        rng = random_state

    # 1. Generate GMM parameters (weights, means, covariances)
    # Generate weights: sum to 1
    # Using Dirichlet distribution for uniform sampling over the simplex
    weights = rng.dirichlet(np.ones(n_components))
    # Ensure weights sum to 1 precisely if dirichlet doesn't guarantee for float precision
    weights /= np.sum(weights)

    # Generate means: n_components mean vectors, each of n_features dimensions
    means = rng.uniform(means_range[0], means_range[1], size=(n_components, n_features))

    # Generate covariances: n_components covariance matrices
    covariances_list = []
    if covariance_type == 'full':
        for _ in range(n_components):
            # make_spd_matrix ensures symmetric positive definite.
            # The scale of its output depends on n_features (diagonal elements mean is n_features).
            cov = make_spd_matrix(n_features, random_state=rng)
            # Add regularization for numerical stability and to ensure positive definiteness
            cov += np.eye(n_features) * reg_covar
            covariances_list.append(cov)
    elif covariance_type == 'diag':
        for _ in range(n_components):
            # Variances for each feature: random values scaled + regularization
            diag_elements = rng.rand(n_features) * cov_scale_diag_max + reg_covar
            cov = np.diag(diag_elements)
            covariances_list.append(cov)
    elif covariance_type == 'spherical':
        for _ in range(n_components):
            # Single variance for all features: random value scaled + regularization
            variance = rng.rand() * cov_scale_spherical + reg_covar
            cov = np.eye(n_features) * variance
            covariances_list.append(cov)
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type}. "
                         "Must be one of 'full', 'diag', or 'spherical'.")
    covariances = np.array(covariances_list)

    # 2. Generate samples from the defined GMM
    # Assign each sample to a GMM component based on the weights
    component_choices = rng.choice(n_components, size=n_samples, p=weights)

    X = np.empty((n_samples, n_features))

    for i in range(n_samples):
        chosen_component_idx = component_choices[i]
        # Draw sample from the multivariate normal distribution of the chosen component
        X[i] = rng.multivariate_normal(
            mean=means[chosen_component_idx],
            cov=covariances[chosen_component_idx]
        )

    labels = component_choices # These are the true component labels for each sample

    return X, labels, means, covariances, weights



def generate_double_torus_samples(R1, R2, r_minor, N_twists, num_u_samples, num_v_samples, noise_std=0):
    """
    Generates samples (grid points) on the surface of a 3D double torus.
    (This function is the same as before)
    """
    u = np.linspace(0, 2 * np.pi, num_u_samples)
    v = np.linspace(0, 2 * np.pi, num_v_samples)
    u, v = np.meshgrid(u, v)

    x_coords = (R1 + r_minor * np.cos(v)) * np.cos(u) + R2 * np.cos(N_twists * u)
    y_coords = (R1 + r_minor * np.cos(v)) * np.sin(u) + R2 * np.sin(N_twists * u)
    z_coords = r_minor * np.sin(v)
    
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    z_flat = z_coords.ravel()
    
    if noise_std > 0:
        x_flat += np.random.normal(0, noise_std, x_flat.shape)
        y_flat += np.random.normal(0, noise_std, y_flat.shape)
        z_flat += np.random.normal(0, noise_std, z_flat.shape)

    samples = np.vstack((x_flat, y_flat, z_flat)).T
    return samples



def generate_cube_edge_samples(side_length, num_points_per_edge, center=None, noise_std=0):
    """
    Generates sample points on the edges of a 3D cube.

    Parameters:
    ----------
    side_length : float
        The length of a side of the cube. Must be positive.
    num_points_per_edge : int
        Number of points to sample on each edge.
        - If 1, samples the midpoint of each edge.
        - If >= 2, samples points including vertices (vertices will be duplicated).
        - If < 1, returns an empty array.
    center : array-like, optional
        (x, y, z) coordinates of the cube's center. Defaults to (0, 0, 0).

    Returns:
    -------
    numpy.ndarray
        An array of shape (12 * num_points_per_edge, 3) containing the
        (x, y, z) coordinates of the sample points, or an empty array
        if num_points_per_edge < 1.
    """
    if side_length <= 0:
        raise ValueError("side_length must be positive.")

    if num_points_per_edge < 1:
        return np.empty((0, 3))

    if center is None:
        center_arr = np.array([0.0, 0.0, 0.0])
    else:
        center_arr = np.asarray(center, dtype=float)
        if center_arr.shape != (3,):
            raise ValueError("center must be a 3-element array-like (x, y, z).")

    half_s = float(side_length) / 2.0

    # Define the min and max coordinates for each axis based on center and half_s
    min_coords = center_arr - half_s
    max_coords = center_arr + half_s
    
    # Convenient names for coordinates
    x0, x1 = min_coords[0], max_coords[0]
    y0, y1 = min_coords[1], max_coords[1]
    z0, z1 = min_coords[2], max_coords[2]

    all_edge_points = []

    if num_points_per_edge == 1:
        # Generate midpoints of the 12 edges
        # Edges parallel to X-axis (center_x, y_fixed, z_fixed)
        all_edge_points.append(np.array([[center_arr[0], y0, z0]]))
        all_edge_points.append(np.array([[center_arr[0], y1, z0]]))
        all_edge_points.append(np.array([[center_arr[0], y0, z1]]))
        all_edge_points.append(np.array([[center_arr[0], y1, z1]]))
        
        # Edges parallel to Y-axis (x_fixed, center_y, z_fixed)
        all_edge_points.append(np.array([[x0, center_arr[1], z0]]))
        all_edge_points.append(np.array([[x1, center_arr[1], z0]]))
        all_edge_points.append(np.array([[x0, center_arr[1], z1]]))
        all_edge_points.append(np.array([[x1, center_arr[1], z1]]))

        # Edges parallel to Z-axis (x_fixed, y_fixed, center_z)
        all_edge_points.append(np.array([[x0, y0, center_arr[2]]]))
        all_edge_points.append(np.array([[x1, y0, center_arr[2]]]))
        all_edge_points.append(np.array([[x0, y1, center_arr[2]]]))
        all_edge_points.append(np.array([[x1, y1, center_arr[2]]]))
    else: # num_points_per_edge >= 2
        # Edges parallel to X-axis
        fixed_y_z_pairs = [(y0, z0), (y1, z0), (y0, z1), (y1, z1)]
        for y_val, z_val in fixed_y_z_pairs:
            x_edge = np.linspace(x0, x1, num_points_per_edge)
            points = np.zeros((num_points_per_edge, 3))
            points[:, 0] = x_edge
            points[:, 1] = y_val
            points[:, 2] = z_val
            all_edge_points.append(points)

        # Edges parallel to Y-axis
        fixed_x_z_pairs = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
        for x_val, z_val in fixed_x_z_pairs:
            y_edge = np.linspace(y0, y1, num_points_per_edge)
            points = np.zeros((num_points_per_edge, 3))
            points[:, 0] = x_val
            points[:, 1] = y_edge
            points[:, 2] = z_val
            all_edge_points.append(points)

        # Edges parallel to Z-axis
        fixed_x_y_pairs = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for x_val, y_val in fixed_x_y_pairs:
            z_edge = np.linspace(z0, z1, num_points_per_edge)
            points = np.zeros((num_points_per_edge, 3))
            points[:, 0] = x_val
            points[:, 1] = y_val
            points[:, 2] = z_edge
            all_edge_points.append(points)
            
    if not all_edge_points:
        return np.empty((0,3)) # Should be covered by num_points_per_edge < 1
    
    samples = np.vstack(all_edge_points)
    
    if noise_std > 0:
        samples[:, 0] += np.random.normal(0, noise_std, samples.shape[0])
        samples[:, 1] += np.random.normal(0, noise_std, samples.shape[0])
        samples[:, 2] += np.random.normal(0, noise_std, samples.shape[0])


    return samples


def generate_arc_samples(start_angle_deg, arc_length_deg, num_samples, radius=1, noise=0.1, random_state=None):
    """
    Generates points along a circular arc.
    """
    # Convert degrees to radians
    start_rad = np.radians(start_angle_deg)
    end_rad = np.radians(start_angle_deg + arc_length_deg)
    
    # Generate linearly spaced angles
    angles = np.linspace(start_rad, end_rad, num_samples)
    
    # Calculate x and y coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    np.random.seed(random_state)
    x_noise = np.random.normal(0, noise**2, num_samples)
    y_noise = np.random.normal(0, noise**2, num_samples)
    
    return x + x_noise, y + y_noise


def generate_exp_mixture(n_samples, components):
    """
    Generates samples and returns their total probability under the mixture model.
    """
    total_w = sum(c['weight'] for c in components)
    weights = np.array([c['weight'] / total_w for c in components])
    
    # 1. Generate Samples
    comp_indices = np.random.choice(len(components), size=n_samples, p=weights)
    samples = np.zeros(n_samples)
    
    for i, comp in enumerate(components):
        mask = (comp_indices == i)
        n_comp = np.sum(mask)
        if n_comp == 0: continue
        
        # Shifted exponential
        s = expon.rvs(loc=comp['loc'], scale=comp['scale'], size=n_comp)
        if comp.get('type') == 'left':
            s = comp['loc'] - (s - comp['loc'])
        samples[mask] = s

    # 2. Calculate Probabilities (PDF)
    # We sum the PDF of each component weighted by its mixture weight
    probs = np.zeros(n_samples)
    for i, comp in enumerate(components):
        loc, scale = comp['loc'], comp['scale']
        
        if comp.get('type') == 'left':
            # For reflected: PDF(x) = f_std(loc - (x - loc)) = f_std(2*loc - x)
            # Simplified: expon.pdf handles the tail; we just reflect the input x
            component_pdf = expon.pdf(loc - (samples - loc), loc=loc, scale=scale)
        else:
            component_pdf = expon.pdf(samples, loc=loc, scale=scale)
            
        probs += weights[i] * component_pdf
        
    return samples, probs




def generate_shape_with_exp_density(n_samples=50000, exp_scale=0.05, noise=0.05):
    a = 2.0 
    
    t = np.linspace(0, 2 * np.pi, n_samples)
    
    # Deltoid parametric equations
    x_base = 2 * a * np.cos(t) + a * np.cos(2 * t)
    y_base = 2 * a * np.sin(t) - a * np.sin(2 * t)

    # 2. Apply Exponential Density Function
    noise_angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.exponential(scale=exp_scale, size=n_samples)

    x = x_base + radii * np.cos(noise_angles)
    y = y_base + radii * np.sin(noise_angles)

    return x, y




def compute_density_component(X, Y, a, b1, b2, c1, c2, d1, d2, p=1, q=1):
    """Calculates the density for a single asymmetric exponential component."""
    x_diff = X - b1
    term_x = np.where(x_diff >= 0, (c1 * x_diff)**p, (d1 * (-x_diff))**q)
    
    y_diff = Y - b2
    term_y = np.where(y_diff >= 0, (c2 * y_diff)**p, (d2 * (-y_diff))**q)
    
    return a * np.exp(-(term_x + term_y))

def sun_mixture_density(X, Y):
    """Combines all components to form the 'Sun' shape."""
    density = np.zeros_like(X)
    
    # 1. Core
    # density = samples_gauss_2d  # Add some noise to the core for a more natural look
    # 2. Main Rays (N, S, E, W)
    density += compute_density_component(X, Y, a=3, b1=1.5, b2=0, c1=.35, c2=10, d1=10, d2=10)
    density += compute_density_component(X, Y, a=3, b1=-1.5, b2=0, c1=10, c2=10, d1=0.35, d2=10)
    density += compute_density_component(X, Y, a=3, b1=0, b2=1.5, c1=10, c2=0.35, d1=10, d2=10)
    density += compute_density_component(X, Y, a=3, b1=0, b2=-1.5, c1=10, c2=10, d1=10, d2=0.35)
    # 3. Diagonal Rays
    # density += compute_density_component(X, Y, a=2.5, b1=1.0, b2=1.0, c1=10, c2=10, d1=1, d2=10)
    # density += compute_density_component(X, Y, a=2.5, b1=-1.0, b2=1.0, c1=6, c2=1, d1=1, d2=6)
    # density += compute_density_component(X, Y, a=2.5, b1=1.0, b2=-1.0, c1=1, c2=6, d1=6, d2=1)
    # density += compute_density_component(X, Y, a=2.5, b1=-1.0, b2=-1.0, c1=6, c2=6, d1=1, d2=1)
    
    return density


def generate_point_cloud(num_points=50000):
    """Uses Rejection Sampling to generate points based on the density function."""
    # Define the bounding box to throw darts in
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    
    # M is the estimated maximum density value. 
    # The core's 'a' is 5, plus overlap from rays, so 6.0 is a safe ceiling.
    M = 6.0 
    
    accepted_x = []
    accepted_y = []
    
    # Generate points in large batches for speed
    batch_size = num_points * 5 
    
    while len(accepted_x) < num_points:
        # 1. Generate random coordinates (X, Y)
        x_rand = np.random.uniform(x_min, x_max, batch_size)
        y_rand = np.random.uniform(y_min, y_max, batch_size)
        
        # 2. Calculate the actual equation density at these coordinates
        densities = sun_mixture_density(x_rand, y_rand)
        
        # 3. Generate a random height (U) for each point
        u = np.random.uniform(0, M, batch_size)
        
        # 4. Accept the point ONLY if U is less than the density
        # (This means higher density areas have a much higher chance of keeping their points)
        mask = u < densities
        
        accepted_x.extend(x_rand[mask])
        accepted_y.extend(y_rand[mask])
    
    x = np.array(accepted_x[:num_points])
    y = np.array(accepted_y[:num_points])
    return x, y