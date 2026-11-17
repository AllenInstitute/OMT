import torch
from torch.distributions import MultivariateNormal, Categorical



def random_mus_covs(n, dim):
    
    # torch.manual_seed(seed)  # Set the random seed correctly
    mus = (20 * torch.rand(n, dim) - 10).tolist()
    covs = [(torch.eye(dim) * torch.randn(dim).abs()).tolist() for _ in range(n)]
    return mus, covs
    

class GMMSamplerComponent:
    """
    A Gaussian Mixture Model (GMM) sampler component.
    Calculates theoretical global mean and covariance of the mixture.
    """
    def __init__(self, means, covariances, dim, num_components):
        """
        Initializes the GMM sampler component.

        Args:
            means (list, np.array, or torch.Tensor): Means of the GMM components.
                Shape: (num_components, dim).
            covariances (list, np.array, or torch.Tensor): Covariance matrices of the GMM components.
                Shape: (num_components, dim, dim).
            dim (int): Dimensionality of the GMM.
            num_components (int): Number of components in the GMM.
        """
        self.dim = dim
        self.num_components = num_components

        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means, dtype=torch.float32)
        if not isinstance(covariances, torch.Tensor):
            covariances = torch.tensor(covariances, dtype=torch.float32)

        if means.shape != (num_components, dim):
            raise ValueError(f"Means shape must be ({num_components}, {dim}), got {means.shape}")
        if covariances.shape != (num_components, dim, dim):
            raise ValueError(f"Covariances shape must be ({num_components}, {dim}, {dim}), got {covariances.shape}")

        self.means = means
        self.covariances = covariances

        try:
            self.component_distributions = [
                MultivariateNormal(self.means[i], self.covariances[i])
                for i in range(self.num_components)
            ]
        except RuntimeError as e:
            # This can happen if covariance matrices are not positive definite.
            raise ValueError(
                "Failed to create MultivariateNormal distributions. "
                "Ensure covariance matrices are positive definite. "
                f"Error: {e}"
            )

        # Assume equal weights for components for sampling and global parameter calculation
        self.component_weights = torch.ones(self.num_components, dtype=torch.float32) / self.num_components
        # Categorical distribution for choosing components
        self.categorical_dist = Categorical(self.component_weights)

        # Calculate theoretical global mean and covariance
        self._calculate_global_parameters()

    def _calculate_global_parameters(self):
        """
        Calculates the theoretical global mean and covariance of the GMM.
        Assumes equal component weights.
        Global Mean (E[X]): mu_global = sum(w_i * mu_i)
        Global Covariance (Cov(X)): Sigma_global = sum(w_i * (Sigma_i + mu_i @ mu_i^T)) - mu_global @ mu_global^T
        """
        # Global Mean: E[X] = sum(w_i * mu_i)
        # self.component_weights is (num_components), self.means is (num_components, dim)
        # unsqueeze component_weights to (num_components, 1) for broadcasting
        self.global_mean = torch.sum(self.component_weights.unsqueeze(1) * self.means, dim=0)

        # Global Covariance: E[XX^T] - E[X]E[X]^T
        # E[XX^T] = sum(w_i * (Sigma_i + mu_i @ mu_i^T))
        e_xx_t = torch.zeros((self.dim, self.dim), dtype=torch.float32)
        for i in range(self.num_components):
            mean_i = self.means[i].unsqueeze(1)  # Shape: (dim, 1)
            cov_i = self.covariances[i]          # Shape: (dim, dim)
            # self.component_weights[i] is a scalar
            e_xx_t += self.component_weights[i] * (cov_i + mean_i @ mean_i.T)

        # Outer product of global_mean: global_mean (dim,) -> (dim,1) @ (1,dim) -> (dim,dim)
        global_mean_outer = self.global_mean.unsqueeze(1) @ self.global_mean.unsqueeze(0)
        self.global_cov = e_xx_t - global_mean_outer

    def sample(self, num_smp):
        """
        Generates samples from the GMM.

        Args:
            num_smp (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples, shape (num_smp, dim).
        """
        if not isinstance(num_smp, int) or num_smp <= 0:
            raise ValueError("Number of samples (num_smp) must be a positive integer.")

        # Sample component indices: shape (num_smp,)
        # These indices determine which Gaussian component each sample will come from.
        component_indices = self.categorical_dist.sample((num_smp,))

        # Prepare tensor for samples
        samples = torch.empty((num_smp, self.dim), dtype=torch.float32)

        # Sample from the chosen components
        for i in range(self.num_components):
            # Get a boolean mask for samples belonging to component i
            mask = (component_indices == i)
            # Count how many samples are to be drawn from this component
            num_component_samples = mask.sum().item()

            if num_component_samples > 0:
                # Sample from the i-th component's distribution
                samples[mask] = self.component_distributions[i].sample((num_component_samples,))
        return samples

class GMMDistributionSamplers:
    """
    A class that provides access to input and output GMM samplers.
    The input sampler is a GMM with 3 components.
    The output sampler is a GMM with 10 components.
    """
    def __init__(self, dim,
                 input_means, input_covariances,
                 output_means, output_covariances,
                 random_seed):
        """
        Initializes the GMMDistributionSamplers.

        Args:
            dim (int): Dimensionality of the GMMs (same for input and output).
            input_means (list, np.array, or torch.Tensor): Means for the 3 components of the input GMM.
                Expected shape: (3, dim).
            input_covariances (list, np.array, or torch.Tensor): Covariances for the 3 components of the input GMM.
                Expected shape: (3, dim, dim).
            output_means (list, np.array, or torch.Tensor): Means for the 10 components of the output GMM.
                Expected shape: (10, dim).
            output_covariances (list, np.array, or torch.Tensor): Covariances for the 10 components of the output GMM.
                Expected shape: (10, dim, dim).
            random_seed (int): Random seed for reproducibility of sampling.
        """
        self.dim = dim
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed) # Set global seed for PyTorch sampling operations

        num_input_components = len(input_means)
        num_output_components = len(output_means)

        self._input_gmm_component = GMMSamplerComponent(
            means=input_means,
            covariances=input_covariances,
            dim=dim,
            num_components=num_input_components
        )

        self._output_gmm_component = GMMSamplerComponent(
            means=output_means,
            covariances=output_covariances,
            dim=dim,
            num_components=num_output_components
        )

        # Create callable sampler interfaces that also expose GMM properties
        self.input_sampler = self._create_sampler_interface(self._input_gmm_component)
        self.output_sampler = self._create_sampler_interface(self._output_gmm_component)

    def _create_sampler_interface(self, gmm_component):
        """
        Helper method to create a sampler interface object.
        This interface object is callable (for sampling) and exposes GMM properties.
        """
        class SamplerInterface:
            def __init__(self, component):
                self._component = component

            def __call__(self, num_smp):
                """Generates samples from the GMM component."""
                return self._component.sample(num_smp)

            @property
            def means(self):
                """Component means of the GMM."""
                return self._component.means

            @property
            def covariances(self):
                """Component covariances of the GMM."""
                return self._component.covariances

            @property
            def global_mean(self):
                """Theoretical global mean of the GMM."""
                return self._component.global_mean

            @property
            def global_cov(self):
                """Theoretical global covariance of the GMM."""
                return self._component.global_cov

        return SamplerInterface(gmm_component)
