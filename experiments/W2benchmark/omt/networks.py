import torch
import torch.nn as nn
from torch.nn import ModuleList as mdl
from torch.nn import Sequential as seq
import torch.nn.functional as F
import pdb


class shallow_vae(nn.Module):
    """
    Class for the neural network module for variational inference. 
    The module contains an VAE with a continuouse latent representation using reparameterization trick.
    The default setting of this network is for normalized scRNA-seq datasets. If you want to use another 
    data modality, you may need to modify the network's parameters/architecture.

    Methods
        encoder: encoder network.
        decoder: decoder network.
        forward: module for forward path.
        reparam_trick: module for reparameterization.
        loss: ELBO loss function module
    """
    def __init__(self, input_dim, fc_dim, lowD_dim, n_layer, x_drop, variational, device, eps, momentum):
        """
        Class instantiation.

        input args
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            n_layer: number of hidden layers in the encoder and decoder networks, at least one.
            x_drop: dropout probability at the first (input) layer.
            variational: a boolean variable, True uses sampling, False is just a regular AE.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
        """
        super(shallow_vae, self).__init__()
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.x_dp = nn.Dropout(x_drop)
        self.variational = variational
        self.eps = eps
        self.momentum = momentum
        self.device = device

        self.sigmoid = nn.Sigmoid()
        
        self.fc_mu = nn.Linear(fc_dim, lowD_dim)
        self.fc_sigma = nn.Linear(fc_dim, lowD_dim)

        self.layer_e = [nn.Linear(input_dim, fc_dim), nn.ReLU()]
        for i in range(n_layer - 1):
            self.layer_e.extend([nn.Linear(fc_dim, fc_dim), nn.ReLU()])
        self.layer_e = seq(*self.layer_e)
    
        self.layer_d = [nn.Linear(lowD_dim, fc_dim), nn.ReLU()]
        for i in range(n_layer - 1):
            self.layer_d.extend([nn.Linear(fc_dim, fc_dim), nn.ReLU()])
            
        self.layer_d.extend([nn.Linear(fc_dim, input_dim), nn.ReLU()])
        self.layer_d = seq(*self.layer_d)

        self.batchnorm = nn.BatchNorm1d(num_features=lowD_dim, eps=eps, momentum=momentum, affine=False)


    def encoder(self, x) -> torch.Tensor:
        h = self.layer_e(x)
        if self.variational:
            return self.fc_mu(h), self.sigmoid(self.fc_sigma(h))
        else:
            return self.batchnorm(self.fc_mu(h))


    def decoder(self, z) -> torch.Tensor:
        return self.layer_d(z)


    def forward(self, x) -> torch.Tensor:
        """
        input args
            x: input data.

        return
            recon_x: reconstructed data.
            z: latent representation following Gaussian distribution.
            mu: mean of the latent representation.
            var: variance of the latent representation.
        """
        mu, log_var = [], []
        if self.variational:
            mu, var = self.encoder(x)
            log_var = torch.log(var + self.eps)
            z = self.reparam_trick(mu, var)
        else:
            z = self.encoder(x)

        recon_x = self.decoder(z)

        return recon_x, z, mu, log_var


    def reparam_trick(self, mu, sigma):
        """
        Generate samples from a normal distribution for reparametrization trick.

        input args
            mu: mean of the Gaussian distribution for q(z|x) = N(mu, sigma^2*I).
            sigma: variance of the Gaussian distribution for q(s|z,x) = N(mu, sigma^2*I).

        return
            a sample from Gaussian distribution N(mu, sigma^2*I).
        """
        std = sigma.sqrt()
        eps = torch.rand_like(std).to(self.device)
        return eps.mul(std).add(mu)



class deep_vae(nn.Module):
    """
    Class for the neural network module for variational inference. 
    The module contains an VAE with a continuouse latent representation using reparameterization trick.
    The default setting of this network is for normazlied scRNA-seq datasets. If you want to use another 
    data modality, you may need to modify the network's parameters/architecture.

    Methods
        encoder: encoder network.
        decoder: decoder network.
        forward: module for forward path.
        reparam_trick: module for reparameterization.
        loss: ELBO loss function module
    """
    def __init__(self, input_dim, fc_dim, lowD_dim, n_layer, x_drop, variational, device, eps, momentum):
        """
        Class instantiation.

        input args
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            n_layer: number of hidden layers in the encoder and decoder networks, at least one.
            x_drop: dropout probability at the first (input) layer.
            variational: a boolean variable, True uses sampling, False is just a regular AE.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
        """
        super(deep_vae, self).__init__()
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.x_dp = nn.Dropout(x_drop)
        self.variational = variational
        self.eps = eps
        self.momentum = momentum
        self.device = device

        self.sigmoid = nn.Sigmoid()
        
        self.fc_mu = nn.Linear(fc_dim, lowD_dim)
        self.fc_sigma = nn.Linear(fc_dim, lowD_dim)

        self.layer_e = [nn.Linear(input_dim, 1000), nn.ReLU()]
        self.layer_e.extend([nn.Linear(1000, fc_dim), nn.ReLU()])
        
        for i in range(n_layer - 2):
            self.layer_e.extend([nn.Linear(fc_dim, fc_dim), nn.ReLU()])
        self.layer_e = seq(*self.layer_e)
    
        self.layer_d = [nn.Linear(lowD_dim, fc_dim), nn.ReLU()]
        for i in range(n_layer - 2):
            self.layer_d.extend([nn.Linear(fc_dim, fc_dim), nn.ReLU()])
            
        self.layer_d.extend([nn.Linear(fc_dim, 1000), nn.ReLU()])
        self.layer_d.extend([nn.Linear(1000, input_dim), nn.ReLU()])
        self.layer_d = seq(*self.layer_d)

        self.batchnorm = nn.BatchNorm1d(num_features=lowD_dim, eps=eps, momentum=momentum, affine=False)


    def encoder(self, x) -> torch.Tensor:
        h = self.layer_e(self.x_dp(x))
        if self.variational:
            return self.fc_mu(h), self.sigmoid(self.fc_sigma(h))
        else:
            return self.batchnorm(self.fc_mu(h))


    def decoder(self, z) -> torch.Tensor:
        return self.layer_d(z)


    def forward(self, x) -> torch.Tensor:
        """
        input args
            x: input data.

        return
            recon_x: reconstructed data.
            z: latent representation following Gaussian distribution.
            mu: mean of the latent representation.
            var: variance of the latent representation.
        """
        mu, log_var = [], []
        if self.variational:
            mu, var = self.encoder(x)
            log_var = torch.log(var + self.eps)
            z = self.reparam_trick(mu, var)
        else:
            z = self.encoder(x)

        recon_x = self.decoder(z)

        return recon_x, z, mu, log_var


    def reparam_trick(self, mu, sigma):
        """
        Generate samples from a normal distribution for reparametrization trick.

        input args
            mu: mean of the Gaussian distribution for q(z|x) = N(mu, sigma^2*I).
            sigma: variance of the Gaussian distribution for q(s|z,x) = N(mu, sigma^2*I).

        return
            a sample from Gaussian distribution N(mu, sigma^2*I).
        """
        std = sigma.sqrt()
        eps = torch.rand_like(std).to(self.device)
        return eps.mul(std).add(mu)
