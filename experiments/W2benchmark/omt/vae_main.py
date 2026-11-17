import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pdb
import pandas as pd
from scipy.spatial.distance import cdist

from .networks import shallow_vae, deep_vae
from .solver import omt
from .functions import points_transport, dynamic_points_transport


class VAE:

    def __init__(self, saving_folder='', device=None, eps=1e-8, saving_flag=True):

        self.eps = eps
        self.save = saving_flag
        self.folder = saving_folder

        self.device = device


    def loss_function(self, x, recon_x, mu, log_var, beta=1, mode='MSE'):

        if self.variational:
            if mode == 'MSE':
                l_rec = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0))
            
            elif mode == 'MSE-BCE':
                l_rec = F.mse_loss(recon_x, x, reduction='sum') / (x.size(0))
                rec_bin = torch.where(recon_x > 0.01, 1., 0.)
                x_bin = torch.where(x > 0.01, 1., 0.)
                l_rec += 0.1 * F.binary_cross_entropy(rec_bin, x_bin)
            
            kld = (-0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=0)).sum()
            loss = l_rec + beta * kld
        else:
            kld = 0.
            if mode == 'MSE':
                l_rec = F.mse_loss(recon_x, x, reduction='mean') 
            
            elif mode == 'MSE-BCE':
                l_rec = F.mse_loss(recon_x, x, reduction='mean') 
                rec_bin = torch.where(recon_x > 0.01, 1., 0.)
                x_bin = torch.where(x > 0.01, 1., 0.)
                l_rec += 0.1 * F.binary_cross_entropy(rec_bin, x_bin)
                
            loss = l_rec

        return loss, l_rec, kld
    
    
    def init_nn(self, input_dim, network, fc_dim=100, lowD_dim=2, n_layer=2, x_drop=0., variational=True, momentum=.01, trained_model=''):
        """
        Initialized the deep mixture model and its optimizer.

        input args
            input_dim: dimension of the input data.
            network: type of the network, either 'shallow' or 'deep'.
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            lr: the learning rate of the optimizer, here Adam.
            beta: regularizer for the KL divergence term.
            variational: if True, the model is a variational autoencoder, otherwise it is a deterministic autoencoder.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            trained_model: a pre-trained model, in case you want to initialized the network with a pre-trained network.
        """
        self.lowD_dim = lowD_dim
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.variational = variational
        self.network = network

        if network == 'shallow':
            self.model = shallow_vae(
                                    input_dim=self.input_dim, 
                                    fc_dim=fc_dim, 
                                    lowD_dim=lowD_dim, 
                                    x_drop=x_drop, 
                                    n_layer=n_layer,
                                    device=self.device, 
                                    eps=self.eps,
                                    variational=variational,
                                    momentum=momentum,
                                    )
        elif network == 'deep':
            self.model = deep_vae(
                                    input_dim=self.input_dim, 
                                    fc_dim=fc_dim, 
                                    lowD_dim=lowD_dim, 
                                    x_drop=x_drop, 
                                    n_layer=n_layer,
                                    device=self.device, 
                                    eps=self.eps,
                                    variational=variational,
                                    momentum=momentum,
                                    )

        self.optimizer = torch.optim.Adam(self.model.parameters())
        

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()


    def train(self, train_loader, validation_loader, n_epoch, lr, beta=1, mode='MSE', wandb_run=None):
        """
        train the VAE with the pre-defined parameters/settings

        input args
            train_loader: train dataloader (from data_loader.py)
            validation_loader: validation dataloader (from data_loader.py)
            n_epoch: number of training epoch, without pruning
            mode: the loss function, either 'MSE' or 'MSE-BCE'
            n_workers: number of workers
            label: any metadata such as class label, sample index, color, etc.

        return
            self.model: VAE model
            trained_model: the trained model file
        """
        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        train_loss_rec = np.zeros(n_epoch)
        train_loss_KL = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        validation_loss_rec = np.zeros(n_epoch)
        # self.model = nn.DataParallel(self.model, device_ids=[0]) # parallelize the model
        self.model = self.model.to(self.device)
        
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        
        print("Start training ...")
        for epoch in range(n_epoch):
            train_loss_ = 0.
            train_loss_rec_ = 0.
            train_loss_KL_ = 0.
            t0 = time.time()
            self.model.train()

            for batch_indx, data in enumerate(train_loader):
                data = (data.squeeze(dim=1)).to(self.device)
                self.optimizer.zero_grad()
                recon_x, z , mu, log_var = self.model(data)
                loss, l_rec, l_KL = self.loss_function(x=data, recon_x=recon_x, mu=mu, log_var=log_var, mode=mode, beta=beta)
                loss.backward()
                self.optimizer.step()
                train_loss_ += loss.data.item()
 
                if self.variational:
                    train_loss_KL_ += l_KL.data.item() 
                    train_loss_rec_ += l_rec.data.item() / self.input_dim
                else:
                    train_loss_KL_ = 0.
                    train_loss_rec_ += l_rec.data.item() 

            if epoch % 100 == 0 and epoch > 0:
                trained_model = self.folder + f'/model/VAE_model_epoch_{epoch}_{self.current_time}.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)

            train_loss[epoch] = train_loss_ / (batch_indx + 1)
            train_loss_rec[epoch] = train_loss_rec_ / (batch_indx + 1)
            train_loss_KL[epoch] = train_loss_KL_ / (batch_indx + 1)
            
            if wandb_run:
                    wandb_run.log(
                                {
                                "train/recon-loss": train_loss_rec[epoch],
                                "train/time": time.time() - t0,
                                }
                                )

            print('---> Epoch:{}, Total Loss: {:.4f}, Rec. Loss: {'':.4f}, KLD: {:.4f}, Elapsed Time:{:.2f}'.format(epoch, train_loss[epoch], train_loss_rec[epoch], train_loss_KL[epoch], time.time() - t0))

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss_rec = 0.
                val_loss = 0.
                for batch_indx, data in enumerate(validation_loader):
                    data = data.squeeze(dim=1).to(self.device)
                    recon_x, z, mu, log_var = self.model(data)
                    loss, l_rec, _ = self.loss_function(x=data, recon_x=recon_x, mu=mu, log_var=log_var, mode=mode, beta=beta)
                    val_loss += loss.data.item()
                    val_loss_rec += l_rec.data.item() #/ self.input_dim

            validation_loss[epoch] = val_loss / (batch_indx + 1)
            validation_loss_rec[epoch] = val_loss_rec / (batch_indx + 1)
            
            if wandb_run:
                wandb_run.log(
                            {
                            "validation/recon-loss": validation_loss[epoch],
                            }
                            )

            print('---> Validation Total Loss: {:.4f}, Rec. Loss: {'':.4f}'.format(validation_loss[epoch], validation_loss_rec[epoch]))

        if self.save and n_epoch > 0:
            trained_model = self.folder + f'/model/VAE_model_{self.current_time}.pth'
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
        
        return self.model, trained_model
    
    @torch.no_grad()
    def test(self, data_loader, mode='MSE', verbose=True):
        """
        Test the VAE model.

        input args
            data_loader: data_loader (from data_loader.py).
            mode: the loss function, either 'MSE' or 'BCE'.
            verbose: if True, print the loss values.

        return
            z: latent representation of the input data.
            d_indx: the index of the input data.
        """

        x_low_smp = []
        x_low = []
        sample_id = []
        test_loss_ = 0.
        test_loss_rec_ = 0.

        for batch_indx, data in enumerate(data_loader):
            data = data.to(self.device)
            recon_x, z, mu, log_var = self.model(data)
            loss, l_rec, _ = self.loss_function(x=data, recon_x=recon_x, mu=mu, log_var=log_var, mode=mode, beta=1)
            test_loss_ += loss.data.item()
            test_loss_rec_ += l_rec.data.item() / self.input_dim
            if self.variational:
                x_low_smp.append(z.cpu().numpy())
                x_low.append(mu.cpu().numpy())
            else:
                x_low.append(z.cpu().numpy())

        test_loss = test_loss_ / (batch_indx + 1)
        test_loss_rec = test_loss_rec_ / (batch_indx + 1)

        if verbose:
            print('---> Total Loss: {:.4f}, Rec. Loss: {'':.4f}'.format(test_loss, test_loss_rec))
        
        if self.variational:
            z = np.concatenate(x_low_smp, axis=0)
            x_lowD = np.concatenate(x_low, axis=0)
        else:
            z = np.concatenate(x_low, axis=0)
            x_lowD = z
            
        return z, x_lowD, sample_id


    @torch.no_grad()
    def get_latent(self, data, variational=False):
        
        # covert to torch tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        train_loader = DataLoader(
                                data,
                                batch_size=512,
                                shuffle=False,
                                drop_last=False,
                                )
        latent_z = []
        self.model.eval()
        for _, x, in enumerate(train_loader):
            x = x.to(self.device)
            _, z, mu, log_var = self.model(x)
            if variational:
                latent_z.append(mu)
            else:
                if len(mu) > 0:
                    latent_z.append(mu)
                else:
                    latent_z.append(z)

        return torch.cat(latent_z, 0).cpu().detach().numpy()


    @torch.no_grad()
    def generate(self, z, batch_size=128):
        recon_x = []
        z = torch.tensor(z, dtype=torch.float32)
        dataset = TensorDataset(z)
        data_loader = DataLoader(
                                dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True, 
                                )
    
        for batch in data_loader:
            # pdb.set_trace()
            if isinstance(batch, list):
                batch = batch[0]
                
            x = self.model.decoder(batch.to(self.device))
            recon_x.append(x.cpu().detach().numpy())
            
        return np.concatenate(recon_x, axis=0)
    
    
    def annotation(self, x, label, x_t, gmm_solver):
        """
        Annotate the transported points with the labels of the target points.

        input args
            z_t: latent representation of the target points.
            label_t: labels of the target points.
            z_t_trans: transported points from the source to the target.
        """

        cmp = gmm_solver.predict(x)
        label_tt = np.full(x_t.shape[0], '', dtype=object)
        
        for i, x_t_i in enumerate(x_t):
            cmp_t = gmm_solver.predict(x_t_i.reshape(1, -1))
            idx = np.where(cmp == cmp_t[0])[0]
            if len(idx) > 0:
                x_ = x[idx]
                #find the most close sample in the source points
                dist = np.linalg.norm(x_ - x_t_i, axis=1)
                min_idx = np.argmin(dist)
                label_tt[i] = label[idx[min_idx]]
            

        # for i, l_t in enumerate(unique_labels):
        #     z_t_l = x[label == l_t]
        #     cmp_t = gmm_solver.predict(z_t_l)
        #     for cp in np.unique(cmp_t):
        #         if sum(cmp_tt == cp) > 0:
        #             label_tt[cmp_tt == cp] = l_t
            
        return label_tt
   
    
    def annotation_2(self, x, label, x_t, gmm_solver):
        """
        Annotate the transported points with the labels of the target points.

        input args
            z_t: latent representation of the target points.
            label_t: labels of the target points.
            z_t_trans: transported points from the source to the target.
        """

        cmp = gmm_solver.predict(x)
        cmp_t_all = gmm_solver.predict(x_t)
        label_tt = np.full(x_t.shape[0], '', dtype=object)
        
        for k in range(gmm_solver.n_components):
            # Find the indices of all source and target points belonging to component k
            source_idx_k = np.where(cmp == k)[0]
            target_idx_k = np.where(cmp_t_all == k)[0]

            # Skip if there are no points in this component to match
            if len(target_idx_k) == 0 or len(source_idx_k) == 0:
                continue

            # Get the actual data points and labels for this component
            source_points_k = x[source_idx_k]
            target_points_k = x_t[target_idx_k]
            source_labels_k = label[source_idx_k]

            # 3. Calculate distances between all target points and all source points
            #    in this component in a single, highly optimized operation.
            #    `dist_matrix` will have shape (n_targets_in_k, n_sources_in_k)
            dist_matrix = cdist(target_points_k, source_points_k)

            # 4. Find the index of the closest source point for EACH target point
            #    `np.argmin` along axis=1 finds the column index with the minimum distance for each row.
            closest_source_indices_in_k = np.argmin(dist_matrix, axis=1)

            # 5. Get the corresponding labels of these closest source points
            matched_labels = source_labels_k[closest_source_indices_in_k]

            # 6. Assign the matched labels to the correct positions in the final output array
            label_tt[target_idx_k] = matched_labels
            
        return label_tt
            
     
        
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)