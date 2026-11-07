import numpy as np
import pandas as pd
import sklearn.datasets
import os, sys, time
import torch
import boto3
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


bucket_name = 'pga-genomics-wg-802451596237-us-west-2'
filehandle_hrot = './'
filehandle_nbs = './Datasets/MERFISH/vizgen/'

# The MERFISH CSVs are read from the local Datasets/ copy when present; the S3
# session is only created (lazily) if a file is missing locally.
s3 = None
def _load_csv_bytes(name):
    local_path = os.path.join(filehandle_nbs, name)
    if os.path.exists(local_path):
        with open(local_path, "rb") as fh:
            return BytesIO(fh.read())
    global s3
    if s3 is None:
        session = boto3.Session(profile_name="802451596237_StorageWorkgroupRW")
        s3 = session.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=filehandle_nbs + name)
    return BytesIO(obj['Body'].read())

sys.path.insert(0, filehandle_hrot)
sys.path.insert(0, filehandle_hrot + '/omt/')
sys.path.insert(0, filehandle_hrot + '/src/')

import importlib
import FRLC
from FRLC import FRLC_opt
import HR_OT
importlib.reload(HR_OT)
import rank_annealing as ra
import util_merfish as um
from functions import sample_gmm_kdim, mmd_rbf, points_transport
from solver import omt, ot_w2, ot_sinkh, shallow_gmm, vectorized_eogt_dist
from omt_vis_tools import plot_gmm_ellipses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'On device: {device}')
dtype = torch.float32

um.set_seed(44)

# contains spatial features
metadata_1 = 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv'
metadata_2 = 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv'

# contains expression data
gene_1 = 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_by_gene_S2R3.csv'
gene_2 = 'datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_by_gene_S2R2.csv'

fname1 = _load_csv_bytes(metadata_1)
fname2 = _load_csv_bytes(metadata_2)
X_fname1 = _load_csv_bytes(gene_1)
X_fname2 = _load_csv_bytes(gene_2)


# loading "Slice 2, Replicate 3":
_, _, df1, _, _, xI_c, yI_c, _, _ = um.plot_cell_positions(fname1)

# loading "Slice 2, Replicate 2":
_, _, df2, _, _, _, _, xJ_c_L, yJ_c_L = um.plot_cell_positions(fname2,
                                                               color='blue',
                                                               theta_deg=-45.0)

plt.close('all')

S1 = np.column_stack((xI_c, yI_c))
S2 = np.column_stack((xJ_c_L, yJ_c_L))
print(f"Number of cells in first dataset: {S1.shape[0]}")
print(f"Number of cells in second dataset: {S2.shape[0]}")

max_S1 = np.max(S1)
max_S2 = np.max(S2)
S1_norm = S1 / max_S1
S2_norm = S2 / max_S2

for Ks, Kt in [(5, 10), (35, 50), (80, 100), (250, 200), (500, 500), (800, 800), (1000, 1000), (2000, 2000)]:
    eps = 0.001
    max_iter = int(1e6)
    stop_thr = 1e-6
    cov_type = 'full'

    omt_w, omt_mu, omt_cov, solver_dict = omt(
                                            data_source=S1_norm, 
                                            data_target=S2_norm, 
                                            n_components=(Ks, Kt), 
                                            eps_gmm=eps, 
                                            eps_w=eps, 
                                            method='sinkhorn', 
                                            cov_type=cov_type,
                                            reg_covar=1e-6,
                                            max_iter=max_iter,
                                            stop_thr=stop_thr,
                                            verbose=False,
                                        )

    mean_s = solver_dict['gmm_s'].means_
    mean_t = solver_dict['gmm_t'].means_
    sigma_s = solver_dict['gmm_s'].covariances_
    sigma_t = solver_dict['gmm_t'].covariances_
    
    
    n_iter = solver_dict['n_iter']
    total_iter = solver_dict['n_total_iter']
    time_elapsed = solver_dict['time']

    if not isinstance(eps, (list, tuple, np.ndarray)):
        eps_gmm = np.full(Ks * Kt, eps)
    else:
        eps_gmm = np.asarray(eps).flatten()

    mean_s_vec = np.max(S1) * mean_s[:, np.newaxis, :]       # Shape: (K_s, 1, d)
    mean_t_vec = np.max(S2) * mean_t[np.newaxis, :, :]       # Shape: (1, K_t, d)

    sigma_s_vec = (np.max(S1)**2) * sigma_s[:, np.newaxis, ...]   # Shape: (K_s, 1, d, d)
    sigma_t_vec = (np.max(S2)**2) * sigma_t[np.newaxis, :, ...]   # Shape: (1, K_t, d, d)

    eps_gmm_mat = eps_gmm.reshape(Ks, Kt)     # Shape: (K_s, K_t)

    n = 84172

    C, _, _ = vectorized_eogt_dist(mean_s_vec, mean_t_vec, sigma_s_vec, sigma_t_vec, eps_gmm_mat)
    total_cost = np.sum(omt_w * C) / n

    print(f"Total transport cost between the two datasets: {total_cost}")


    # --- Sample from the fitted GMM ---
    n_samples = 50000
    gmm_s = solver_dict['gmm_s']
    gmm_t = solver_dict['gmm_t']

    gmm_s_samples, _ = gmm_s.sample(n_samples=n_samples)
    gmm_t_samples, _ = gmm_t.sample(n_samples=n_samples)

    color_source = '#ff715b'
    color_target = '#29335C' 
    color_omt = '#eeab00'
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    marker_size =.5
    alpha = .7
    # Plot data points colored by GMM assignment
    scatter = ax[0].scatter(S1_norm[:, 0], S1_norm[:, 1], s=marker_size, c=color_source, edgecolors='none', alpha=alpha)
    # Plot the GMM component means and ellipses
    plot_gmm_ellipses(gmm_s, ax[0], n_std=2.0, edgecolor='red', lw=1, facecolor='none')
    ax[0].scatter(gmm_s.means_[:, 0], gmm_s.means_[:, 1], marker='X', s=10, c='red', edgecolors='black')
    ax[0].set_title(f'GMM Fit ({Ks} Components)')
    ax[0].axis('equal')
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[0].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0].spines['top'].set_color('gray')
    ax[0].spines['right'].set_color('gray')
    ax[0].spines['left'].set_color('gray')
    ax[0].spines['bottom'].set_color('gray')

    ax[1].scatter(S1_norm[:, 0], S1_norm[:, 1], s=marker_size, c=color_target, edgecolors='none', alpha=alpha)
    # Plot samples generated from the GMM
    ax[1].scatter(gmm_s_samples[:, 0], gmm_s_samples[:, 1], color='red', s=marker_size, alpha=alpha-.5, label=f'GMM Samples ({Kt} components)')
    ax[1].set_title('Original data vs. samples from fitted GMM')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[1].axis('equal')
    ax[1].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1].spines['top'].set_color('gray')
    ax[1].spines['right'].set_color('gray')
    ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_color('gray')
    fig.savefig(filehandle_hrot + f'/figures/merfish/GMM_fit_S1_{Ks}.png', dpi=600, bbox_inches='tight')
    plt.close()

    color_source = '#ff715b'
    color_target = '#29335C' 
    color_omt = '#eeab00'
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    marker_size =.5
    alpha = .7
    # Plot data points colored by GMM assignment
    scatter = ax[0].scatter(S2_norm[:, 0], S2_norm[:, 1], s=marker_size, c=color_target, edgecolors='none', alpha=alpha)
    # Plot the GMM component means and ellipses
    plot_gmm_ellipses(gmm_t, ax[0], n_std=2.0, edgecolor='red', lw=1, facecolor='none')
    ax[0].scatter(gmm_t.means_[:, 0], gmm_t.means_[:, 1], marker='X', s=10, c='red', edgecolors='black')
    ax[0].set_title(f'GMM Fit ({Kt} Components)')
    ax[0].axis('equal')
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[0].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0].spines['top'].set_color('gray')
    ax[0].spines['right'].set_color('gray')
    ax[0].spines['left'].set_color('gray')
    ax[0].spines['bottom'].set_color('gray')

    ax[1].scatter(S2_norm[:, 0], S2_norm[:, 1], s=marker_size, c=color_target, edgecolors='none', alpha=alpha)
    # Plot samples generated from the GMM
    ax[1].scatter(gmm_t_samples[:, 0], gmm_t_samples[:, 1], color='red', s=marker_size, alpha=alpha-.5, label=f'GMM Samples ({Kt} components)')
    ax[1].set_title('Original data vs. samples from fitted GMM')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[1].axis('equal')
    ax[1].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1].spines['top'].set_color('gray')
    ax[1].spines['right'].set_color('gray')
    ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_color('gray')
    fig.savefig(filehandle_hrot + f'/figures/merfish/GMM_fit_S2_{Kt}.png', dpi=600, bbox_inches='tight')
    plt.close()

    S2_pred = points_transport(S1_norm, omt_w, solver_dict)
    S_2_pred_torch = torch.tensor(S2_pred).to(dtype=dtype, device=device)
    S1_norm_torch = torch.tensor(S1_norm).to(dtype=dtype, device=device)
    euc_cost = torch.norm(S1_norm_torch - S_2_pred_torch, dim=1).mean().item() / S1_norm.shape[1] * max_S1
    print(f"Average Euclidean cost after transport: {euc_cost}")

    color_source = '#ff715b'
    color_target = '#29335C' 
    color_omt = '#eeab00'
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    fontsize = 22
    marker_size =.5
    alpha = .7
    # Plot data points colored by GMM assignment
    scatter = ax[0].scatter(S1_norm[:, 0], S1_norm[:, 1], s=marker_size, c=color_source, edgecolors='none', alpha=alpha)
    # Plot the GMM component means and ellipses
    plot_gmm_ellipses(gmm_s, ax[0], n_std=2.0, edgecolor='red', lw=1, facecolor='none')
    ax[0].scatter(gmm_s.means_[:, 0], gmm_s.means_[:, 1], marker='X', s=10, c='red', edgecolors='black')
    ax[0].set_title(rf"$K_s = {Ks}$ (Source)", fontsize=fontsize)
    ax[0].axis('off')
    ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[0].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0].spines['top'].set_color('gray')
    ax[0].spines['right'].set_color('gray')
    ax[0].spines['left'].set_color('gray')
    ax[0].spines['bottom'].set_color('gray')
    
    scatter = ax[1].scatter(S2_norm[:, 0], S2_norm[:, 1], s=marker_size, c=color_target, edgecolors='none', alpha=alpha)
    # Plot the GMM component means and ellipses
    plot_gmm_ellipses(gmm_t, ax[1], n_std=2.0, edgecolor='red', lw=1, facecolor='none')
    ax[1].scatter(gmm_t.means_[:, 0], gmm_t.means_[:, 1], marker='X', s=10, c='red', edgecolors='black')
    ax[1].set_title(rf"$K_t = {Kt}$ (Target)", fontsize=fontsize)
    ax[1].axis('off')
    ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[1].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1].spines['top'].set_color('gray')
    ax[1].spines['right'].set_color('gray')
    ax[1].spines['left'].set_color('gray')
    ax[1].spines['bottom'].set_color('gray')

    # ax[2].scatter(S2_norm[:, 0], S2_norm[:, 1], s=marker_size, c=color_target, edgecolors='none', alpha=alpha)
    # Plot samples generated from the GMM
    ax[2].scatter(S2_pred[:, 0], S2_pred[:, 1], color=color_omt, s=marker_size, edgecolors='none', alpha=alpha)
    ax[2].set_title('Transported samples', fontsize=fontsize-2)
    ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    ax[2].axis('off')
    ax[2].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].spines['top'].set_color('gray')
    ax[2].spines['right'].set_color('gray')
    ax[2].spines['left'].set_color('gray')
    ax[2].spines['bottom'].set_color('gray')
    fig.savefig(filehandle_hrot + f'/figures/merfish/OMT_transport_S1_to_S2_{Ks}_{Kt}.png', dpi=600, bbox_inches='tight')
    plt.close()

    X_fname1.seek(0)                # BytesIO reused across the K loop -> rewind
    X_fname2.seek(0)
    X_df_s = pd.read_csv(X_fname1, sep=',', index_col=0)
    X_df_t = pd.read_csv(X_fname2, sep=',', index_col=0)

    print ('Order of genes matches between source and target datasets?', np.all(X_df_t.keys() == X_df_s.keys()))

    df_s = pd.DataFrame(S1_norm, columns=['X', 'Y'], index=X_df_s.index)
    df_t = pd.DataFrame(S2_norm, columns=['X', 'Y'], index=X_df_t.index)
    df_s_tr = pd.DataFrame(S2_pred, columns=['X', 'Y'], index=X_df_s.index)

    genes = list(X_df_s.columns.values)

    X_df_transfered = X_df_s.copy()
    X_df_s['GMM_Component']  = solver_dict['gmm_s'].predict(S1_norm)
    X_df_t['GMM_Component']  = solver_dict['gmm_t'].predict(S2_norm)
    X_df_transfered['GMM_Component']  = solver_dict['gmm_t'].predict(S2_pred)

    X_df_s[genes] = X_df_s.groupby("GMM_Component")[genes].transform("mean")
    X_df_t[genes] = X_df_t.groupby("GMM_Component")[genes].transform("mean")
    X_df_transfered[genes] = X_df_transfered.groupby("GMM_Component")[genes].transform("mean")

    X_df_s_smoothed = X_df_s.copy()
    X_df_t_smoothed = X_df_t.copy()
    X_df_transfered_smoothed = X_df_transfered.copy()

    # Calculate cosine similarity between true and transferred expression
    selected_genes = ["Slc17a7", "Grm4", "Olig1", "Gad1", "Peg10"]
    cos_sim_value = dict(keys=selected_genes, values=[])

    for gene in selected_genes:
        vec1 = np.zeros((solver_dict['gmm_t'].n_components, 1))
        vec2 = np.zeros((solver_dict['gmm_t'].n_components, 1))
        
        for k in range(solver_dict['gmm_s'].n_components):
            target_cells = np.where(X_df_s['GMM_Component'] == k)[0]
            # Skip if there are no points in this component to match
            if len(target_cells) == 0:
                print('No points in component', k)
                continue
            
            X_df_s_smoothed.loc[X_df_s_smoothed.index[target_cells], gene] = X_df_s.loc[X_df_s.index[target_cells], gene].mean()
            
        for k in range(solver_dict['gmm_t'].n_components):
            transport_cells = np.where(X_df_transfered['GMM_Component'] == k)[0]
            target_cells = np.where(X_df_t['GMM_Component'] == k)[0]
            # Skip if there are no points in this component to match
            if len(target_cells) == 0 or len(transport_cells) == 0:
                print('No points in component', k)
                continue
            
            X_df_t_smoothed.loc[X_df_t_smoothed.index[target_cells], gene] = X_df_t.loc[X_df_t.index[target_cells], gene].mean()
            X_df_transfered_smoothed.loc[X_df_transfered_smoothed.index[transport_cells], gene] = X_df_transfered.loc[X_df_transfered.index[transport_cells], gene].mean()
            vec1[k] = X_df_t.loc[X_df_t.index[target_cells], gene].mean()
            vec2[k] = X_df_transfered.loc[X_df_transfered.index[transport_cells], gene].mean()
            
        cos_sim = cosine_similarity(vec1.T, vec2.T)
        cos_sim_value[gene] = cos_sim[0, 0]
        print(f"Cosine similarity for gene {gene} with component {Kt}: {cos_sim_value[gene]:.4f}")
        

    summary_df = dict()
    summary_df['Gene'] = selected_genes
    summary_df['Cosine_Similarity'] = [cos_sim_value[gene] for gene in selected_genes]
    summary_df['Ks'] = Ks
    summary_df['Kt'] = Kt
    summary_df['Tcost'] = total_cost
    summary_df['euc_cost'] = euc_cost
    summary_df['n_iter'] = n_iter
    summary_df['total_iter'] = total_iter
    summary_df['time_elapsed_sec'] = time_elapsed

    summary_df = pd.DataFrame(summary_df)
    summary_df.to_csv(filehandle_hrot + f'/results/merfish_omt_summary_{Ks}_{Kt}.csv', index=False)

    fontsize = 22
    marker_size =.5
    alpha = 1
    for gene in selected_genes:
        # Plot data points colored by GMM assignment
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        
        clr = X_df_s_smoothed[gene] / X_df_s_smoothed[gene].max()
        scatter = ax[0].scatter(S1_norm[:, 0], S1_norm[:, 1], s=marker_size, c=clr, edgecolors='none', alpha=alpha)
        ax[0].set_title(rf"${gene}$ (Source)", fontsize=fontsize)
        ax[0].axis('off')
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
        ax[0].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[0].spines['top'].set_color('gray')
        ax[0].spines['right'].set_color('gray')
        ax[0].spines['left'].set_color('gray')
        ax[0].spines['bottom'].set_color('gray')
        
        clr = X_df_t_smoothed[gene] / X_df_t_smoothed[gene].max()
        scatter = ax[1].scatter(S2_norm[:, 0], S2_norm[:, 1], s=marker_size, c=clr, edgecolors='none', alpha=alpha)
        ax[1].set_title(rf"${gene}$ (Target)", fontsize=fontsize)
        ax[1].axis('off')
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
        ax[1].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[1].spines['top'].set_color('gray')
        ax[1].spines['right'].set_color('gray')
        ax[1].spines['left'].set_color('gray')
        ax[1].spines['bottom'].set_color('gray')

        clr = X_df_transfered_smoothed[gene] / X_df_transfered_smoothed[gene].max()
        scatter = ax[2].scatter(S2_pred[:, 0], S2_pred[:, 1], s=marker_size, c=clr, edgecolors='none', alpha=alpha)
        ax[2].set_title(rf"Predicted ${gene}$", fontsize=fontsize)
        ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
        ax[2].axis('off')
        ax[2].grid(True, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[2].spines['top'].set_color('gray')
        ax[2].spines['right'].set_color('gray')
        ax[2].spines['left'].set_color('gray')
        ax[2].spines['bottom'].set_color('gray')
        # fig.colorbar(scatter, ax=ax[2], fraction=0.046, pad=0.04)

        fig.savefig(filehandle_hrot + f'/figures/merfish/gene_{gene}_S1_to_S2_{Ks}_{Kt}.png', dpi=600, bbox_inches='tight')
        plt.close()