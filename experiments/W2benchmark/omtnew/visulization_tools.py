import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_ot_map(
                x, 
                y, 
                pi, 
                prob_x,
                prob_y,
                hspace=0.02, 
                wspace=.02, 
                linewidth = 1.5,
                color_x = 'dimgray',
                color_y = 'tomato',
                pi_cmap = 'Reds',
                dots=None,
                axis_range=None,
                fig_grid=False,
                levels=100,
                legend=True,
                plot_mode='contour', # 'contour' or 'heatmap'
                grid_color='#d4d4cd',
                text=[''],
                text_loc=[(0.95, 0.95)],
                title='',
                title_loc=(0.1, 0.9),
                ):
    
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=hspace, wspace=wspace)

    # Main contour plot (joint)
    ax_joint = fig.add_subplot(grid[1:4, 1:4])
    if plot_mode == 'contour':
        ax_joint.contour(Y, X, pi, cmap=pi_cmap, levels=levels)
    elif plot_mode == 'heatmap':
        ax_joint.imshow(pi.T, interpolation='nearest', cmap=pi_cmap, aspect='auto', origin='lower', extent=[np.min(y), np.max(y), np.min(x), np.max(x)], alpha=0.8)
    else:
        raise ValueError("plot_mode must be 'contour' or 'heatmap'.")
    
    if dots is not None:
        for i in range(dots.shape[0]):
            for j in range(dots.shape[1]):
                ax_joint.scatter(dots[i,j][0][1], dots[i,j][0][0], s=20, color='black', alpha=1, marker='*', zorder=10)

    # cbar = fig.colorbar(cs, ax=ax_joint)
    if axis_range is None:
        ax_joint.set_ylim(np.min(x), np.max(x))
        ax_joint.set_xlim(np.min(y), np.max(y))
    else:
        ax_joint.set_ylim(axis_range[0,0], axis_range[1,0])
        ax_joint.set_xlim(axis_range[0,1], axis_range[1,1])
        
    ax_joint.invert_yaxis()
    # Hide axis spines, ticks, and labels individually
    ax_joint.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, which='both')
    if fig_grid:
        ax_joint.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5) # Use True for major grid
    else:
        ax_joint.grid(False)
    ax_joint.spines['top'].set_color('#8e8e7b')
    ax_joint.spines['right'].set_color('#8e8e7b')
    ax_joint.spines['left'].set_color('#8e8e7b')
    ax_joint.spines['bottom'].set_color('#8e8e7b')
    
    for tx, loc in zip(text, text_loc):
        ax_joint.text(
                    loc[0],
                    loc[1], 
                    f"{tx}",
                    transform=ax_joint.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    horizontalalignment='right',
                    )
    
    ax_joint.text(
                title_loc[0],
                title_loc[1], 
                f"{title}",
                transform=ax_joint.transAxes,
                fontsize=20,
                verticalalignment='top',
                horizontalalignment='right',
                )   

    # Marginal plot for X (top)
    ax_marg_x = fig.add_subplot(grid[0, 1:4])
    plt.fill_between(y, prob_y, color=color_y, linewidth=linewidth)
    
    if axis_range is None:
        ax_marg_x.set_xlim(np.min(y), np.max(y))
    else:
        ax_marg_x.set_xlim(axis_range[0,1], axis_range[1,1])
    # ax_marg_x.set_ylabel('Density')
    # ax_marg_x.set_title('Target Distribution')
    
    ax_marg_x.tick_params(axis='x', labelbottom=False) 
    ax_marg_x.spines['top'].set_visible(False)
    ax_marg_x.spines['right'].set_visible(False)
    ax_marg_x.spines['left'].set_visible(False)
    ax_marg_x.spines['bottom'].set_visible(False)
    ax_marg_x.tick_params(axis='y', labelleft=False)  
    ax_marg_x.tick_params(axis='x', labelbottom=False)
    for tick in ax_marg_x.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax_marg_x.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    # # Marginal plot for Y (right)
    ax_marg_y = fig.add_subplot(grid[1:4, 0])
    plt.fill_betweenx(x, prob_x, color=color_x, linewidth=linewidth)
    
    if axis_range is None:
        ax_marg_y.set_ylim(np.min(x), np.max(x))
    else:
        ax_marg_y.set_ylim(axis_range[0,0], axis_range[1,0])
        
    ax_marg_y.invert_xaxis()
    ax_marg_y.invert_yaxis()
    
    ax_marg_y.set_xticklabels([])
    ax_marg_y.set_yticklabels([])
    ax_marg_y.spines['top'].set_visible(False)
    ax_marg_y.spines['right'].set_visible(False)
    ax_marg_y.spines['left'].set_visible(False)
    ax_marg_y.spines['bottom'].set_visible(False)
    for tick in ax_marg_y.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in ax_marg_y.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    if legend:
        # Create legend handles
        source_patch = mpatches.Patch(color=color_x, label='Source', linewidth=0.5)
        target_patch = mpatches.Patch(color=color_y, label='Target', linewidth=0.5)

        # Add legend to the main joint plot
        fig.legend(handles=[source_patch, target_patch], loc='upper left', frameon=False, bbox_to_anchor=(0.15, 0.88))
    
    plt.show()
    
    return fig
    
    
def plot_gmm_ellipses(gmm, ax, n_std=2.0, facecolor='none', **kwargs):
    """Plots ellipses representing the GMM components."""
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

        # Eigenvalue decomposition for ellipse orientation and size
        v, w = linalg.eigh(covariances)
        u = w[0] / linalg.norm(w[0]) # Eigenvector for major axis
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # Convert to degrees
        v = n_std * np.sqrt(v) # Scale eigenvalues to represent n_std deviations

        ell = mpatches.Ellipse(gmm.means_[n, :2], v[0], v[1], angle=180 + angle,
                              facecolor=facecolor, **kwargs)
        ax.add_patch(ell)
