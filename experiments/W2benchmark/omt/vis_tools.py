import numpy as np
from scipy import linalg
import matplotlib.patches as mpatches


def plot_gmm_ellipses(gmm, ax, n_std=2.0, facecolor='none', **kwargs):
    """Plots ellipses representing the GMM components."""
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

        # Eigenvalue decomposition for ellipse orientation and size
        v, w = linalg.eigh(covariances)
        u = w[0] / linalg.norm(w[0]) # Eigenvector for major axis
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # Convert to degrees
        v = n_std * np.sqrt(v) # Scale eigenvalues to represent n_std deviations

        ell = mpatches.Ellipse(gmm.means_[n, :2], v[0], v[1], angle=180 + angle, facecolor=facecolor, **kwargs)
        ax.add_patch(ell)
        

def darken_hex_color(hex_color, factor=0.8):
    """
    Darkens a hex color by a given factor.
    
    Parameters:
        hex_color (str): Hex color code, e.g., '#RRGGBB'
        factor (float): Darkness factor (0 < factor < 1). Lower means darker.
        
    Returns:
        str: Darkened hex color code
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Apply darkening factor
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Ensure values stay within 0-255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)



def invert_hex_color(hex_color):
    """
    Returns the opposite (complementary) hex color code.
    
    Parameters:
        hex_color (str): Hex color code, e.g., '#RRGGBB'
    
    Returns:
        str: Inverted hex color code
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Invert each component
    r_inv = 255 - r
    g_inv = 255 - g
    b_inv = 255 - b

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r_inv, g_inv, b_inv)

