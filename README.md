# OMT: Optimal Mixture Transport


> **A Biconvex Formulation for Stable Transport of Mixture Models with a Unique Solution**
> Yeganeh Marghi, Kelly Jin, Uygar Sümbül — Allen Institute, Brain Science
> *40th Conference on Neural Information Processing Systems (NeurIPS 2026)*

**Optimal Mixture Transport (OMT)** is a scalable framework that shifts the transport paradigm from individual samples to mixtures of subpopulations, reformulating the transport problem as a strictly biconvex optimization with a unique global minimizer.

- **Strictly biconvex with a unique solution.** Entropic regularization at the component level (ε₁) and at the mixing-weight level (ε₂) turns the non-convex mixture-transport objective into a strictly biconvex problem (Lemma 1). The GOP algorithm reaches its unique global minimizer in a single iteration (Theorem 1).
- **Stable.** Under weak regularity conditions, which hold for unbounded densities, the OMT map is Lipschitz and tilt-stable. Bounded perturbations of the marginals produce bounded changes in the map (Theorem 2).
- **Scalable.** Each subpopulation is an exponential-family component (Gaussian, `p=q=2`, or asymmetric Laplace / double-exponential, `p=q=1`). These have closed-form component-to-component maps, so the cost scales with the number of components, `O(K₀K₁)`, rather than with the number of samples.

The OMT map is a barycentric combination of closed-form component maps:

```
T_omt(x) = Σ_ij  ω̃_ij(x) · T_ij(x),     ω̃_ij(x) = ω_ij · μ_0i(x) / ν_0(x)
```

---

## Repository structure

```
OMT/
├── README.md
├── omt/                        # The OMT library
│   ├── src/
│   │   ├── general_solver.py   # Main entry point: omt(), mm_ot(), mixture fitting, point transport
│   │   ├── exp_functions.py    # Asymmetric exponential (p,q ∈ {1,2}) components: EM fitting,
│   │   │                       #   closed-form W2 costs, CDF/quantile maps, points_transport()
│   │   ├── exp_functions_gpu.py# GPU (PyTorch) versions of the exponential-family routines
│   │   ├── omt_mix.py          # Mixture-model classes (DoubleExponential, ...)
│   │   ├── functions.py        # Gaussian/Bures-Wasserstein utilities, entropy, MMD, transport cost
│   │   ├── utils.py            # Synthetic data generators (GMMs, shapes, SPD matrices)
│   │   ├── fid.py              # FID evaluation for image experiments
│   │   └── omt_exp_w2.py       # Driver for the Wasserstein-2 benchmark
│   └── examples/
│       └── distributions.py    # Toy source/target distributions for quick demos
│
└── experiments/                # Scripts and notebooks that reproduce the paper
    ├── Synthetic/              # §5 "Impact of mixture parameterization" (Fig. 1, 6, 7)
    ├── W2/                     # §5 "Stability under noise" on the W2 benchmark (Fig. 2, 8–12)
    ├── sci-Plex/               # §5 single-cell drug perturbation, human scRNA-seq (Table 1, Fig. 13)
    ├── MERFISH/                # §5 spatial alignment of mouse brain slices (Table 2, Fig. 4, 14, 15)
    ├── MNIST/                  # §5 / App. F unpaired image-to-image translation (Fig. 5, 20)
    └── ImageNet/               # §5 / App. F high-dimensional alignment (Table 3)
```

### `omt/` — library

| Module | Role |
|---|---|
| `general_solver.py` | `omt(data_source, data_target, ...)` fits source/target mixtures and solves the OMT problem. Also contains `mm_ot` (solver on pre-fitted mixtures), `gmm_fitting`, `sample_mm`, baseline `ot_w2` / `ot_sinkh`, and `points_transport_mm`. |
| `exp_functions.py` | Factorized asymmetric exponential components (Eq. 12): `em_mixture`, `exp_fitting`, `vectorized_W2_cost_matrix`, and `points_transport` (the barycentric map in Eq. 9 and Eq. 13). |
| `exp_functions_gpu.py` | Torch/CUDA counterparts for large datasets. |
| `functions.py`, `utils.py` | Shared math utilities and synthetic data generation. |
| `fid.py` | FID metric for image tasks. |

### `experiments/` — reproducing the paper

Each subfolder is self-contained. **Planned contents:** a short `README.md` (data source, preprocessing, and how to run), the scripts or notebooks, and a `configs/` or parameter block holding the settings used in the paper.

| Folder | Paper section | Data | Baselines | Main outputs | Status |
|---|---|---|---|---|---|
| `Synthetic/` | §5, App. C | 2D/3D shapes (circles, star, cube, …) | — | Fig. 1, 6, 7 | `5_circles.ipynb`, `12_star.ipynb`, `8_cube.ipynb`, `simulations.ipynb` |
| `W2/` | §5, App. D | Wasserstein-2 benchmark [Korotin et al.], d ∈ {2,16,64,128,256} | W2-OT, ExNOT, ENOT, PROGOT, GMM-OT | Fig. 2, 8–12 (noise robustness, runtime, ablations) | *to be added* |
| `sci-Plex/` | §5, App. E.4 | sci-Plex3: A549/K562/MCF7, 5 drugs, 24 h (77,920 cells) | EOT, HiRef, PROGOT, GMM-OT | Table 1, Fig. 13, Tables 4–5 | *to be added* |
| `MERFISH/` | §5, App. E.3/E.5 | Vizgen MERFISH Mouse Brain Receptor Map, 2 slices (~85k cells each) | HiRef, GMM-OT | Table 2, Fig. 4, 14, 15, Table 6 | `fit_components.py`, `aligment_task.ipynb` |
| `MNIST/` | App. F.2 | MNIST (and CIFAR-10) | — | Fig. 5, 20, Table 7 | *to be added* |
| `ImageNet/` | §5, App. F | ImageNet embeddings | Mini-batch OT, FRLC, HiRef | Table 3 | *to be added* |

> The mouse brain development (E11.5–P28, visual cortex) and ageing (adult vs. aged, oligodendrocyte lineage) analyses in Fig. 3 and App. E.6 will go in a separate folder, e.g. `experiments/MouseBrain/`.

Large data files (`*.h5ad`, `*.pkl`) are git-ignored. Each experiment README should explain how to download or regenerate its data.

---

## Installation

```bash
git clone https://github.com/AllenInstitute/OMT.git
cd OMT
pip install numpy scipy scikit-learn pot cvxpy joblib matplotlib torch
# W2 benchmark / PROGOT baselines additionally need:
pip install jax ott-jax
```

The modules in `omt/src` currently import each other by bare module name, so add the folder to your path:

```python
import sys; sys.path.append("omt/src")
```

## Quick start

```python
import numpy as np
import sys; sys.path.append("omt/src")
from general_solver import omt, points_transport_mm

X_src = np.random.randn(5000, 2)              # source samples
X_tgt = np.random.randn(5000, 2) * 0.5 + 3.0  # target samples

# fitting_mode: 'gauss' (p=q=2, GMM), 'exp' (asymmetric exponential), or 'mix'
omt_w, solver = omt(
    X_src, X_tgt,
    n_components=(3, 15),   # (K0, K1)
    eps_gmm=0.1,            # component-level entropic regularization (ε1)
    eps_w=0.0,              # mixing-matrix entropic regularization (ε2)
    method='cvx',           # 'cvx' | 'sinkhorn' | 'w2'
    fitting_mode='gauss',
    random_seed=0,
)

X_mapped = points_transport_mm(X_src, omt_w, solver)   # T_omt(x)
```

`omt_w` is the K₀×K₁ component-coupling matrix Ω. `solver` holds the fitted mixtures (`mm_s`, `mm_t`), timings, and iteration counts.

---



