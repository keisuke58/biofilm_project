# src/posterior_tsm_rom.py

import numpy as np
from src.config import get_model_config
from src.solver_newton import BiofilmNewtonSolver
from src.tsm import BiofilmTSM   # solve_tsm はクラスメソッド

def tsm_generate_phi_timeseries(theta, CONFIG, model_id):
    """
    TSM time-separated mechanics を使い、
    posterior parameter θ → φ(t) を生成する。

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector (14,)
    CONFIG : dict
        Configuration dictionary with M1, M2, M3 settings
    model_id : str
        "M1", "M2", or "M3"

    Returns
    -------
    t : np.ndarray
        Time array
    phi : np.ndarray
        Volume fractions (Nt, Nspecies)
    """

    # 1) Get model-specific configuration
    model_config = get_model_config(model_id, CONFIG)

    # 2) Create Newton solver with proper parameters
    solver_kwargs = {k: v for k, v in model_config.items() if k not in {"phi_init", "num_species", "theta_indices", "global_species_indices"}}
    solver = BiofilmNewtonSolver(
        phi_init=model_config.get("phi_init", 0.02),
        species_count=model_config.get("num_species"),
        theta_indices=model_config.get("theta_indices"),
        use_numba=True,
        **solver_kwargs,
    )

    # 3) Determine active theta indices based on model
    active_indices = model_config.get("theta_indices")
    Nspecies = model_config.get("num_species", 4)

    # 4) Create TSM instance
    tsm = BiofilmTSM(
        solver,
        cov_rel=CONFIG.get("cov_rel", 0.005),
        active_theta_indices=active_indices
    )

    # 5) Solve TSM
    result = tsm.solve_tsm(theta)

    # 6) Extract phi (volume fractions for species of interest)
    # result.mu has shape (Nt, 10): [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma]
    phi = result.mu[:, :Nspecies]
    t = result.t_array

    return t, phi
