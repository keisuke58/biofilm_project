# src/posterior_tsm_rom.py

import numpy as np
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
    model_config = CONFIG[model_id]
    phi_init_key = f"phi_init_{model_id}"
    phi_init = CONFIG[phi_init_key]

    # 2) Create Newton solver with proper parameters
    solver = BiofilmNewtonSolver(
        phi_init=phi_init,
        use_numba=True,
        **model_config
    )

    # 3) Determine active theta indices based on model
    if model_id == "M1":
        active_indices = CONFIG.get("theta_active_indices_M1", [0, 1, 2, 3, 4])
        Nspecies = 2
    elif model_id == "M2":
        active_indices = CONFIG.get("theta_active_indices_M2", [5, 6, 7, 8, 9])
        Nspecies = 2
    else:  # M3
        active_indices = CONFIG.get("theta_active_indices_M3", [10, 11, 12, 13])
        Nspecies = 4

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
