# src/config.py
import os
import numpy as np

# Hierarchical calibration overview (M1/M2/M3 + validation)
# - 14 total parameters with Uniform(0, 3) priors
# - cov_rel = sigma_obs = 0.005 (0.5% CoV)
# - Sparse data: Ndata = 20
# - M1 & M2 are two-species submodels; M3 activates all four species

# DEBUG / プロット
# Allow overriding via environment variable (DEBUG=true/false)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENABLE_PLOTS = True

# ─────────────────────────────────────────────
# Paper Table 3 に合わせた設定（re_numba版）
# ─────────────────────────────────────────────

def get_config(debug: bool):
    """
    Paper-accurate configuration from Table 3 (Case II)

    Case II uses genuine 2-species submodels:
    - M1: Species 1-2 only (coarse model)
    - M2: Species 3-4 only (medium model)
    - M3: All 4 species (fine model)
    """
    if debug:
        return {
            "M1": dict(dt=1e-5, maxtimestep=25, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=50, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=75,  c_const=25.0,  alpha_const=0.0),
            # initial φ (SCALAR: all species start at same value)
            # 2-species behavior comes from active_species masking
            "phi_init_M1": 0.2,   # All species start at 0.2
            "phi_init_M2": 0.2,   # All species start at 0.2
            "phi_init_M3": 0.02,  # All species start at 0.02
            # Active species for true 2-species submodels
            "active_species_M1": [0, 1],  # Only 1-2 grow/interact
            "active_species_M2": [2, 3],  # Only 3-4 grow/interact
            "active_species_M3": None,    # All species active
            # sparse data
            "Ndata": 20,
            # TMCMC
            "N0": 10,
            "Nposterior": 10,
            "stages": 3,
            "target_ess_ratio": 0.8,
            # TSM
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,
            # lik
            "sigma_obs": 0.005,
        }
    else:
        # FULL 版は re_numba と同じで OK（必要に応じて調整）
        return {
            "M1": dict(dt=1e-5, maxtimestep=2500, c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-5, maxtimestep=5000, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=750,  c_const=25.0,  alpha_const=0.0),
            # initial φ (SCALAR: all species start at same value)
            # 2-species behavior comes from active_species masking
            "phi_init_M1": 0.2,   # All species start at 0.2
            "phi_init_M2": 0.2,   # All species start at 0.2
            "phi_init_M3": 0.02,  # All species start at 0.02
            # Active species for true 2-species submodels
            "active_species_M1": [0, 1],  # Only 1-2 grow/interact
            "active_species_M2": [2, 3],  # Only 3-4 grow/interact
            "active_species_M3": None,    # All species active
            "Ndata": 20,
            "N0": 500,
            "Nposterior": 5000,
            "stages": 15,
            "target_ess_ratio": 0.8,
            "theta_active_indices_M1": [0, 1, 2, 3, 4],
            "theta_active_indices_M2": [5, 6, 7, 8, 9],
            "theta_active_indices_M3": [10, 11, 12, 13],
            "cov_rel": 0.005,
            "sigma_obs": 0.005,
        }

CONFIG = get_config(DEBUG)

# ─────────────────────────────────────────────
# 真のパラメータ θ* （re_numba の TRUE_PARAMS）
# ─────────────────────────────────────────────

TRUE_PARAMS = {
    "a11": 0.8, "a12": 2.0, "a22": 1.0, "b1": 0.1, "b2": 0.2,
    "a33": 1.5, "a34": 1.0, "a44": 2.0, "b3": 0.3, "b4": 0.4,
    "a13": 2.0, "a14": 1.0, "a23": 2.0, "a24": 1.0,
}

def get_theta_true() -> np.ndarray:
    return np.array([
        TRUE_PARAMS["a11"], TRUE_PARAMS["a12"], TRUE_PARAMS["a22"],
        TRUE_PARAMS["b1"],  TRUE_PARAMS["b2"],
        TRUE_PARAMS["a33"], TRUE_PARAMS["a34"], TRUE_PARAMS["a44"],
        TRUE_PARAMS["b3"],  TRUE_PARAMS["b4"],
        TRUE_PARAMS["a13"], TRUE_PARAMS["a14"],
        TRUE_PARAMS["a23"], TRUE_PARAMS["a24"],
    ])
