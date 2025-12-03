# src/config.py
import numpy as np

# DEBUG / プロット
# Default to paper-accurate (Case II) settings; enable DEBUG for faster test runs.
DEBUG =  True    # Paper-accurate: False
ENABLE_PLOTS = True

# ─────────────────────────────────────────────
# Paper Table 3 に合わせた設定（re_numba版）
# ─────────────────────────────────────────────

def get_config(debug: bool):
    """
    Paper-accurate configuration from Table 3 (Case II)
    """
    if debug:
        return {
            "M1": dict(dt=1e-4, maxtimestep=80,  c_const=100.0, alpha_const=100.0),
            "M2": dict(dt=1e-4, maxtimestep=100, c_const=100.0, alpha_const=10.0),
            "M3": dict(dt=1e-4, maxtimestep=60,  c_const=25.0,  alpha_const=0.0),
            # initial φ
            "phi_init_M1": 0.2,
            "phi_init_M2": 0.2,
            "phi_init_M3": 0.02,
            # sparse data
            "Ndata": 20,
            # TMCMC
            "N0": 10,
            "Nposterior": 50,
            "stages": 10,
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
            "phi_init_M1": 0.2,
            "phi_init_M2": 0.2,
            "phi_init_M3": 0.02,
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
