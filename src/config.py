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

    The configuration now separates **local solver settings** (per-model
    dimensionality, initial conditions, and parameter block mapping) from
    **global** calibration knobs, avoiding the previous "4-species with masking"
    indirection. Each model entry (``M1/M2/M3``) is self-contained and only
    includes the species it actually evolves.
    """

    base = dict(
        Ndata=20,
        cov_rel=0.005,
        sigma_obs=0.005,
        target_ess_ratio=0.8,
    )

    if debug:
        base.update(
            N0=10,
            Nposterior=10,
            stages=3,
        )
    else:
        base.update(
            N0=500,
            Nposterior=5000,
            stages=15,
        )

    base.update(
        M1=dict(
            dt=1e-5 if debug else 1e-5,
            maxtimestep=25 if debug else 2500,
            c_const=100.0,
            alpha_const=100.0,
            num_species=2,
            global_species_indices=[0, 1],
            phi_init=[0.2, 0.2],
            theta_indices=[0, 1, 2, 3, 4],
        ),
        M2=dict(
            dt=1e-5 if debug else 1e-5,
            maxtimestep=50 if debug else 5000,
            c_const=100.0,
            alpha_const=10.0,
            num_species=2,
            global_species_indices=[2, 3],
            phi_init=[0.2, 0.2],
            theta_indices=[5, 6, 7, 8, 9],
        ),
        M3=dict(
            dt=1e-4 if debug else 1e-4,
            maxtimestep=75 if debug else 750,
            c_const=25.0,
            alpha_const=0.0,
            num_species=4,
            global_species_indices=[0, 1, 2, 3],
            phi_init=[0.02, 0.02, 0.02, 0.02],
            theta_indices=[10, 11, 12, 13],
        ),
    )

    return base

CONFIG = get_config(DEBUG)


def get_model_config(model_id: str, config: dict | None = None) -> dict:
    """Return a self-contained solver configuration for ``model_id``.

    The helper normalizes legacy keys if present but prefers the explicit
    ``num_species``/``phi_init``/``theta_indices`` fields bundled within each
    model entry.
    """

    cfg = CONFIG if config is None else config
    if model_id not in cfg:
        raise KeyError(f"Model '{model_id}' not found in configuration")

    model_cfg = cfg[model_id].copy()

    # Backward compatibility for legacy layouts (e.g., phi_init_M1)
    legacy_phi_key = f"phi_init_{model_id}"
    if "phi_init" not in model_cfg and legacy_phi_key in cfg:
        model_cfg["phi_init"] = cfg[legacy_phi_key]

    legacy_theta_key = f"theta_active_indices_{model_id}"
    if "theta_indices" not in model_cfg and legacy_theta_key in cfg:
        model_cfg["theta_indices"] = cfg[legacy_theta_key]

    if "num_species" not in model_cfg:
        phi_init = np.asarray(model_cfg.get("phi_init", 0.02), dtype=float)
        model_cfg["num_species"] = phi_init.size if phi_init.ndim else 1

    if "global_species_indices" not in model_cfg:
        n = model_cfg.get("num_species", 1)
        model_cfg["global_species_indices"] = list(range(n))

    return model_cfg

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
