# src/posterior_simulator_tsm.py
import numpy as np
from src.posterior_tsm_rom import tsm_generate_phi_timeseries
from src.config import get_theta_true

def generate_posterior_phi_tsm(posterior_samples, CONFIG, model_id, Ns=40,
                                theta_base=None):
    """
    Posterior から Ns 本サンプリングし、
    TSM（time-separated mechanics）で φ(t) を高速生成する。

    Parameters
    ----------
    posterior_samples : List[np.ndarray] or np.ndarray
        TMCMC posterior samples. If List, uses last stage samples[-1]
    CONFIG : dict
        Configuration dictionary
    model_id : str
        "M1", "M2", or "M3"
    Ns : int
        Number of samples to draw
    theta_base : np.ndarray, optional
        Base theta vector (14,). Model-specific parameters will be replaced.
        If None, uses zeros for non-active parameters.

    Returns
    -------
    t : np.ndarray
        Time array
    phi_all : np.ndarray
        Volume fractions for all samples (Ns, Nt, Nspecies)
    """

    # Handle List[np.ndarray] from TMCMCResult
    if isinstance(posterior_samples, list):
        samples = posterior_samples[-1]  # Last stage
    else:
        samples = posterior_samples

    # Initialize base theta if not provided
    if theta_base is None:
        theta_base = np.zeros(14)

    # Determine parameter indices for this model
    if model_id == "M1":
        param_indices = range(0, 5)  # a11, a12, a22, b1, b2
    elif model_id == "M2":
        param_indices = range(5, 10)  # a33, a34, a44, b3, b4
    else:  # M3
        param_indices = range(10, 14)  # a13, a14, a23, a24

    # 1) Posterior index sampling
    n_available = len(samples)
    if Ns > n_available:
        Ns = n_available
    idx = np.random.choice(n_available, Ns, replace=False)
    thetas_partial = samples[idx]

    # 2) Build full theta vectors
    thetas_full = []
    for theta_p in thetas_partial:
        theta_f = theta_base.copy()
        for i, param_idx in enumerate(param_indices):
            theta_f[param_idx] = theta_p[i]
        thetas_full.append(theta_f)

    # 3) First sample to determine dimensions
    t0, phi0 = tsm_generate_phi_timeseries(thetas_full[0], CONFIG, model_id)
    Nt = len(t0)
    Nspecies = phi0.shape[1]

    phi_all = np.zeros((Ns, Nt, Nspecies))
    phi_all[0] = phi0

    # 4) Remaining Ns-1 samples
    for i in range(1, Ns):
        _, phi_i = tsm_generate_phi_timeseries(thetas_full[i], CONFIG, model_id)
        phi_all[i] = phi_i

    return t0, phi_all
