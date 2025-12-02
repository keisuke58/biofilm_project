# src/validation_m3.py
import numpy as np
from src.posterior_tsm_rom import tsm_generate_phi_timeseries

def generate_M3_validation(results, CONFIG, Ns=40):
    """
    M3 posterior mean (or selected samples) using TSM to create validation time-series.

    Parameters
    ----------
    results : HierarchicalResults
        Results from hierarchical_case2
    CONFIG : dict
        Configuration dictionary
    Ns : int
        Number of posterior samples to draw

    Returns
    -------
    t_val : np.ndarray
        Time array
    phi_val : np.ndarray
        Mean volume fractions (Nt, 4)
    phi_post : np.ndarray
        Posterior samples volume fractions (Ns, Nt, 4)
    """

    # Get last stage samples (M3 posterior)
    samples_M3 = results.tmcmc_M3.samples[-1]  # Last stage

    # Posterior mean of M3 parameters (4 parameters)
    theta_M3_mean = np.mean(samples_M3, axis=0)

    # Build full theta using hierarchical structure
    # Use mean values from M1 and M2, plus M3 parameters
    theta_full = results.theta_final.copy()
    theta_full[10:14] = theta_M3_mean

    # 1本目（基準）- use full theta
    t_val, phi_val = tsm_generate_phi_timeseries(theta_full, CONFIG, "M3")

    # Posterior サンプルから Ns 本
    n_samples = len(samples_M3)
    if Ns > n_samples:
        Ns = n_samples
    idx = np.random.choice(n_samples, Ns, replace=False)
    thetas_M3 = samples_M3[idx]

    Nt = len(t_val)
    phi_post = np.zeros((Ns, Nt, 4))

    for i in range(Ns):
        # Build full theta for each M3 sample
        theta_i = results.theta_final.copy()
        theta_i[10:14] = thetas_M3[i]
        _, phi_i = tsm_generate_phi_timeseries(theta_i, CONFIG, "M3")
        phi_post[i] = phi_i

    return t_val, phi_val, phi_post
