# src/validation_m3.py
import numpy as np
from src.posterior_tsm_rom import tsm_generate_phi_timeseries

def generate_M3_validation(results, CONFIG, Ns=40):
    """
    M3 posterior mean (or selected samples) using TSM to create validation time-series.
    Returns
    -------
    t_val, phi_val: (Nt, 4)
    phi_post_val: (Ns, Nt, 4)
    """

    # posterior mean
    theta_mean = np.mean(results.tmcmc_M3.samples, axis=0)

    # 1本目（基準）
    t_val, phi_val = tsm_generate_phi_timeseries(theta_mean, CONFIG, "M3")

    # posterior サンプルから Ns 本
    idx = np.random.choice(len(results.tmcmc_M3.samples), Ns, replace=False)
    thetas = results.tmcmc_M3.samples[idx]

    Nt = len(t_val)
    phi_post = np.zeros((Ns, Nt, 4))

    for i in range(Ns):
        _, phi_i = tsm_generate_phi_timeseries(thetas[i], CONFIG, "M3")
        phi_post[i] = phi_i

    return t_val, phi_val, phi_post
