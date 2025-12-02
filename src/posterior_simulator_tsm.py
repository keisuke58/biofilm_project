# src/posterior_simulator_tsm.py
import numpy as np
from src.posterior_tsm_rom import tsm_generate_phi_timeseries

def generate_posterior_phi_tsm(posterior_samples, CONFIG, model_id, Ns=40):
    """
    posterior から Ns 本サンプリングし、
    TSM（time-separated mechanics）で φ(t) を高速生成する。
    """

    # 1) posterior index sampling
    idx = np.random.choice(len(posterior_samples), Ns, replace=False)
    thetas = posterior_samples[idx]

    # 2) 1本目で時間長 Nt と Nspecies を確定
    t0, phi0 = tsm_generate_phi_timeseries(thetas[0], CONFIG, model_id)
    Nt = len(t0)
    Nspecies = phi0.shape[1]

    phi_all = np.zeros((Ns, Nt, Nspecies))
    phi_all[0] = phi0

    # 3) 残り Ns-1 本
    for i in range(1, Ns):
        _, phi_i = tsm_generate_phi_timeseries(thetas[i], CONFIG, model_id)
        phi_all[i] = phi_i

    return t0, phi_all
