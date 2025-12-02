# src/posterior_tsm_rom.py

import numpy as np
from src.solver_newton import BiofilmNewtonSolver
from src.tsm import BiofilmTSM   # solve_tsm はクラスメソッド

def tsm_generate_phi_timeseries(theta, CONFIG, model_id):
    """
    TSM time-separated mechanics を使い、
    posterior parameter θ → φ(t) を生成する。
    """

    # 1) Newton solver (deterministic) を作る
    solver = BiofilmNewtonSolver(CONFIG, model_id=model_id)

    # 2) TSM インスタンス
    tsm = BiofilmTSM(solver)

    # 3) solve_tsm を実行
    result = tsm.solve_tsm(theta)

    # 4) モデルごとに species 数を決定
    if model_id == "M1":
        Nspecies = 2
    elif model_id == "M2":
        Nspecies = 2
    else:
        Nspecies = 4

    phi = result.mu[:, :Nspecies]
    t = result.t_array

    return t, phi
