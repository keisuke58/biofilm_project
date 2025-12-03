# src/hierarchical.py
"""
Hierarchical Bayesian inference for biofilm multi-scale models.

This module implements the three-stage hierarchical parameter estimation:
- M1: Coarse model (species 1-2, parameters θ[0:5])
- M2: Medium model (species 3-4, parameters θ[5:10])
- M3: Fine model (cross-interactions, parameters θ[10:14])

References
----------
.. [1] Fritsch et al. (2025), "Hierarchical Bayesian Inference for
       Multi-Scale Biofilm Formation Models"
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .config import CONFIG, get_theta_true
from .solver_newton import BiofilmNewtonSolver
from .tsm import BiofilmTSM, TSMResult, log_likelihood_sparse
from .tmcmc import TMCMCResult, tmcmc
from .data_utils import select_sparse_data_indices
from .progress import ProgressTracker


@dataclass
class HierarchicalResults:
    """
    Results from hierarchical Bayesian calibration.

    Attributes
    ----------
    M1_samples : np.ndarray
        Final posterior samples for M1 parameters (N, 5)
    M2_samples : np.ndarray
        Final posterior samples for M2 parameters (N, 5)
    M3_samples : np.ndarray
        Final posterior samples for M3 parameters (N, 4)
    data_M1 : np.ndarray
        Sparse observational data used in M1 calibration (Ndata, 2)
    data_M2 : np.ndarray
        Sparse observational data used in M2 calibration (Ndata, 2)
    data_M3 : np.ndarray
        Sparse observational data used in M3 calibration (Ndata, 4)
    t1_sparse : np.ndarray
        Time points corresponding to data_M1 indices
    t2_sparse : np.ndarray
        Time points corresponding to data_M2 indices
    t3_sparse : np.ndarray
        Time points corresponding to data_M3 indices
    idx1 : np.ndarray
        Indices of selected sparse observations for M1
    idx2 : np.ndarray
        Indices of selected sparse observations for M2
    idx3 : np.ndarray
        Indices of selected sparse observations for M3
    theta_M1_mean : np.ndarray
        Posterior mean of M1 parameters (5,)
    theta_M2_mean : np.ndarray
        Posterior mean of M2 parameters (5,)
    theta_M3_mean : np.ndarray
        Posterior mean of M3 parameters (4,)
    theta_final : np.ndarray
        Complete parameter vector with all posterior means (14,)
    tmcmc_M1 : TMCMCResult
        Complete TMCMC results for M1
    tmcmc_M2 : TMCMCResult
        Complete TMCMC results for M2
    tmcmc_M3 : TMCMCResult
        Complete TMCMC results for M3
    """
    M1_samples: np.ndarray
    M2_samples: np.ndarray
    M3_samples: np.ndarray
    data_M1: np.ndarray
    data_M2: np.ndarray
    data_M3: np.ndarray
    t1_sparse: np.ndarray
    t2_sparse: np.ndarray
    t3_sparse: np.ndarray
    idx1: np.ndarray
    idx2: np.ndarray
    idx3: np.ndarray
    theta_M1_mean: np.ndarray
    theta_M2_mean: np.ndarray
    theta_M3_mean: np.ndarray
    theta_final: np.ndarray
    tmcmc_M1: TMCMCResult
    tmcmc_M2: TMCMCResult
    tmcmc_M3: TMCMCResult

    # =============================================================================
# NUMBA ACCELERATION
# =============================================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not available: using pure NumPy (slower)")


def hierarchical_case2(config: Optional[Dict] = None) -> HierarchicalResults:
    """
    Hierarchical Bayesian parameter estimation for biofilm models.

    Implements sequential updating through three model scales:
    1. M1: Estimates θ[0:5] (species 1-2 interaction parameters)
    2. M2: Estimates θ[5:10] (species 3-4 interaction parameters)
    3. M3: Estimates θ[10:14] (cross-species interaction parameters)

    Each stage uses the posterior from the previous stage as prior,
    enabling efficient exploration of the 14-dimensional parameter space.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary containing:
        - M1, M2, M3: Model-specific settings (dt, maxtimestep, etc.)
        - N0: Initial number of TMCMC samples
        - stages: Number of TMCMC stages
        - Ndata: Number of sparse data points
        - sigma_obs: Observation noise standard deviation
        If None, uses global CONFIG from config.py

    Returns
    -------
    HierarchicalResults
        Complete results including:
        - Final posterior samples for all three stages
        - Posterior means for each parameter group
        - Complete TMCMC diagnostics (convergence, ESS, etc.)
        - Combined parameter vector theta_final

    Notes
    -----
    The hierarchical structure assumes:
    - M1 calibrates species 1-2 independently
    - M2 calibrates species 3-4 using M1 posterior mean
    - M3 calibrates cross-interactions using M1+M2 means

    Prior distributions are U(0,3) for all parameters as per the paper.

    Examples
    --------
    >>> from src.config import CONFIG
    >>> results = hierarchical_case2(CONFIG)
    >>> print(f"Final RMSE: {np.sqrt(np.mean((results.theta_final - theta_true)**2))}")

    References
    ----------
    .. [1] Fritsch et al. (2025), "Hierarchical Bayesian Inference..."
    """
    if config is None:
        config = CONFIG

    theta_true = get_theta_true()
    sigma_obs = config.get("sigma_obs", 0.005)
    Ndata = config.get("Ndata", 20)

    # ── ここから先は re_numba.py の hierarchical_case2 をそのまま移植 ──
    # - bounds の設定
    # - BiofilmNewtonSolver の生成 (M1/M2/M3)
    # - 前進シミュレーション
    # - select_sparse_data_indices で疎データを生成
    # - Stage 1: M1 (TSM + tmcmc)
    # - Stage 2: M2
    # - Stage 3: M3
    # - HierarchicalResults を return

    # それぞれの logL_M1, logL_M2, logL_M3 では、
    # log_likelihood_sparse を呼ぶように re_numba と同じ構成にする。
    # tmcmc 呼び出しだけは `from .tmcmc import tmcmc` を使う。
    # ─────────────────────────────────────────────────────
    # ← re_numba.hierarchical_case2 の実装をコピペ & import 修正

    ## Prior bounds for Case II (Paper: "All prior distributions are chosen as U(0,3)")
    
    # bounds = [(0.0, 3.0)] * 14

    bounds = [
        (0.0, 3.0),  # a11
        (0.0, 3.0),  # a12
        (0.0, 3.0),  # a22
        (0.0, 3.0),  # b1
        (0.0, 3.0),  # b2
        (0.0, 3.0),  # a33
        (0.0, 3.0),  # a34
        (0.0, 3.0),  # a44
        (0.0, 3.0),  # b3
        (0.0, 3.0),  # b4
        (0.0, 3.0),  # a13
        (0.0, 3.0),  # a14
        (0.0, 3.0),  # a23
        (0.0, 3.0),  # a24
    ]
    
    # bounds = [
    #     (0.5, 2.5),  # a11
    #     (0.5, 2.5),  # a12
    #     (0.5, 2.5),  # a22
    #     (0.0, 1.0),  # b1
    #     (0.0, 1.0),  # b2
    #     (0.5, 2.5),  # a33
    #     (0.5, 2.5),  # a34
    #     (0.5, 2.5),  # a44
    #     (0.0, 1.0),  # b3
    #     (0.0, 1.0),  # b4
    #     (0.5, 2.5),  # a13
    #     (0.5, 2.5),  # a14
    #     (0.5, 2.5),  # a23
    #     (0.5, 2.5),  # a24
    # ]
    
    # Prior bounds tuned for better performance
    # bounds = [
    #     (0.4, 1.2),   # a11
    #     (1.0, 3.0),   # a12
    #     (0.5, 1.5),   # a22
    #     (0.05, 0.15), # b1
    #     (0.10, 0.30), # b2
    #     (0.75, 2.25), # a33
    #     (0.5, 1.5),   # a34
    #     (1.0, 3.0),   # a44
    #     (0.15, 0.45), # b3
    #     (0.20, 0.60), # b4
    #     (1.0, 3.0),   # a13
    #     (0.5, 1.5),   # a14
    #     (1.0, 3.0),   # a23
    #     (0.5, 1.5),   # a24
    # ]

    def log_prior_full(theta):
        theta = np.asarray(theta, dtype=float)
        for i, (low, high) in enumerate(bounds):
            if theta[i] < low or theta[i] > high:
                return -np.inf
        return 0.0

    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n[Step 0] Generating synthetic data...")
    np.random.seed(42)
    
    # M1 solver with phi_init = 0.2
    solver_M1 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M1"],  # 0.2
        use_numba=HAS_NUMBA,
        **config["M1"]
    )
    t1, g1 = solver_M1.run_deterministic(theta_true, show_progress=True)
    
    # M2 solver with phi_init = 0.2
    solver_M2 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M2"],  # 0.2
        use_numba=HAS_NUMBA,
        **config["M2"]
    )
    t2, g2 = solver_M2.run_deterministic(theta_true, show_progress=True)
    
    # M3 solver with phi_init = 0.02
    solver_M3 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M3"],  # 0.02
        use_numba=HAS_NUMBA,
        **config["M3"]
    )
    t3, g3 = solver_M3.run_deterministic(theta_true, show_progress=True)
    
    # Compute observables (phi_bar = phi * psi)
    obs1_full = np.stack([g1[:, 0]*g1[:, 5], g1[:, 1]*g1[:, 6]], axis=1)
    obs2_full = np.stack([g2[:, 2]*g2[:, 7], g2[:, 3]*g2[:, 8]], axis=1)
    obs3_full = np.stack([g3[:, i]*g3[:, 5+i] for i in range(4)], axis=1)
    
    # SELECT SPARSE DATA POINTS (CRITICAL!)
    idx1 = select_sparse_data_indices(len(t1), Ndata)
    idx2 = select_sparse_data_indices(len(t2), Ndata)
    idx3 = select_sparse_data_indices(len(t3), Ndata)
    
    t1_sparse, data_M1 = t1[idx1], obs1_full[idx1]
    t2_sparse, data_M2 = t2[idx2], obs2_full[idx2]
    t3_sparse, data_M3 = t3[idx3], obs3_full[idx3]
    
    # Add observation noise
    data_M1 += np.random.normal(0, sigma_obs, data_M1.shape)
    data_M2 += np.random.normal(0, sigma_obs, data_M2.shape)
    data_M3 += np.random.normal(0, sigma_obs, data_M3.shape)
    
    print(f"  ✓ SPARSE Data: M1={data_M1.shape}, M2={data_M2.shape}, M3={data_M3.shape}")
    print(f"  ✓ Data indices: M1={idx1[:3]}...{idx1[-1]}, M2={idx2[:3]}...{idx2[-1]}")

    # =========================================================================
    # STAGE 1: M1 (species 1 & 2)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 1: M1 (species 1 & 2)")
    print(f"  Initial ϕ = {config['phi_init_M1']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M1 = BiofilmTSM(solver_M1, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M1"])
    
    theta_prior_center = theta_true.copy()
    
    def logL_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        try:
            tsm_res = tsm_M1.solve_tsm(theta_full)
            # Extract at SPARSE data indices
            phi, psi = tsm_res.mu[idx1, 0:4], tsm_res.mu[idx1, 5:9]
            obs = np.stack([phi[:, 0]*psi[:, 0], phi[:, 1]*psi[:, 1]], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx1, 0:4], tsm_res.sigma2[idx1, 5:9]
            obs_var = np.stack([
                phi[:, 0]**2 * var_psi[:, 0] + psi[:, 0]**2 * var_phi[:, 0],
                phi[:, 1]**2 * var_psi[:, 1] + psi[:, 1]**2 * var_phi[:, 1],
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M1, sigma_obs)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            # Catch numerical errors from solver/TSM
            if config.get("verbose", False):
                print(f"[M1] Likelihood evaluation failed for theta_M1: {type(e).__name__}: {e}")
            return -1e20

    def log_prior_M1(theta_M1):
        theta_full = theta_prior_center.copy()
        theta_full[0:5] = theta_M1
        return log_prior_full(theta_full)

    rng = np.random.default_rng(1234)
    init_M1 = rng.uniform([bounds[i][0] for i in range(5)], 
                          [bounds[i][1] for i in range(5)], 
                          size=(config["N0"], 5))
    
    t0 = time.time()
    res_M1 = tmcmc(logL_M1, log_prior_M1, init_M1, n_stages=config["stages"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=1234,
                   show_progress=True, model_name="M1")
    t1_time = time.time() - t0
    
    samples_M1 = res_M1.samples[-1]
    theta_M1_mean = np.mean(samples_M1, axis=0)
    print(f"  M1 posterior mean: {theta_M1_mean}")
    print(f"  M1 true values:    {theta_true[0:5]}")
    print(f"  M1 time: {t1_time:.1f}s, converged: {res_M1.converged}")

    theta_stage2_center = theta_prior_center.copy()
    theta_stage2_center[0:5] = theta_M1_mean

    # =========================================================================
    # STAGE 2: M2 (species 3 & 4)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 2: M2 (species 3 & 4)")
    print(f"  Initial ϕ = {config['phi_init_M2']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M2 = BiofilmTSM(solver_M2, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M2"])
    
    def logL_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        try:
            tsm_res = tsm_M2.solve_tsm(theta_full)
            phi, psi = tsm_res.mu[idx2, 0:4], tsm_res.mu[idx2, 5:9]
            obs = np.stack([phi[:, 2]*psi[:, 2], phi[:, 3]*psi[:, 3]], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx2, 0:4], tsm_res.sigma2[idx2, 5:9]
            obs_var = np.stack([
                phi[:, 2]**2 * var_psi[:, 2] + psi[:, 2]**2 * var_phi[:, 2],
                phi[:, 3]**2 * var_psi[:, 3] + psi[:, 3]**2 * var_phi[:, 3],
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M2, sigma_obs)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            # Catch numerical errors from solver/TSM
            if config.get("verbose", False):
                print(f"[M2] Likelihood evaluation failed for theta_M2: {type(e).__name__}: {e}")
            return -1e20

    def log_prior_M2(theta_M2):
        theta_full = theta_stage2_center.copy()
        theta_full[5:10] = theta_M2
        return log_prior_full(theta_full)

    init_M2 = rng.uniform([bounds[i][0] for i in range(5, 10)], 
                          [bounds[i][1] for i in range(5, 10)], 
                          size=(config["N0"], 5))
    
    t0 = time.time()
    res_M2 = tmcmc(logL_M2, log_prior_M2, init_M2, n_stages=config["stages"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=5678,
                   show_progress=True, model_name="M2")
    t2_time = time.time() - t0
    
    samples_M2 = res_M2.samples[-1]
    theta_M2_mean = np.mean(samples_M2, axis=0)
    print(f"  M2 posterior mean: {theta_M2_mean}")
    print(f"  M2 true values:    {theta_true[5:10]}")
    print(f"  M2 time: {t2_time:.1f}s, converged: {res_M2.converged}")

    theta_stage3_center = theta_stage2_center.copy()
    theta_stage3_center[5:10] = theta_M2_mean

    # =========================================================================
    # STAGE 3: M3 (cross interactions)
    # =========================================================================
    print("\n" + "="*72)
    print("  Stage 3: M3 (cross interactions)")
    print(f"  Initial ϕ = {config['phi_init_M3']}, Ndata = {Ndata}")
    print("="*72)
    
    tsm_M3 = BiofilmTSM(solver_M3, cov_rel=config["cov_rel"],
                        active_theta_indices=config["theta_active_indices_M3"])
    
    def logL_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        try:
            tsm_res = tsm_M3.solve_tsm(theta_full)
            phi, psi = tsm_res.mu[idx3, 0:4], tsm_res.mu[idx3, 5:9]
            obs = np.stack([phi[:, i]*psi[:, i] for i in range(4)], axis=1)
            var_phi, var_psi = tsm_res.sigma2[idx3, 0:4], tsm_res.sigma2[idx3, 5:9]
            obs_var = np.stack([
                phi[:, i]**2 * var_psi[:, i] + psi[:, i]**2 * var_phi[:, i]
                for i in range(4)
            ], axis=1)
            return log_likelihood_sparse(obs, obs_var, data_M3, sigma_obs)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            # Catch numerical errors from solver/TSM
            if config.get("verbose", False):
                print(f"[M3] Likelihood evaluation failed for theta_M3: {type(e).__name__}: {e}")
            return -1e20

    def log_prior_M3(theta_M3):
        theta_full = theta_stage3_center.copy()
        theta_full[10:14] = theta_M3
        return log_prior_full(theta_full)

    init_M3 = rng.uniform([bounds[i][0] for i in range(10, 14)], 
                          [bounds[i][1] for i in range(10, 14)], 
                          size=(config["N0"], 4))
    
    t0 = time.time()
    res_M3 = tmcmc(logL_M3, log_prior_M3, init_M3, n_stages=config["stages"],
                   target_ess_ratio=config["target_ess_ratio"], random_state=9012,
                   show_progress=True, model_name="M3")
    t3_time = time.time() - t0
    
    samples_M3 = res_M3.samples[-1]
    theta_M3_mean = np.mean(samples_M3, axis=0)
    print(f"  M3 posterior mean: {theta_M3_mean}")
    print(f"  M3 true values:    {theta_true[10:14]}")
    print(f"  M3 time: {t3_time:.1f}s, converged: {res_M3.converged}")

    theta_final = theta_stage3_center.copy()
    theta_final[10:14] = theta_M3_mean

    return HierarchicalResults(
        M1_samples=samples_M1,
        M2_samples=samples_M2,
        M3_samples=samples_M3,
        data_M1=data_M1,
        data_M2=data_M2,
        data_M3=data_M3,
        t1_sparse=t1_sparse,
        t2_sparse=t2_sparse,
        t3_sparse=t3_sparse,
        idx1=idx1,
        idx2=idx2,
        idx3=idx3,
        theta_M1_mean=theta_M1_mean,
        theta_M2_mean=theta_M2_mean,
        theta_M3_mean=theta_M3_mean,
        theta_final=theta_final,
        tmcmc_M1=res_M1,
        tmcmc_M2=res_M2,
        tmcmc_M3=res_M3,
    )