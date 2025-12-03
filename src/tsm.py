# src/tsm.py
import numpy as np
from dataclasses import dataclass

from .numerics import dQ_dtheta_analytical_numba, sigma2_accumulate_numba
from .progress import ProgressTracker

HAS_NUMBA = True  # 必要なら try/except で切り替え

@dataclass
class TSMResult:
    t_array: np.ndarray
    mu: np.ndarray
    sigma2: np.ndarray
    x0: np.ndarray
    x1: np.ndarray


class BiofilmTSM:
    """
    Taylor Series Method (1st order) with ANALYTICAL SENSITIVITY
    （biofilm_ultimate / re_numba の実装をモジュール化）
    """
    THETA_NAMES = ["a11","a12","a22","b1","b2",
                   "a33","a34","a44","b3","b4",
                   "a13","a14","a23","a24"]

    def __init__(self, solver, cov_rel=0.005,
                 active_theta_indices=None, use_analytical=True):
        self.solver = solver
        self.cov_rel = cov_rel
        self.active_idx = np.arange(14) if active_theta_indices is None else np.array(active_theta_indices)
        self.use_analytical = use_analytical and HAS_NUMBA

    # dQ/dθ（analytical）: biofilm_ultimate の _dG_dtheta_analytical / _dQ_dtheta_numpy をここに集約
    def _dQ_dtheta_numpy(self, phi_new, psi_new, c_val, alpha_val, Eta_vec, CapitalPhi, theta_idx):
        """NumPy fallback for analytical sensitivity"""
        dQ = np.zeros(10)
        
        # Parameter index mapping (same logic as Numba version)
        if theta_idx == 0:  # a11
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[0]
        elif theta_idx == 1:  # a12
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[1]
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[0]
        elif theta_idx == 2:  # a22
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[1]
        elif theta_idx == 3:  # b1
            dQ[5] = (alpha_val / Eta_vec[0]) * psi_new[0]
        elif theta_idx == 4:  # b2
            dQ[6] = (alpha_val / Eta_vec[1]) * psi_new[1]
        elif theta_idx == 5:  # a33
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[2]
        elif theta_idx == 6:  # a34
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[2]
        elif theta_idx == 7:  # a44
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[3]
        elif theta_idx == 8:  # b3
            dQ[7] = (alpha_val / Eta_vec[2]) * psi_new[2]
        elif theta_idx == 9:  # b4
            dQ[8] = (alpha_val / Eta_vec[3]) * psi_new[3]
        elif theta_idx == 10:  # a13
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[0]
        elif theta_idx == 11:  # a14
            dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[0]
            dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[0]
        elif theta_idx == 12:  # a23
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[2]
            dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[2]
            dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[1]
        elif theta_idx == 13:  # a24
            dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[3]
            dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[1]
            dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[3]
            dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[1]
        
        return dQ

    def _dG_dtheta_analytical(self, g_new, theta):
        phi_new, psi_new = g_new[0:4], g_new[5:9]
        CapitalPhi = phi_new * psi_new
        c_val = self.solver.c_const
        alpha_val = self.solver.alpha_const
        Eta_vec = self.solver.Eta_vec

        dG_dict = {}
        for idx in self.active_idx:
            if HAS_NUMBA:
                dQ = dQ_dtheta_analytical_numba(
                    phi_new, psi_new, c_val, alpha_val, Eta_vec, CapitalPhi, idx
                )
            else:
                dQ = self._dQ_dtheta_numpy(phi_new, psi_new, c_val, alpha_val,
                                           Eta_vec, CapitalPhi, idx)
            dG_dict[self.THETA_NAMES[idx]] = dQ
        return dG_dict

    # 数値差分版 dG/dθ（必要なら re_numba の _dG_dtheta_numeric をここに移植）
    def _dG_dtheta_numeric(self, g_new, g_old, t, dt, theta):
        dG_dict = {}
        for idx in self.active_idx:
            th_plus, th_minus = theta.copy(), theta.copy()
            eps_theta = 1e-6 * max(1.0, abs(theta[idx]))
            th_plus[idx] += eps_theta
            th_minus[idx] -= eps_theta
            A_p, b_p = self.solver.theta_to_matrices(th_plus)
            A_m, b_m = self.solver.theta_to_matrices(th_minus)
            Q_p = self.solver.compute_Q_vector(g_new, g_old, t, dt, A_p, b_p)
            Q_m = self.solver.compute_Q_vector(g_new, g_old, t, dt, A_m, b_m)
            dG_dict[self.THETA_NAMES[idx]] = (Q_p - Q_m) / (2.0 * eps_theta)
        return dG_dict

    # メインの solve_tsm: re_numba / biofilm_ultimate の BiofilmTSM.solve_tsm をそのまま移植
    def solve_tsm(self, theta):
        """
        ここに BiofilmTSM.solve_tsm の本体をコピペしてください。
        中で:
          - Newton ステップで g を更新
          - dG/dθ を解いて x1 を計算
          - var_theta_active / sigma2_accumulate_numba で σ² を求める
        """
        """
        Solve TSM with analytical or numerical sensitivity
        
        x0 = g(t; θ)           - deterministic solution
        x1 = ∂g/∂θ             - sensitivity (1st order Taylor)
        σ² = Σ_k (x1_k)² Var(θ_k)  - propagated variance
        """
        theta = np.asarray(theta, dtype=float)
        A, b_diag = self.solver.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.solver.dt, self.solver.maxtimestep, self.solver.eps

        # g_prev = np.array([0.02, 0.02, 0.02, 0.02, 0.92, 0.999, 0.999, 0.999, 0.999, 1e-6])
        g_prev = self.solver.get_initial_state()

        t_list, x0_list = [0.0], [g_prev.copy()]
        theta_dim = len(self.active_idx)
        x1_list = [np.zeros((10, theta_dim))]

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            g_new = g_prev.copy()
            
            # Newton iteration
            for _ in range(100):
                Q = self.solver.compute_Q_vector(g_new, g_prev, tt, dt, A, b_diag)
                K = self.solver.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN at t={tt}")
                dg = np.linalg.solve(K, -Q)
                g_new = g_new + dg
                if np.max(np.abs(Q)) < eps:
                    break

            # Compute sensitivity ∂g/∂θ
            if self.use_analytical:
                dG_dict = self._dG_dtheta_analytical(g_new, theta)
            else:
                dG_dict = self._dG_dtheta_numeric(g_new, g_prev, tt, dt, theta)
            
            J = self.solver.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
            x1_t = np.zeros((10, theta_dim))
            for k, idx in enumerate(self.active_idx):
                x1_t[:, k] = np.linalg.solve(J, -dG_dict[self.THETA_NAMES[idx]])

            g_prev = g_new.copy()
            t_list.append(tt)
            x0_list.append(g_prev.copy())
            x1_list.append(x1_t)

        t_array = np.array(t_list)
        x0 = np.vstack(x0_list)
        x1 = np.stack(x1_list, axis=0)

        # Compute variance propagation
        var_theta_full = (self.cov_rel * theta)**2
        var_theta_active = var_theta_full[self.active_idx]

        mu = x0.copy()
        if HAS_NUMBA:
            sigma2 = sigma2_accumulate_numba(x1, var_theta_active)
        else:
            sigma2 = np.zeros_like(mu) + 1e-12
            for k in range(theta_dim):
                sigma2 += (x1[:, :, k]**2) * var_theta_active[k]

        return TSMResult(t_array=t_array, mu=mu, sigma2=sigma2, x0=x0, x1=x1)
        # return TSMResult(t_array=t_array, mu=mu, sigma2=sigma2, x0=x0, x1=x1)


# ─────────────────────────────────────────────
# Likelihood: sparse data / Eq.29
# ─────────────────────────────────────────────

def log_likelihood_sparse(obs, obs_var, data, sigma_obs):
    """Gaussian heteroscedastic log-likelihood for sparse observations.

    The total variance combines propagated state uncertainty ``obs_var``
    (from :class:`BiofilmTSM.solve_tsm`) with measurement noise
    ``sigma_obs``. ``sigma_obs`` may be a scalar or an array broadcastable
    to ``obs`` so each observation can be weighted by its own variance.
    """

    obs_arr = np.asarray(obs, dtype=float)
    data_arr = np.asarray(data, dtype=float)
    prop_var = np.asarray(obs_var, dtype=float)

    if data_arr.shape != obs_arr.shape:
        raise ValueError(f"shape mismatch: data={data_arr.shape}, mu={obs_arr.shape}")
    if prop_var.shape != obs_arr.shape:
        raise ValueError(f"shape mismatch: obs_var={prop_var.shape}, obs={obs_arr.shape}")

    meas_var = np.asarray(sigma_obs, dtype=float)
    try:
        total_var = prop_var + np.square(np.broadcast_to(meas_var, obs_arr.shape))
    except ValueError:
        raise ValueError(f"sigma_obs with shape {meas_var.shape} not broadcastable to {obs_arr.shape}")

    v = total_var.ravel()
    D = data_arr.ravel()
    m = obs_arr.ravel()

    bad = (~np.isfinite(v)) | (v <= 0)
    if np.any(bad):
        return -1e20

    diff = D - m

    ll = -0.5 * np.sum(np.log(2 * np.pi * v)) - 0.5 * np.sum(diff * diff / v)

    return max(ll, -1e20) if np.isfinite(ll) else -1e20


def compute_tsm_sensitivity(theta, config, active_theta_indices=None, use_analytical=True):
    """
    Compute the TSM sensitivity matrix for a given parameter vector.

    Parameters
    ----------
    theta : array_like
        Parameter vector for the active subset of parameters.
    config : dict
        Configuration dictionary containing solver settings such as
        ``dt``, ``maxtimestep``, ``c_const``, and ``alpha_const``.
    active_theta_indices : list[int], optional
        Indices (0-based) of the active parameters within the full 14-D
        parameter vector. If omitted, the first ``len(theta)`` entries
        are used.
    use_analytical : bool
        Whether to use analytical sensitivity (Numba-backed) when available.

    Returns
    -------
    np.ndarray
        Sensitivity matrix of shape ``(n_outputs, n_parameters)`` where
        ``n_outputs`` corresponds to the flattened state trajectory over
        time and species.
    """

    from .solver_newton import BiofilmNewtonSolver

    theta = np.asarray(theta, dtype=float)
    active_idx = (list(range(len(theta))) if active_theta_indices is None
                  else list(active_theta_indices))

    # Build full 14-dimensional theta vector
    theta_full = np.zeros(14, dtype=float)
    theta_full[active_idx] = theta

    # Instantiate solver with sensible defaults pulled from config
    solver = BiofilmNewtonSolver(
        dt=config.get("dt", 1e-5),
        maxtimestep=config.get("maxtimestep", 2500),
        eps=config.get("eps", 1e-6),
        Kp1=config.get("Kp1", 1e-4),
        eta_vec=config.get("eta_vec"),
        c_const=config.get("c_const", 100.0),
        alpha_const=config.get("alpha_const", 100.0),
        phi_init=config.get("phi_init", config.get("phi_init_M1", 0.02)),
        use_numba=config.get("use_numba", True),
    )

    tsm = BiofilmTSM(
        solver,
        cov_rel=config.get("cov_rel", 0.005),
        active_theta_indices=active_idx,
        use_analytical=use_analytical,
    )

    result = tsm.solve_tsm(theta_full)

    # Flatten (time, state) dimensions to align with finite-difference checks
    n_outputs = result.x1.shape[0] * result.x1.shape[1]
    return result.x1.reshape(n_outputs, result.x1.shape[2])
