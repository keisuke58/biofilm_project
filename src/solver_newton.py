# src/solver_newton.py
import numpy as np

from .numerics import (
    compute_Q_vector_numba,
    compute_jacobian_numba,
)
from .progress import ProgressTracker

HAS_NUMBA = True  # 環境に合わせて切り替え（例外処理を足してもOK）


class BiofilmNewtonSolver:
    """
    Newton solver with configurable initial conditions
    （re_numba.py のものをほぼそのまま移植）
    """
    THETA_NAMES = ["a11","a12","a22","b1","b2",
                   "a33","a34","a44","b3","b4",
                   "a13","a14","a23","a24"]

    def __init__(self, dt=1e-5, maxtimestep=2500, eps=1e-6, Kp1=1e-4,
                 eta_vec=None, c_const=100.0, alpha_const=100.0,
                 phi_init=0.02, use_numba=True, active_species=None):
        """
        Initialize BiofilmNewtonSolver.

        Parameters
        ----------
        phi_init : float
            Scalar initial volume fraction used for all four species.
            Two-species behavior is enforced via `active_species` masking,
            not by zeroing initial conditions.
        active_species : list of int, optional
            Indices of active species (0-3). If provided, inactive species
            interaction parameters will be zeroed out and their states held
            at `phi_init` during the Newton solve.
            - M1: active_species=[0, 1] for species 1-2
            - M2: active_species=[2, 3] for species 3-4
            - M3: active_species=None or [0,1,2,3] for all species
        """
        self.dt = dt
        self.maxtimestep = maxtimestep
        self.eps = eps
        self.Kp1 = Kp1
        self.Eta_vec = np.ones(4) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = self.Eta_vec.copy()
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)

        # Support scalar or 4-vector phi_init (2-species masking is controlled via active_species)
        if np.isscalar(phi_init):
            self.phi_init = np.full(4, float(phi_init))
        else:
            self.phi_init = np.asarray(phi_init, dtype=float)
            if self.phi_init.shape != (4,):
                raise ValueError("phi_init must be a scalar or length-4 vector")
        self.active_species = active_species
        self.use_numba = use_numba and HAS_NUMBA

    def c(self, t): return self.c_const
    def alpha(self, t): return self.alpha_const

    # θ → A, b の変換は re_numba.py の実装をコピペ
    def theta_to_matrices(self, theta):
        """
        Convert parameter vector to interaction matrix A and growth vector b.

        If active_species is specified, inactive species parameters are zeroed out.
        This enables true 2-species submodels (M1: species 1-2, M2: species 3-4).
        """
        theta = np.asarray(theta, dtype=float)
        a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24 = theta
        A = np.array([
            [a11, a12, a13, a14],
            [a12, a22, a23, a24],
            [a13, a23, a33, a34],
            [a14, a24, a34, a44]
        ], dtype=float)
        b_diag = np.array([b1, b2, b3, b4], dtype=float)

        # Zero out inactive species interactions for 2-species submodels
        if self.active_species is not None:
            inactive = [i for i in range(4) if i not in self.active_species]
            for i in inactive:
                A[i, :] = 0.0  # Row i: species i interactions
                A[:, i] = 0.0  # Col i: interactions with species i
                b_diag[i] = 0.0  # Growth rate of species i

        return A, b_diag

    # 初期状態
    def get_initial_state(self):
        """
        Get initial state vector [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma].

        For 2-species submodels, all species start at the same `phi_init`,
        and inactive species are kept fixed via masking during iteration.
        """
        phi_vec = np.maximum(self.phi_init, 1e-8)
        phi0 = 1.0 - np.sum(phi_vec)
        psi_vec = np.array([0.999] * 4)
        gamma = 1e-6
        g0 = np.zeros(10)
        g0[0:4] = phi_vec
        g0[4] = phi0
        g0[5:9] = psi_vec
        g0[9] = gamma
        return g0

    # Q（numpy版）: re_numba.py からコピペ
    def _compute_Q_vector_numpy(self, g_new, g_old, t, dt, A, b_diag):
        phi_new, phi0_new, psi_new, gamma_new = g_new[0:4], g_new[4], g_new[5:9], g_new[9]
        phi_old, phi0_old, psi_old = g_old[0:4], g_old[4], g_old[5:9]
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt
        Q = np.zeros(10)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        c_val = self.c(t)
        term1_phi = (self.Kp1 * (2.0 - 4.0 * phi_new)) / (np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
        term2_phi = (1.0 / self.Eta_vec) * (gamma_new + (self.Eta_phi_vec + self.Eta_vec * psi_new**2) * phidot +
                                             self.Eta_vec * phi_new * psi_new * psidot)
        term3_phi = (c_val / self.Eta_vec) * psi_new * Interaction
        Q[0:4] = term1_phi + term2_phi - term3_phi
        Q[4] = gamma_new + (self.Kp1 * (2.0 - 4.0 * phi0_new)) / (np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)) + phi0dot
        term1_psi = (-2.0 * self.Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - \
                    (2.0 * self.Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2))
        term2_psi = (b_diag * self.alpha(t) / self.Eta_vec) * psi_new
        term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
        term4_psi = (c_val / self.Eta_vec) * phi_new * Interaction
        Q[5:9] = term1_psi + term2_psi + term3_psi - term4_psi
        Q[9] = np.sum(phi_new) + phi0_new - 1.0
        return Q

    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return compute_Q_vector_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag
            )
        else:
            return self._compute_Q_vector_numpy(g_new, g_old, t, dt, A, b_diag)

    # K（numpy版）: re_numba.py の _compute_Jacobian_numpy をコピペ
    def _compute_Jacobian_numpy(self, g_new, g_old, t, dt, A, b_diag):
        v = g_new
        phi_new, phi0_new, psi_new = g_new[0:4], g_new[4], g_new[5:9]
        phidot = (phi_new - g_old[0:4]) / dt
        psidot = (psi_new - g_old[5:9]) / dt
        c_val = self.c(t)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        K = np.zeros((10, 10))
        
        phi_p_deriv = (self.Kp1*(-4. + 8.*v[0:4]))/(np.power(v[0:4],3)*np.power(v[0:4]-1.,3)) - \
                      (self.Kp1*(2. - 4.*v[0:4]))*(3./(np.power(v[0:4],4)*np.power(v[0:4]-1.,3)) +
                                                   3./(np.power(v[0:4],3)*np.power(v[0:4]-1.,4)))
        phi0_p_deriv = (self.Kp1*(-4. + 8.*v[4]))/(np.power(v[4],3)*np.power(v[4]-1.,3)) - \
                       (self.Kp1*(2. - 4.*v[4]))*(3./(np.power(v[4],4)*np.power(v[4]-1.,3)) +
                                                  3./(np.power(v[4],3)*np.power(v[4]-1.,4)))
        psi_p_deriv = (4.0 * self.Kp1 * (3.0 - 5.0*v[5:9] + 5.0*v[5:9]**2)) / \
                      (np.power(v[5:9], 4) * np.power(v[5:9] - 1.0, 4))
        
        for i in range(4):
            for j in range(4):
                K[i, j] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / self.Eta_vec[i]) * (
                (self.Eta_phi_vec[i] + self.Eta_vec[i] * psi_new[i]**2) / dt +
                self.Eta_vec[i] * psi_new[i] * psidot[i]) - \
                (c_val / self.Eta_vec[i]) * (psi_new[i] * (Interaction[i] + A[i, i] * psi_new[i]))
            K[i, 4] = 0.0
            for j in range(4):
                K[i, j+5] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i+5] = (1.0 / self.Eta_vec[i]) * (
                2.0 * self.Eta_vec[i] * psi_new[i] * phidot[i] +
                self.Eta_vec[i] * phi_new[i] * psidot[i] +
                self.Eta_vec[i] * phi_new[i] * psi_new[i] / dt) - \
                (c_val / self.Eta_vec[i]) * ((Interaction[i] + A[i, i] * phi_new[i] * psi_new[i]) +
                                              psi_new[i] * (A[i, i] * phi_new[i]))
            K[i, 9] = 1.0 / self.Eta_vec[i]
        
        K[4, 4] = phi0_p_deriv + 1.0/dt
        K[4, 9] = 1.0
        
        for i in range(4):
            k = i + 5
            for j in range(4):
                K[k, j] = -(c_val / self.Eta_vec[i]) * (A[i, j] * psi_new[j] * phi_new[i] +
                           Interaction[i] * (1.0 if i == j else 0.0))
            K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt +
                       2.0 * phi_new[i] * psidot[i]) - \
                      (c_val / self.Eta_vec[i]) * (A[i, i] * psi_new[i] * phi_new[i] +
                                                    Interaction[i] + phi_new[i] * A[i, i] * psi_new[i])
            K[k, 4] = 0.0
            for j in range(4):
                K[k, j+5] = -(c_val / self.Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
            K[k, i+5] = psi_p_deriv[i] + (b_diag[i] * self.alpha(t) / self.Eta_vec[i]) + \
                        (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                        (c_val / self.Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
            K[k, 9] = 0.0
        
        K[9, 0:5] = 1.0
        return K

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return compute_jacobian_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag
            )
        else:
            return self._compute_Jacobian_numpy(g_new, g_old, t, dt, A, b_diag)

    # 前進シミュレーション（re_numba の run_deterministic とほぼ同じ）
    def run_deterministic(self, theta, show_progress=False):
        A, b_diag = self.theta_to_matrices(theta)
        dt, maxtimestep, eps = self.dt, self.maxtimestep, self.eps
        g_prev = self.get_initial_state()
        t_list, g_list = [0.0], [g_prev.copy()]

        pbar = ProgressTracker(maxtimestep, "Forward sim") if show_progress else None

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            g_new = g_prev.copy()
            for _ in range(100):
                Q = self.compute_Q_vector(g_new, g_prev, tt, dt, A, b_diag)
                K = self.compute_Jacobian_matrix(g_new, g_prev, tt, dt, A, b_diag)
                if np.isnan(Q).any() or np.isnan(K).any():
                    raise RuntimeError(f"NaN at t={tt}")
                dg = np.linalg.solve(K, -Q)
                g_new = g_new + dg

                # Safeguards: prevent phi/psi from approaching zero
                phi_min = 1e-8
                psi_min = 1e-8
                g_new[0:4] = np.maximum(g_new[0:4], phi_min)
                g_new[4] = np.maximum(g_new[4], phi_min)
                g_new[5:9] = np.maximum(g_new[5:9], psi_min)

                # Enforce inactive species staying at initial values
                if self.active_species is not None:
                    inactive = [i for i in range(4) if i not in self.active_species]
                    for i in inactive:
                        g_new[i] = max(self.phi_init[i], phi_min)
                        g_new[i+5] = g_prev[i+5]
                if np.max(np.abs(Q)) < eps:
                    break
            g_prev = g_new.copy()
            t_list.append(tt)
            g_list.append(g_new.copy())

            if pbar and step % 100 == 0:
                pbar.set(step)

        if pbar:
            pbar.close()

        return np.array(t_list), np.vstack(g_list)
