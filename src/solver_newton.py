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
                 phi_init=0.02, use_numba=True, active_species=None,
                 species_count=None, theta_indices=None):
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
        # Species dimensionality
        if active_species is not None:
            inferred_species = len(active_species)
        elif np.isscalar(phi_init):
            inferred_species = 4 if species_count is None else species_count
        else:
            inferred_species = len(np.asarray(phi_init, dtype=float))
        self.n = species_count or inferred_species

        # Optional mapping that selects which five Case-II parameters belong to
        # a 2-species submodel (e.g., [0..4] for M1 or [5..9] for M2). When
        # provided, legacy 14-length theta vectors are sliced before building
        # A/b, ensuring that “absent” species never influence the dynamics.
        self.theta_indices = None if theta_indices is None else np.asarray(theta_indices, dtype=int)

        self.Eta_vec = np.ones(self.n) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        if self.Eta_vec.shape != (self.n,):
            raise ValueError(f"eta_vec must have length {self.n}")
        self.Eta_phi_vec = self.Eta_vec.copy()
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)

        # Support scalar or vector phi_init
        if np.isscalar(phi_init):
            self.phi_init = np.full(self.n, float(phi_init))
        else:
            self.phi_init = np.asarray(phi_init, dtype=float)
            if self.phi_init.shape != (self.n,):
                raise ValueError(f"phi_init must be a scalar or length-{self.n} vector")
        self.active_species = active_species
        self.use_numba = use_numba and HAS_NUMBA and self.n == 4

    def c(self, t): return self.c_const
    def alpha(self, t): return self.alpha_const

    # θ → A, b の変換は re_numba.py の実装をコピペ
    def theta_to_matrices(self, theta):
        """
        Convert parameter vector to interaction matrix A and growth vector b.

        Two formats are supported:
        - Legacy 4-species vector of length 14 (Case II parameterization)
        - Generic symmetric upper-triangular layout of length n(n+1)/2 + n

        If active_species is provided, the returned matrices are restricted to
        the active subset to enable true lower-order submodels.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape == (14,) and self.theta_indices is not None and self.n == 2:
            theta = theta[self.theta_indices]

        if theta.shape == (14,):
            a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24 = theta
            A_full = np.array([
                [a11, a12, a13, a14],
                [a12, a22, a23, a24],
                [a13, a23, a33, a34],
                [a14, a24, a34, a44]
            ], dtype=float)
            b_full = np.array([b1, b2, b3, b4], dtype=float)
        else:
            expected = self.n * (self.n + 1) // 2 + self.n
            if theta.size != expected:
                raise ValueError(
                    f"theta must have length 14 for legacy 4-species mode or {expected} for n={self.n}"
                )
            # When theta_indices is provided, we already sliced the legacy
            # vector above; here we simply unpack the generic symmetric layout
            # for arbitrary n.
            A_full = np.zeros((self.n, self.n), dtype=float)
            idx = 0
            for i in range(self.n):
                for j in range(i, self.n):
                    A_full[i, j] = theta[idx]
                    A_full[j, i] = theta[idx]
                    idx += 1
            b_full = theta[idx: idx + self.n]

        if self.active_species is not None:
            active_idx = np.array(self.active_species, dtype=int)
            A = A_full[np.ix_(active_idx, active_idx)]
            b_diag = b_full[active_idx]
        else:
            A = A_full
            b_diag = b_full

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
        psi_vec = np.array([0.999] * self.n)
        gamma = 1e-6
        g0 = np.zeros(2 * self.n + 2)
        g0[0:self.n] = phi_vec
        g0[self.n] = phi0
        g0[self.n + 1:self.n + 1 + self.n] = psi_vec
        g0[-1] = gamma
        return g0

    # Q（numpy版）: re_numba.py からコピペ
    def _compute_Q_vector_numpy(self, g_new, g_old, t, dt, A, b_diag):
        n = self.n
        phi_new, phi0_new = g_new[0:n], g_new[n]
        psi_new, gamma_new = g_new[n + 1:n + 1 + n], g_new[-1]
        phi_old, phi0_old = g_old[0:n], g_old[n]
        psi_old = g_old[n + 1:n + 1 + n]

        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt
        Q = np.zeros(2 * n + 2)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        c_val = self.c(t)
        term1_phi = (self.Kp1 * (2.0 - 4.0 * phi_new)) / (np.power(phi_new - 1.0, 3) * np.power(phi_new, 3))
        term2_phi = (1.0 / self.Eta_vec) * (
            gamma_new + (self.Eta_phi_vec + self.Eta_vec * psi_new**2) * phidot +
            self.Eta_vec * phi_new * psi_new * psidot
        )
        term3_phi = (c_val / self.Eta_vec) * psi_new * Interaction
        Q[0:n] = term1_phi + term2_phi - term3_phi
        Q[n] = gamma_new + (self.Kp1 * (2.0 - 4.0 * phi0_new)) / (
            np.power(phi0_new - 1.0, 3) * np.power(phi0_new, 3)
        ) + phi0dot
        term1_psi = (-2.0 * self.Kp1) / (np.power(psi_new - 1.0, 2) * np.power(psi_new, 3)) - (
            2.0 * self.Kp1) / (np.power(psi_new - 1.0, 3) * np.power(psi_new, 2)
        )
        term2_psi = (b_diag * self.alpha(t) / self.Eta_vec) * psi_new
        term3_psi = phi_new * psi_new * phidot + phi_new**2 * psidot
        term4_psi = (c_val / self.Eta_vec) * phi_new * Interaction
        Q[n + 1:n + 1 + n] = term1_psi + term2_psi + term3_psi - term4_psi
        Q[-1] = np.sum(phi_new) + phi0_new - 1.0
        return Q

    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return compute_Q_vector_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag
            )
        return self._compute_Q_vector_numpy(g_new, g_old, t, dt, A, b_diag)

    # K（numpy版）: re_numba.py の _compute_Jacobian_numpy をコピペ
    def _compute_Jacobian_numpy(self, g_new, g_old, t, dt, A, b_diag):
        n = self.n
        v = g_new
        phi_new, phi0_new, psi_new = g_new[0:n], g_new[n], g_new[n + 1:n + 1 + n]
        phidot = (phi_new - g_old[0:n]) / dt
        psidot = (psi_new - g_old[n + 1:n + 1 + n]) / dt
        c_val = self.c(t)
        CapitalPhi = phi_new * psi_new
        Interaction = A @ CapitalPhi
        K = np.zeros((2 * n + 2, 2 * n + 2))

        phi_p_deriv = (self.Kp1 * (-4. + 8. * v[0:n])) / (np.power(v[0:n], 3) * np.power(v[0:n] - 1., 3)) - (
            self.Kp1 * (2. - 4. * v[0:n])
        ) * (
            3. / (np.power(v[0:n], 4) * np.power(v[0:n] - 1., 3)) +
            3. / (np.power(v[0:n], 3) * np.power(v[0:n] - 1., 4))
        )
        phi0_p_deriv = (self.Kp1 * (-4. + 8. * v[n])) / (np.power(v[n], 3) * np.power(v[n] - 1., 3)) - (
            self.Kp1 * (2. - 4. * v[n])
        ) * (
            3. / (np.power(v[n], 4) * np.power(v[n] - 1., 3)) +
            3. / (np.power(v[n], 3) * np.power(v[n] - 1., 4))
        )
        psi_p_deriv = (4.0 * self.Kp1 * (3.0 - 5.0 * v[n + 1:n + 1 + n] + 5.0 * v[n + 1:n + 1 + n]**2)) / (
            np.power(v[n + 1:n + 1 + n], 4) * np.power(v[n + 1:n + 1 + n] - 1.0, 4)
        )

        for i in range(n):
            for j in range(n):
                K[i, j] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
            K[i, i] = phi_p_deriv[i] + (1.0 / self.Eta_vec[i]) * (
                (self.Eta_phi_vec[i] + self.Eta_vec[i] * psi_new[i]**2) / dt +
                self.Eta_vec[i] * psi_new[i] * psidot[i]
            ) - (c_val / self.Eta_vec[i]) * (psi_new[i] * (Interaction[i] + A[i, i] * psi_new[i]))
            K[i, n] = 0.0
            for j in range(n):
                K[i, j + n + 1] = (c_val / self.Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
            K[i, i + n + 1] = (1.0 / self.Eta_vec[i]) * (
                2.0 * self.Eta_vec[i] * psi_new[i] * phidot[i] +
                self.Eta_vec[i] * phi_new[i] * psidot[i] +
                self.Eta_vec[i] * phi_new[i] * psi_new[i] / dt
            ) - (c_val / self.Eta_vec[i]) * (
                (Interaction[i] + A[i, i] * phi_new[i] * psi_new[i]) + psi_new[i] * (A[i, i] * phi_new[i])
            )
            K[i, -1] = 1.0 / self.Eta_vec[i]

        K[n, n] = phi0_p_deriv + 1.0 / dt
        K[n, -1] = 1.0

        for i in range(n):
            k = i + n + 1
            for j in range(n):
                K[k, j] = -(c_val / self.Eta_vec[i]) * (
                    A[i, j] * psi_new[j] * phi_new[i] + Interaction[i] * (1.0 if i == j else 0.0)
                )
            K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt + 2.0 * phi_new[i] * psidot[i]) - (
                c_val / self.Eta_vec[i]
            ) * (A[i, i] * psi_new[i] * phi_new[i] + Interaction[i] + phi_new[i] * A[i, i] * psi_new[i])
            K[k, n] = 0.0
            for j in range(n):
                K[k, j + n + 1] = -(c_val / self.Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
            K[k, i + n + 1] = psi_p_deriv[i] + (b_diag[i] * self.alpha(t) / self.Eta_vec[i]) + (
                phi_new[i] * phidot[i] + phi_new[i]**2 / dt
            ) - (c_val / self.Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
            K[k, -1] = 0.0

        K[-1, 0:n + 1] = 1.0
        return K

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba:
            return compute_jacobian_numba(
                g_new[0:4], g_new[4], g_new[5:9], g_new[9],
                g_old[0:4], g_old[5:9],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c(t), self.alpha(t), A, b_diag
            )
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
                g_new[0:self.n] = np.maximum(g_new[0:self.n], phi_min)
                g_new[self.n] = np.maximum(g_new[self.n], phi_min)
                g_new[self.n + 1:self.n + 1 + self.n] = np.maximum(
                    g_new[self.n + 1:self.n + 1 + self.n], psi_min
                )

                # Enforce inactive species staying at initial values
                if self.active_species is not None:
                    inactive = [i for i in range(self.n) if i not in self.active_species]
                    for i in inactive:
                        g_new[i] = max(self.phi_init[i], phi_min)
                        g_new[self.n + 1 + i] = g_prev[self.n + 1 + i]
                if np.max(np.abs(Q)) < eps:
                    break
            g_prev = g_new.copy()
            t_list.append((step + 1) / maxtimestep)
            g_list.append(g_new.copy())

            if pbar and step % 100 == 0:
                pbar.set(step)

        if pbar:
            pbar.close()

        return np.array(t_list), np.vstack(g_list)
