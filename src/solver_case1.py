"""Case I: Table 1 two-species solver.

This module implements the paper's Case I settings (Table 1) with
species 1–2 only. It mirrors the reference residual formulation but
keeps the API lightweight for quick simulations and data generation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root


class BiofilmSolverCase1Table:
    """Two-species Newton solver using Table 1 defaults.

    Parameters
    ----------
    dt : float, optional
        Time-step size (default 1e-4 as in Table 1).
    N_steps : int, optional
        Number of time-steps (default 1000, giving t_end = 0.1).
    Kp1 : float, optional
        Penalty coefficient from Table 1.
    eta_vec : array-like, optional
        Per-species viscosity overrides. Defaults to Table 1 values
        ``[1.0, 2.0]``.
    """

    def __init__(self, dt: float = 1e-4, N_steps: int = 1000, Kp1: float = 1e-4,
                 eta_vec: np.ndarray | None = None) -> None:
        self.dt = dt
        self.N_steps = N_steps
        self.t_end = self.dt * self.N_steps
        self.Kp1 = Kp1

        self.Eta_vec = np.array([1.0, 2.0]) if eta_vec is None else np.asarray(eta_vec, dtype=float)
        if self.Eta_vec.shape != (2,):
            raise ValueError("eta_vec must be length 2 for the two-species Case I model")
        self.Eta_phi_vec = self.Eta_vec.copy()

        # Precompute normalized time grid for convenience
        raw_time = np.linspace(0.0, self.t_end, self.N_steps + 1)
        self.time_grid = raw_time / self.t_end

    def c(self, t: float) -> float:
        """Nutrient concentration (Table 1 constant)."""
        return 100.0

    def run(self, params: np.ndarray, N_data: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Run the two-species simulation and return sampled trajectories.

        Parameters
        ----------
        params : array-like
            Case I parameter vector ``[a11, a12, a22, b1, b2]``.
        N_data : int, optional
            Number of evenly sampled observations to return (default 20).

        Returns
        -------
        t_sample : np.ndarray
            Normalized sample times in ``[0, 1]`` (exclusive of 0, <=1).
        traj_data : np.ndarray
            Sampled ``phi1, phi2`` trajectories with shape ``(N_data, 2)``.
        """
        params = np.asarray(params, dtype=float)
        if params.shape != (5,):
            raise ValueError("params must be length 5: [a11, a12, a22, b1, b2]")

        p_a11, p_a12, p_a22, p_b1, p_b2 = params
        val_alpha = 10.0  # Antibiotics (Table 1)

        A = np.array([[p_a11, p_a12], [p_a12, p_a22]], dtype=float)
        b_diag = np.array([p_b1, p_b2], dtype=float)

        # Initial state (Table 1): phi1=0.25, phi2=0.30, psi≈1.0
        phi1_init = 0.25
        phi2_init = 0.30
        phi0_init = 1.0 - (phi1_init + phi2_init)
        g_current = np.array([
            phi1_init,
            phi2_init,
            phi0_init,
            0.999,
            0.999,
            1e-3,
        ])

        sample_indices = np.linspace(0, self.N_steps, N_data, endpoint=False, dtype=int)
        sample_lookup = set(sample_indices.tolist())
        traj_data = []
        t_sample = []

        for step in range(self.N_steps):
            current_time = (step + 1) * self.dt

            sol = root(
                self._residual_func,
                g_current,
                args=(g_current, self.dt, current_time, A, b_diag, val_alpha),
                method="lm",
                tol=1e-6,
            )
            if not sol.success:
                sol = root(
                    self._residual_func,
                    g_current,
                    args=(g_current, self.dt, current_time, A, b_diag, val_alpha),
                    method="hybr",
                    tol=1e-6,
                )
                if not sol.success:
                    raise RuntimeError(f"Newton solve failed at step {step}: {sol.message}")

            g_new = sol.x
            g_new[0:3] = np.clip(g_new[0:3], 1e-6, 1.0 - 1e-6)
            g_new[3:5] = np.clip(g_new[3:5], 0.1, 5.0)
            g_current = g_new.copy()

            if step in sample_lookup:
                traj_data.append(g_current[0:2].copy())
                t_sample.append(current_time / self.t_end)

        return np.array(t_sample), np.array(traj_data)

    def _residual_func(self, g_new: np.ndarray, g_old: np.ndarray, dt: float, t: float,
                       A: np.ndarray, b_diag: np.ndarray, val_alpha: float) -> np.ndarray:
        phi = g_new[0:2]
        phi0 = g_new[2]
        psi = g_new[3:5]
        gamma = g_new[5]

        phidot = (phi - g_old[0:2]) / dt
        phi0dot = (phi0 - g_old[2]) / dt
        psidot = (psi - g_old[3:5]) / dt

        Q = np.zeros(6)
        c_val = self.c(t)
        CapitalPhi = phi * psi
        interaction_dot = A @ CapitalPhi

        denom_phi = np.sign((phi - 1) ** 3 * phi ** 3) * np.maximum(np.abs((phi - 1) ** 3 * phi ** 3), 1e-12)
        Q[0:2] = (
            (self.Kp1 * (2.0 - 4.0 * phi)) / denom_phi
            + (1.0 / self.Eta_vec)
            * (gamma + (self.Eta_phi_vec + self.Eta_vec * psi**2) * phidot + self.Eta_vec * phi * psi * psidot)
            - (c_val / self.Eta_vec) * psi * interaction_dot
        )

        denom_phi0 = np.sign((phi0 - 1) ** 3 * phi0 ** 3) * np.maximum(np.abs((phi0 - 1) ** 3 * phi0 ** 3), 1e-12)
        Q[2] = gamma + (self.Kp1 * (2.0 - 4.0 * phi0)) / denom_phi0 + phi0dot

        denom_psiA = np.sign((psi - 1) ** 2 * psi ** 3) * np.maximum(np.abs((psi - 1) ** 2 * psi ** 3), 1e-12)
        denom_psiB = np.sign((psi - 1) ** 3 * psi ** 2) * np.maximum(np.abs((psi - 1) ** 3 * psi ** 2), 1e-12)
        Q[3:5] = (
            (-2.0 * self.Kp1) / denom_psiA
            - (2.0 * self.Kp1) / denom_psiB
            + (b_diag * val_alpha / self.Eta_vec) * psi
            + phi * psi * phidot
            + phi**2 * psidot
            - (c_val / self.Eta_vec) * phi * interaction_dot
        )

        Q[5] = np.sum(phi) + phi0 - 1.0
        return Q
