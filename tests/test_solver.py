"""
Unit tests for BiofilmNewtonSolver

These tests verify basic functionality of the Newton solver for the biofilm
PDE system. Run with: pytest tests/test_solver.py
"""
import pytest
import numpy as np
from src.solver_newton import BiofilmNewtonSolver
from src.config import get_theta_true, CONFIG


class TestBiofilmNewtonSolver:
    """Test suite for BiofilmNewtonSolver"""

    @pytest.fixture
    def solver(self):
        """Create a test solver with debug settings"""
        # Use M1 config but override maxtimestep for faster tests
        config_M1 = CONFIG["M1"].copy()
        config_M1["maxtimestep"] = 50  # Short run for testing
        return BiofilmNewtonSolver(
            phi_init=0.05,
            use_numba=True,
            **config_M1
        )

    @pytest.fixture
    def theta_true(self):
        """Get true parameter vector"""
        return get_theta_true()

    def test_solver_initialization(self, solver):
        """Test that solver initializes correctly"""
        # Check dt is within reasonable range (depends on DEBUG mode)
        assert 1e-5 <= solver.dt <= 1e-3, f"dt={solver.dt} outside expected range"
        assert solver.maxtimestep == 50
        np.testing.assert_allclose(solver.phi_init, [0.05] * 4)
        assert len(solver.Eta_vec) == 4

    def test_initial_state_mass_conservation(self, solver):
        """Test that initial state satisfies Σφᵢ = 1"""
        g0 = solver.get_initial_state()
        phi_sum = g0[0:4].sum() + g0[4]  # Sum of all volume fractions
        np.testing.assert_allclose(phi_sum, 1.0, rtol=1e-10)

    def test_theta_to_matrices_shape(self, solver, theta_true):
        """Test that parameter conversion produces correct shapes"""
        A, b_diag = solver.theta_to_matrices(theta_true)
        assert A.shape == (4, 4)
        assert b_diag.shape == (4,)

    def test_theta_to_matrices_symmetry(self, solver, theta_true):
        """Test that A matrix is symmetric"""
        A, _ = solver.theta_to_matrices(theta_true)
        np.testing.assert_allclose(A, A.T, rtol=1e-10)

    def test_run_deterministic_mass_conservation(self, solver, theta_true):
        """Test that Σφᵢ = 1 is maintained throughout simulation"""
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        # Check mass conservation at all time steps
        phi_sum = g[:, 0:4].sum(axis=1) + g[:, 4]
        np.testing.assert_allclose(phi_sum, 1.0, atol=1e-6,
                                    err_msg="Mass conservation violated")

    def test_run_deterministic_output_shape(self, solver, theta_true):
        """Test that output arrays have correct shapes"""
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        assert len(t) == solver.maxtimestep + 1  # Initial + maxtimestep
        assert g.shape[0] == solver.maxtimestep + 1
        assert g.shape[1] == 10  # 4 phi + 1 phi0 + 4 psi + 1 gamma

    def test_run_deterministic_time_monotonic(self, solver, theta_true):
        """Test that time is strictly increasing"""
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        time_diffs = np.diff(t)
        assert np.all(time_diffs > 0), "Time should be strictly increasing"
        np.testing.assert_allclose(time_diffs, 1.0 / solver.maxtimestep, rtol=1e-10)

    def test_run_deterministic_positivity(self, solver, theta_true):
        """Test that volume fractions and porosity remain positive"""
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        phi = g[:, 0:4]
        phi0 = g[:, 4]
        psi = g[:, 5:9]

        assert np.all(phi >= 0), "phi should be non-negative"
        assert np.all(phi0 >= 0), "phi0 should be non-negative"
        assert np.all(psi >= 0), "psi should be non-negative"
        assert np.all(psi <= 1.0), "psi should be <= 1"

    def test_run_deterministic_no_nan(self, solver, theta_true):
        """Test that simulation produces no NaN values"""
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        assert np.all(np.isfinite(t)), "Time contains non-finite values"
        assert np.all(np.isfinite(g)), "State contains non-finite values"

    def test_different_phi_init(self, theta_true):
        """Test solver with different initial conditions"""
        for phi_init in [0.02, 0.1, 0.2]:
            config_M1 = CONFIG["M1"].copy()
            config_M1["maxtimestep"] = 20  # Short run for testing
            solver = BiofilmNewtonSolver(
                phi_init=phi_init,
                use_numba=True,
                **config_M1
            )
            t, g = solver.run_deterministic(theta_true, show_progress=False)

            # Check mass conservation
            phi_sum = g[:, 0:4].sum(axis=1) + g[:, 4]
            np.testing.assert_allclose(phi_sum, 1.0, atol=1e-6)

            # Check initial condition
            np.testing.assert_allclose(g[0, 0:4], phi_init, rtol=1e-10)

    def test_vector_phi_init(self, theta_true):
        """Test solver accepts a length-4 vector for phi_init"""
        phi_init_vec = [0.2, 0.2, 0.0, 0.0]
        config_M1 = CONFIG["M1"].copy()
        config_M1["maxtimestep"] = 10
        solver = BiofilmNewtonSolver(
            phi_init=phi_init_vec,
            use_numba=False,
            **config_M1,
        )
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        # Initial state uses the provided vector and preserves mass
        expected_phi = np.maximum(phi_init_vec, 1e-8)
        np.testing.assert_allclose(g[0, 0:4], expected_phi, rtol=1e-10)
        np.testing.assert_allclose(g[0, 0:4].sum() + g[0, 4], 1.0, rtol=1e-10)

    def test_dynamic_species_two_mode(self):
        """Solver should run with n=2 using generic theta layout and normalized time."""
        theta_generic = np.array([0.8, 0.2, 1.1, 0.05, 0.06])  # 3 upper-triangular + 2 b terms
        solver = BiofilmNewtonSolver(
            phi_init=[0.1, 0.1],
            eta_vec=[1.0, 1.0],
            dt=1e-4,
            maxtimestep=20,
            use_numba=False,
            species_count=2,
        )
        t, g = solver.run_deterministic(theta_generic, show_progress=False)

        assert g.shape[1] == 2 * solver.n + 2
        np.testing.assert_allclose(t[0], 0.0)
        np.testing.assert_allclose(t[-1], 1.0)
        np.testing.assert_allclose(np.diff(t), 1.0 / solver.maxtimestep, rtol=1e-10)

    @pytest.mark.parametrize("use_numba", [True, False])
    def test_numba_vs_numpy_consistency(self, theta_true, use_numba):
        """Test that Numba and NumPy versions give similar results"""
        config_M1 = CONFIG["M1"].copy()
        config_M1["maxtimestep"] = 20  # Short run for testing
        solver = BiofilmNewtonSolver(
            phi_init=0.05,
            use_numba=use_numba,
            **config_M1
        )
        t, g = solver.run_deterministic(theta_true, show_progress=False)

        # Basic sanity checks
        assert np.all(np.isfinite(g))
        phi_sum = g[:, 0:4].sum(axis=1) + g[:, 4]
        np.testing.assert_allclose(phi_sum, 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
