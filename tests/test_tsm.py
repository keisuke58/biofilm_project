"""
Unit tests for BiofilmTSM (Time-Separated Mechanics)

Run with: pytest tests/test_tsm.py
"""
import pytest
import numpy as np
from src.tsm import BiofilmTSM
from src.solver_newton import BiofilmNewtonSolver
from src.config import get_theta_true, CONFIG


class TestBiofilmTSM:
    """Test suite for BiofilmTSM"""

    @pytest.fixture
    def solver(self):
        """Create a test solver"""
        # Use M1 config but override maxtimestep for faster tests
        config_M1 = CONFIG["M1"].copy()
        config_M1["maxtimestep"] = 30  # Short for testing
        return BiofilmNewtonSolver(
            phi_init=0.05,
            use_numba=True,
            **config_M1
        )

    @pytest.fixture
    def tsm(self, solver):
        """Create a test TSM instance"""
        return BiofilmTSM(
            solver,
            cov_rel=0.01,
            active_theta_indices=[0, 1, 2, 3, 4],  # M1 parameters
            use_analytical=True
        )

    @pytest.fixture
    def theta_true(self):
        """Get true parameter vector"""
        return get_theta_true()

    def test_tsm_initialization(self, tsm):
        """Test that TSM initializes correctly"""
        assert tsm.cov_rel == 0.01
        assert len(tsm.active_idx) == 5
        assert tsm.use_analytical is True

    def test_solve_tsm_output_shape(self, tsm, theta_true):
        """Test that solve_tsm returns correct shapes"""
        result = tsm.solve_tsm(theta_true)

        n_time = tsm.solver.maxtimestep + 1
        n_state = 10
        n_active = len(tsm.active_idx)

        assert result.t_array.shape == (n_time,)
        assert result.mu.shape == (n_time, n_state)
        assert result.sigma2.shape == (n_time, n_state)
        assert result.x0.shape == (n_time, n_state)
        assert result.x1.shape == (n_time, n_state, n_active)

    def test_solve_tsm_positive_variance(self, tsm, theta_true):
        """Test that variance is always non-negative"""
        result = tsm.solve_tsm(theta_true)

        assert np.all(result.sigma2 >= 0), "Variance must be non-negative"

    def test_solve_tsm_mean_equals_deterministic(self, tsm, theta_true):
        """Test that mean trajectory matches deterministic solution"""
        result = tsm.solve_tsm(theta_true)

        # Run deterministic solver
        t_det, g_det = tsm.solver.run_deterministic(theta_true, show_progress=False)

        # TSM mean should match deterministic solution
        np.testing.assert_allclose(result.mu, g_det, rtol=1e-10,
                                    err_msg="TSM mean ≠ deterministic solution")

    def test_solve_tsm_no_nan(self, tsm, theta_true):
        """Test that TSM produces no NaN values"""
        result = tsm.solve_tsm(theta_true)

        assert np.all(np.isfinite(result.t_array))
        assert np.all(np.isfinite(result.mu))
        assert np.all(np.isfinite(result.sigma2))
        assert np.all(np.isfinite(result.x1))

    def test_variance_scales_with_cov_rel(self, solver, theta_true):
        """Test that variance scales quadratically with cov_rel"""
        cov_rel_1 = 0.01
        cov_rel_2 = 0.02

        tsm_1 = BiofilmTSM(solver, cov_rel=cov_rel_1,
                           active_theta_indices=[0, 1, 2], use_analytical=True)
        tsm_2 = BiofilmTSM(solver, cov_rel=cov_rel_2,
                           active_theta_indices=[0, 1, 2], use_analytical=True)

        result_1 = tsm_1.solve_tsm(theta_true)
        result_2 = tsm_2.solve_tsm(theta_true)

        # Variance should scale as (cov_rel)^2
        # Check mean variance ratio (more robust than element-wise)
        mean_var_1 = np.mean(result_1.sigma2)
        mean_var_2 = np.mean(result_2.sigma2)
        ratio = mean_var_2 / (mean_var_1 + 1e-12)
        expected_ratio = (cov_rel_2 / cov_rel_1) ** 2

        # Allow generous tolerance for statistical test
        assert abs(ratio - expected_ratio) / expected_ratio < 0.3, \
            f"Variance scaling ratio {ratio:.2f} differs from expected {expected_ratio:.2f}"

    def test_active_indices_subset(self, solver, theta_true):
        """Test TSM with different active parameter subsets"""
        # Test M1 parameters
        tsm_M1 = BiofilmTSM(solver, active_theta_indices=[0, 1, 2, 3, 4])
        result_M1 = tsm_M1.solve_tsm(theta_true)
        assert result_M1.x1.shape[2] == 5

        # Test M2 parameters
        tsm_M2 = BiofilmTSM(solver, active_theta_indices=[5, 6, 7, 8, 9])
        result_M2 = tsm_M2.solve_tsm(theta_true)
        assert result_M2.x1.shape[2] == 5

        # Test M3 parameters
        tsm_M3 = BiofilmTSM(solver, active_theta_indices=[10, 11, 12, 13])
        result_M3 = tsm_M3.solve_tsm(theta_true)
        assert result_M3.x1.shape[2] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
