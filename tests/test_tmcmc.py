"""
Unit tests for TMCMC (Transitional Markov Chain Monte Carlo)

Run with: pytest tests/test_tmcmc.py
"""
import pytest
import numpy as np
from src.tmcmc import tmcmc, TMCMCResult
from src.config import get_theta_true, CONFIG


def simple_log_likelihood(theta):
    """
    Simple 2D Gaussian likelihood for testing.
    p(data | θ) ∝ exp(-0.5 * ||θ - θ_true||²)

    Parameters
    ----------
    theta : np.ndarray
        Single parameter vector (d,) or array of vectors (N, d)

    Returns
    -------
    float or np.ndarray
        Log-likelihood value(s)
    """
    theta_true = np.array([1.0, 2.0])
    if theta.ndim == 1:
        diff = theta - theta_true
        return -0.5 * np.sum(diff**2)
    else:
        diff = theta - theta_true
        return -0.5 * np.sum(diff**2, axis=1)


def simple_log_prior(theta):
    """
    Uniform prior on [-5, 5]² for testing

    Parameters
    ----------
    theta : np.ndarray
        Single parameter vector (d,) or array of vectors (N, d)

    Returns
    -------
    float or np.ndarray
        Log-prior value(s)
    """
    if theta.ndim == 1:
        if np.all((-5 <= theta) & (theta <= 5)):
            return 0.0
        else:
            return -np.inf
    else:
        valid = np.all((-5 <= theta) & (theta <= 5), axis=1)
        logp = np.full(len(theta), -np.inf)
        logp[valid] = 0.0
        return logp


def get_initial_samples(n_samples):
    """Generate initial samples from prior"""
    return np.random.uniform(-5, 5, size=(n_samples, 2))


class TestTMCMC:
    """Test suite for TMCMC algorithm"""

    def test_tmcmc_basic_convergence(self):
        """Test that TMCMC converges to posterior for simple 2D problem"""
        np.random.seed(42)
        theta_init = get_initial_samples(500)

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=20,
            target_ess_ratio=0.8,
            model_name="test_2D",
            show_progress=False,
        )

        # Should converge
        assert result.converged, "TMCMC did not converge"

        # Final samples should be near true value [1.0, 2.0]
        final_samples = result.samples[-1]
        mean_estimate = np.mean(final_samples, axis=0)

        # Allow generous tolerance for stochastic test
        np.testing.assert_allclose(mean_estimate, [1.0, 2.0], atol=0.5,
                                    err_msg="Posterior mean far from true value")

    def test_tmcmc_output_structure(self):
        """Test that TMCMC returns correct data structures"""
        np.random.seed(42)
        theta_init = get_initial_samples(200)

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=10,
            model_name="test_structure",
            show_progress=False,
        )

        # Check result attributes
        assert hasattr(result, "samples")
        assert hasattr(result, "beta_schedule")
        assert hasattr(result, "logL_trace")
        assert hasattr(result, "converged")

        # Check types
        assert isinstance(result.samples, list)
        assert isinstance(result.beta_schedule, list)
        assert isinstance(result.logL_trace, list)
        assert isinstance(result.converged, bool)

        # Check list lengths match
        n_stages = len(result.beta_schedule)
        assert len(result.samples) == n_stages
        assert len(result.logL_trace) == n_stages

        # Check sample shapes
        for samples in result.samples:
            assert samples.shape[0] == 200  # n_samples
            assert samples.shape[1] == 2    # n_params

    def test_tmcmc_beta_monotonic(self):
        """Test that β schedule is monotonically increasing to 1.0"""
        np.random.seed(42)
        theta_init = get_initial_samples(200)

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=15,
            model_name="test_beta",
            show_progress=False,
        )

        beta = np.array(result.beta_schedule)

        # Should start at 0
        assert beta[0] == 0.0, "Beta should start at 0"

        # Should be monotonically increasing
        assert np.all(np.diff(beta) >= 0), "Beta schedule not monotonic"

        # Should reach 1.0 if converged
        if result.converged:
            np.testing.assert_allclose(beta[-1], 1.0, rtol=1e-10,
                                        err_msg="Final beta ≠ 1.0")

    def test_tmcmc_ess_tracking(self):
        """Test that ESS is tracked throughout stages"""
        np.random.seed(42)
        theta_init = get_initial_samples(300)

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=20,
            target_ess_ratio=0.8,
            model_name="test_ess",
            show_progress=False,
        )

        # Check that ESS is tracked
        assert hasattr(result, 'ess_trace')
        # ESS trace has one less entry than beta_schedule (no ESS for initial β=0)
        assert len(result.ess_trace) == len(result.beta_schedule) - 1
        assert len(result.samples) > 1, "Should have multiple stages"

    def test_tmcmc_reproducibility(self):
        """Test that TMCMC is reproducible with same seed"""
        np.random.seed(42)
        theta_init1 = get_initial_samples(200)
        result1 = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init1,
            n_stages=5,
            model_name="test_repro",
            random_state=123,
            show_progress=False,
        )

        np.random.seed(42)
        theta_init2 = get_initial_samples(200)
        result2 = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init2,
            n_stages=5,
            model_name="test_repro",
            random_state=123,
            show_progress=False,
        )

        # Check that results match
        assert len(result1.beta_schedule) == len(result2.beta_schedule)
        np.testing.assert_allclose(result1.beta_schedule, result2.beta_schedule)

    def test_tmcmc_max_stages_limit(self):
        """Test that TMCMC respects max_stages limit"""
        np.random.seed(42)
        theta_init = get_initial_samples(200)
        max_stages = 5

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=max_stages,
            target_ess_ratio=0.99,  # High target may not converge quickly
            model_name="test_limit",
            show_progress=False,
        )

        # beta_schedule includes initial β=0, so length is n_stages + 1
        assert len(result.beta_schedule) <= max_stages + 1, \
            f"Exceeded max_stages: {len(result.beta_schedule)} > {max_stages + 1}"

    def test_tmcmc_positive_loglikelihood(self):
        """Test TMCMC with positive log-likelihood values"""
        def positive_log_likelihood(theta):
            """Likelihood that produces positive values"""
            if theta.ndim == 1:
                return 10.0 - 0.5 * np.sum(theta**2)
            else:
                return 10.0 - 0.5 * np.sum(theta**2, axis=1)

        np.random.seed(42)
        theta_init = get_initial_samples(200)

        result = tmcmc(
            log_likelihood=positive_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=10,
            model_name="test_positive",
            show_progress=False,
        )

        # Should still work correctly
        assert result.converged or len(result.beta_schedule) > 1

    def test_tmcmc_no_nan_values(self):
        """Test that TMCMC produces no NaN values"""
        np.random.seed(42)
        theta_init = get_initial_samples(200)

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=10,
            model_name="test_nan",
            show_progress=False,
        )

        # Check all samples for NaN
        for stage_samples in result.samples:
            assert np.all(np.isfinite(stage_samples)), \
                "Samples contain NaN or Inf"

        # Check beta schedule
        assert np.all(np.isfinite(result.beta_schedule)), \
            "Beta schedule contains NaN or Inf"

        # Check log-likelihood values
        for logL in result.logL_trace:
            assert np.all(np.isfinite(logL)), \
                "Log-likelihood contains NaN or Inf"


class TestTMCMCEdgeCases:
    """Test edge cases and error handling"""

    def test_tmcmc_with_flat_likelihood(self):
        """Test TMCMC when likelihood is constant (uninformative)"""
        def flat_log_likelihood(theta):
            """Constant likelihood - no information"""
            if theta.ndim == 1:
                return 0.0
            else:
                return np.zeros(len(theta))

        np.random.seed(42)
        theta_init = get_initial_samples(200)

        result = tmcmc(
            log_likelihood=flat_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=5,
            model_name="test_flat",
            show_progress=False,
        )

        # Should still run but may not converge
        assert len(result.samples) > 0

    def test_tmcmc_small_sample_size(self):
        """Test TMCMC with very small sample size"""
        np.random.seed(42)
        theta_init = get_initial_samples(50)  # Small

        result = tmcmc(
            log_likelihood=simple_log_likelihood,
            log_prior=simple_log_prior,
            theta_init_samples=theta_init,
            n_stages=5,
            model_name="test_small",
            show_progress=False,
        )

        # Should still work
        assert len(result.samples) > 0
        assert result.samples[-1].shape[0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
