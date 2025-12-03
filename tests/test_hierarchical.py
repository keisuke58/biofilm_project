"""
Unit tests for hierarchical Bayesian updating

Run with: pytest tests/test_hierarchical.py -v
"""
import copy
import pytest
import numpy as np
from src.hierarchical import (
    hierarchical_case2,
    HierarchicalResults,
    _normalize_phi_init,
)


def test_normalize_phi_init_scalar_to_active_subset():
    vec = _normalize_phi_init(0.2, 2, [0, 1])
    np.testing.assert_allclose(vec, [0.2, 0.2])


def test_normalize_phi_init_short_vector_pad_and_slice():
    vec = _normalize_phi_init([0.1], 2, [1])
    np.testing.assert_allclose(vec, [0.0])
from src.config import get_theta_true, CONFIG


@pytest.mark.slow
class TestHierarchicalCase2:
    """
    Test suite for hierarchical_case2 function.

    Note: These are marked as slow tests since hierarchical inference
    takes significant time. Run with: pytest -m slow
    """

    @pytest.fixture
    def debug_config(self):
        """Create a fast debug configuration for testing"""
        config = copy.deepcopy(CONFIG)
        config["DEBUG"] = True
        config["Ndata"] = 5
        config["N0"] = 50
        config["N_resamples"] = 1
        config["stages"] = 3

        # Reduce timesteps for speed
        for model_id in ["M1", "M2", "M3"]:
            config[model_id]["maxtimestep"] = 15
            config[model_id]["dt"] = 2e-3

        return config

    def test_hierarchical_output_structure(self, debug_config):
        """Test that hierarchical_case2 returns correct structure"""
        results = hierarchical_case2(debug_config)

        # Check return type
        assert isinstance(results, HierarchicalResults)

        # Check required attributes
        assert hasattr(results, "M1_samples")
        assert hasattr(results, "M2_samples")
        assert hasattr(results, "M3_samples")
        assert hasattr(results, "theta_final")
        assert hasattr(results, "theta_M1_mean")
        assert hasattr(results, "theta_M2_mean")
        assert hasattr(results, "theta_M3_mean")
        assert hasattr(results, "theta_M1_map")
        assert hasattr(results, "theta_M2_map")
        assert hasattr(results, "theta_M3_map")
        assert hasattr(results, "theta_final_map")
        assert hasattr(results, "tmcmc_M1")
        assert hasattr(results, "tmcmc_M2")
        assert hasattr(results, "tmcmc_M3")
        assert hasattr(results, "data_M1")
        assert hasattr(results, "data_M2")
        assert hasattr(results, "data_M3")
        assert hasattr(results, "t1_sparse")
        assert hasattr(results, "t2_sparse")
        assert hasattr(results, "t3_sparse")
        assert hasattr(results, "idx1")
        assert hasattr(results, "idx2")
        assert hasattr(results, "idx3")

    def test_hierarchical_theta_dimensions(self, debug_config):
        """Test that estimated parameters have correct dimensions"""
        results = hierarchical_case2(debug_config)

        # M1: 5 parameters (a11, a12, a22, b1, b2)
        assert results.theta_M1_mean.shape == (5,)
        assert results.M1_samples.shape[1] == 5

        # M2: 5 parameters (a33, a34, a44, b3, b4)
        assert results.theta_M2_mean.shape == (5,)
        assert results.M2_samples.shape[1] == 5

        # M3: 4 parameters (a13, a14, a23, a24)
        assert results.theta_M3_mean.shape == (4,)
        assert results.M3_samples.shape[1] == 4

        # Final: 14 parameters total
        assert results.theta_final.shape == (14,)

    def test_hierarchical_theta_assembly(self, debug_config):
        """Test that theta_final is correctly assembled from MAP estimates"""
        results = hierarchical_case2(debug_config)

        # Check that theta_final contains M1, M2, M3 MAP estimates in correct order
        np.testing.assert_allclose(results.theta_final[0:5], results.theta_M1_map)
        np.testing.assert_allclose(results.theta_final[5:10], results.theta_M2_map)
        np.testing.assert_allclose(results.theta_final[10:14], results.theta_M3_map)
        np.testing.assert_allclose(results.theta_final, results.theta_final_map)

    def test_hierarchical_convergence_flags(self, debug_config):
        """Test that convergence flags are boolean"""
        results = hierarchical_case2(debug_config)

        assert isinstance(results.tmcmc_M1.converged, bool)
        assert isinstance(results.tmcmc_M2.converged, bool)
        assert isinstance(results.tmcmc_M3.converged, bool)

    def test_hierarchical_data_storage(self, debug_config):
        """Test that observational data is stored in results"""
        results = hierarchical_case2(debug_config)

        # Check that data is stored
        # Check data shapes and time indices
        Ndata = debug_config["Ndata"]
        assert results.data_M1.shape == (Ndata, 2)  # 2 species for M1
        assert results.data_M2.shape == (Ndata, 2)  # 2 species for M2
        assert results.data_M3.shape == (Ndata, 4)  # 4 species for M3
        assert results.t1_sparse.shape[0] == Ndata
        assert results.t2_sparse.shape[0] == Ndata
        assert results.t3_sparse.shape[0] == Ndata
        assert results.idx1.shape[0] == Ndata
        assert results.idx2.shape[0] == Ndata
        assert results.idx3.shape[0] == Ndata

    def test_hierarchical_no_nan(self, debug_config):
        """Test that hierarchical updating produces no NaN values"""
        results = hierarchical_case2(debug_config)

        # Check final estimates
        assert np.all(np.isfinite(results.theta_final))
        assert np.all(np.isfinite(results.theta_M1_mean))
        assert np.all(np.isfinite(results.theta_M2_mean))
        assert np.all(np.isfinite(results.theta_M3_mean))

        # Check samples
        assert np.all(np.isfinite(results.M1_samples))
        assert np.all(np.isfinite(results.M2_samples))
        assert np.all(np.isfinite(results.M3_samples))

    def test_hierarchical_sample_sizes(self, debug_config):
        """Test that sample sizes match configuration"""
        N0 = debug_config["N0"]
        results = hierarchical_case2(debug_config)

        # Final stage samples should have N0 samples
        assert results.M1_samples.shape[0] == N0
        assert results.M2_samples.shape[0] == N0
        assert results.M3_samples.shape[0] == N0


class TestHierarchicalResultsDataclass:
    """Test HierarchicalResults dataclass"""

    def test_hierarchical_results_creation(self):
        """Test that HierarchicalResults can be instantiated"""
        # Create mock data
        M1_samples = np.random.randn(100, 5)
        M2_samples = np.random.randn(100, 5)
        M3_samples = np.random.randn(100, 4)
        data_M1 = np.random.randn(5, 2)
        data_M2 = np.random.randn(5, 2)
        data_M3 = np.random.randn(5, 4)
        t1_sparse = np.linspace(0, 1, 5)
        t2_sparse = np.linspace(0, 1, 5)
        t3_sparse = np.linspace(0, 1, 5)
        idx = np.arange(5)
        theta_final = np.random.randn(14)

        results = HierarchicalResults(
            M1_samples=M1_samples,
            M2_samples=M2_samples,
            M3_samples=M3_samples,
            data_M1=data_M1,
            data_M2=data_M2,
            data_M3=data_M3,
            t1_sparse=t1_sparse,
            t2_sparse=t2_sparse,
            t3_sparse=t3_sparse,
            idx1=idx,
            idx2=idx,
            idx3=idx,
            theta_final=theta_final,
            theta_M1_mean=theta_final[0:5],
            theta_M2_mean=theta_final[5:10],
            theta_M3_mean=theta_final[10:14],
            theta_M1_map=theta_final[0:5],
            theta_M2_map=theta_final[5:10],
            theta_M3_map=theta_final[10:14],
            theta_final_map=theta_final,
            tmcmc_M1=None,
            tmcmc_M2=None,
            tmcmc_M3=None,
        )

        assert isinstance(results, HierarchicalResults)
        np.testing.assert_array_equal(results.M1_samples, M1_samples)
        np.testing.assert_array_equal(results.theta_final, theta_final)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
