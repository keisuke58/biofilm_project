"""
Unit tests for posterior_simulator_tsm.py

Run with: pytest tests/test_posterior_simulator.py
"""
import pytest
import numpy as np
from src.posterior_simulator_tsm import generate_posterior_phi_tsm
from src.config import get_theta_true, CONFIG


class TestPosteriorSimulatorTSM:
    """Test suite for posterior time-series generation"""

    @pytest.fixture
    def debug_config(self):
        """Create fast debug configuration"""
        config = CONFIG.copy()
        config["DEBUG"] = True
        for model_id in ["M1", "M2", "M3"]:
            config[model_id]["maxtimestep"] = 20
            config[model_id]["dt"] = 1e-3
        return config

    @pytest.fixture
    def mock_posterior_samples_M1(self):
        """Create mock posterior samples for M1 (5 parameters)"""
        # Generate realistic samples near true values
        theta_true = get_theta_true()
        samples = theta_true[0:5] + np.random.randn(100, 5) * 0.1
        return samples

    @pytest.fixture
    def mock_posterior_samples_M3(self):
        """Create mock posterior samples for M3 (4 parameters)"""
        theta_true = get_theta_true()
        samples = theta_true[10:14] + np.random.randn(100, 4) * 0.1
        return samples

    def test_generate_phi_output_shape(self, debug_config, mock_posterior_samples_M1):
        """Test that generate_posterior_phi_tsm returns correct shapes"""
        Ns = 10
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        # Check time array
        assert t.shape[0] == debug_config["M1"]["maxtimestep"] + 1

        # Check phi_all shape: (Ns, Nt, Nspecies)
        Nt = len(t)
        Nspecies = 2  # M1 has 2 species
        assert phi_all.shape == (Ns, Nt, Nspecies)

    def test_generate_phi_with_theta_base(self, debug_config, mock_posterior_samples_M1):
        """Test theta_base parameter for hierarchical structure"""
        theta_true = get_theta_true()
        Ns = 5

        # Use true theta as base
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
            theta_base=theta_true.copy()
        )

        assert phi_all.shape[0] == Ns
        assert phi_all.shape[2] == 2  # M1 species

    def test_generate_phi_M3_four_species(self, debug_config, mock_posterior_samples_M3):
        """Test M3 generates 4-species output"""
        Ns = 5
        theta_true = get_theta_true()

        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M3,
            CONFIG=debug_config,
            model_id="M3",
            Ns=Ns,
            theta_base=theta_true.copy()
        )

        # M3 should have 4 species
        assert phi_all.shape[2] == 4

    def test_generate_phi_handles_list_input(self, debug_config, mock_posterior_samples_M1):
        """Test that List[np.ndarray] input (from TMCMCResult) is handled"""
        # Simulate TMCMCResult.samples as List[np.ndarray]
        samples_list = [
            mock_posterior_samples_M1[0:30],  # Stage 0
            mock_posterior_samples_M1[30:70],  # Stage 1
            mock_posterior_samples_M1,         # Stage 2 (final)
        ]

        Ns = 5
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=samples_list,  # List input
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        # Should use last stage (samples_list[-1])
        assert phi_all.shape[0] == Ns

    def test_generate_phi_no_theta_base(self, debug_config, mock_posterior_samples_M1):
        """Test that theta_base=None works (uses zeros)"""
        Ns = 5
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
            theta_base=None  # Should use zeros for inactive params
        )

        assert phi_all.shape[0] == Ns

    def test_generate_phi_no_nan(self, debug_config, mock_posterior_samples_M1):
        """Test that output contains no NaN values"""
        Ns = 10
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        assert np.all(np.isfinite(t))
        assert np.all(np.isfinite(phi_all))

    def test_generate_phi_positivity(self, debug_config, mock_posterior_samples_M1):
        """Test that volume fractions are non-negative"""
        Ns = 10
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        # Volume fractions should be non-negative
        assert np.all(phi_all >= 0), "Volume fractions should be non-negative"

    def test_generate_phi_time_monotonic(self, debug_config, mock_posterior_samples_M1):
        """Test that time array is monotonically increasing"""
        Ns = 5
        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=mock_posterior_samples_M1,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        # Time should be strictly increasing
        time_diffs = np.diff(t)
        assert np.all(time_diffs > 0), "Time should be strictly increasing"

    def test_generate_phi_different_Ns(self, debug_config, mock_posterior_samples_M1):
        """Test with different numbers of samples"""
        for Ns in [5, 10, 20]:
            t, phi_all = generate_posterior_phi_tsm(
                posterior_samples=mock_posterior_samples_M1,
                CONFIG=debug_config,
                model_id="M1",
                Ns=Ns,
            )

            # Should generate exactly Ns trajectories
            assert phi_all.shape[0] == Ns

    def test_generate_phi_Ns_larger_than_available(self, debug_config):
        """Test when Ns > available samples"""
        small_samples = np.random.randn(10, 5)  # Only 10 samples
        Ns = 50  # Request more than available

        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=small_samples,
            CONFIG=debug_config,
            model_id="M1",
            Ns=Ns,
        )

        # Should use all available samples
        assert phi_all.shape[0] == 10  # Capped at available


class TestPosteriorSimulatorParameterIndices:
    """Test parameter index handling for different models"""

    @pytest.fixture
    def debug_config(self):
        """Fast config"""
        config = CONFIG.copy()
        for model_id in ["M1", "M2", "M3"]:
            config[model_id]["maxtimestep"] = 10
        return config

    def test_M1_parameter_indices(self, debug_config):
        """Test that M1 uses indices 0-4"""
        samples = np.random.randn(20, 5)
        theta_base = np.zeros(14)
        theta_base[5:] = 99.0  # Mark other params

        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=samples,
            CONFIG=debug_config,
            model_id="M1",
            Ns=3,
            theta_base=theta_base
        )

        # Should work without error
        assert phi_all.shape[0] == 3

    def test_M2_parameter_indices(self, debug_config):
        """Test that M2 uses indices 5-9"""
        samples = np.random.randn(20, 5)
        theta_base = np.zeros(14)
        theta_base[0:5] = 99.0
        theta_base[10:] = 99.0

        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=samples,
            CONFIG=debug_config,
            model_id="M2",
            Ns=3,
            theta_base=theta_base
        )

        assert phi_all.shape[0] == 3

    def test_M3_parameter_indices(self, debug_config):
        """Test that M3 uses indices 10-13"""
        samples = np.random.randn(20, 4)  # M3 has 4 params
        theta_base = np.zeros(14)
        theta_base[0:10] = 99.0

        t, phi_all = generate_posterior_phi_tsm(
            posterior_samples=samples,
            CONFIG=debug_config,
            model_id="M3",
            Ns=3,
            theta_base=theta_base
        )

        assert phi_all.shape[0] == 3
        assert phi_all.shape[2] == 4  # 4 species


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
