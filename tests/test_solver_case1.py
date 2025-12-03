import numpy as np
import pytest

from src.config import get_theta_true
from src.solver_case1 import BiofilmSolverCase1Table


def test_case1_table_run_produces_two_species_samples():
    solver = BiofilmSolverCase1Table(N_steps=200)
    theta = get_theta_true()
    params = theta[:5]

    t_sample, traj = solver.run(params)

    assert traj.shape == (20, 2)
    assert t_sample.shape == (20,)
    assert np.all((t_sample > 0.0) & (t_sample <= 1.0))
    assert np.all(np.diff(t_sample) > 0.0)
    assert np.all(traj > 0.0)
    assert np.all(traj < 1.0)
    assert np.all(traj.sum(axis=1) < 1.0)


def test_case1_table_validates_eta_length():
    with pytest.raises(ValueError):
        BiofilmSolverCase1Table(eta_vec=[1.0])
