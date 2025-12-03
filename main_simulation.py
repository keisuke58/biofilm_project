# main_simulation.py
from pathlib import Path

import numpy as np

from src.config import CONFIG, get_theta_true
from src.solver_newton import BiofilmNewtonSolver

def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    config = CONFIG
    theta_true = get_theta_true()

    # M1/M2/M3 solver を作って単純に走らせるだけ
    solver_M1 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M1"], use_numba=True, **config["M1"]
    )
    t1, g1 = solver_M1.run_deterministic(theta_true, show_progress=True)

    np.savez(output_dir / "forward_simulation_timeseries.npz", t=t1, g=g1)

    num_species = g1.shape[1] if g1.ndim > 1 else 1
    header_columns = ["time"] + [f"phi{i+1}" for i in range(num_species)]
    np.savetxt(
        output_dir / "forward_simulation_timeseries.csv",
        np.column_stack([t1, g1]),
        delimiter=",",
        header=",".join(header_columns),
        comments="",
    )

    print("Simulation finished.")
    print(f"M1: t.shape={t1.shape}, g.shape={g1.shape}")
    print(f"Saved timeseries to {output_dir}/forward_simulation_timeseries.(npz,csv)")

if __name__ == "__main__":
    main()
