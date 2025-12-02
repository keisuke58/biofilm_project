# main_simulation.py
from src.config import CONFIG, get_theta_true
from src.solver_newton import BiofilmNewtonSolver

def main():
    config = CONFIG
    theta_true = get_theta_true()

    # M1/M2/M3 solver を作って単純に走らせるだけ
    solver_M1 = BiofilmNewtonSolver(
        phi_init=config["phi_init_M1"], use_numba=True, **config["M1"]
    )
    t1, g1 = solver_M1.run_deterministic(theta_true, show_progress=True)

    print("Simulation finished.")
    print(f"M1: t.shape={t1.shape}, g.shape={g1.shape}")

if __name__ == "__main__":
    main()
