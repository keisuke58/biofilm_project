# main_calibration.py
import time

from src.config import CONFIG, DEBUG, get_theta_true
from src.hierarchical import hierarchical_case2

def main():
    print("="*72)
    print("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    print("  MODULE STRUCTURE (re_numba-based)")
    print("="*72)
    print(f"DEBUG : {DEBUG}")
    print(f"Ndata : {CONFIG['Ndata']}, N0 = {CONFIG['N0']}")
    print("="*72)

    theta_true = get_theta_true()
    print("\n[True Parameters]")
    print("  θ_true =", theta_true)

    t_start = time.time()
    results = hierarchical_case2(CONFIG)
    total_time = time.time() - t_start

    print("\n" + "="*72)
    print("  FINAL RESULTS")
    print("="*72)
    print("True θ:     ", theta_true)
    print("Estimated θ:", results.theta_final)
    print("Error:      ", results.theta_final - theta_true)
    print(f"RMSE:       {(( (results.theta_final - theta_true)**2 ).mean())**0.5:.4f}")
    print(f"Total time: {total_time:.1f} s")
    print(f"Convergence: M1={results.tmcmc_M1.converged}, "
          f"M2={results.tmcmc_M2.converged}, M3={results.tmcmc_M3.converged}")

if __name__ == "__main__":
    main()
