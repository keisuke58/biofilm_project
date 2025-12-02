# main_calibration.py
import time
import logging

from src.config import CONFIG, DEBUG, get_theta_true
from src.hierarchical import hierarchical_case2
from src.logger import setup_logger

def main():
    # Setup logging
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logger = setup_logger(level=log_level, log_file="results/biofilm_calibration.log")

    logger.info("="*72)
    logger.info("  Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    logger.info("  MODULE STRUCTURE (re_numba-based)")
    logger.info("="*72)
    logger.info(f"DEBUG : {DEBUG}")
    logger.info(f"Ndata : {CONFIG['Ndata']}, N0 = {CONFIG['N0']}")
    logger.info("="*72)

    theta_true = get_theta_true()
    logger.info("")
    logger.info("[True Parameters]")
    logger.info(f"  θ_true = {theta_true}")

    logger.info("Starting hierarchical calibration...")
    t_start = time.time()
    results = hierarchical_case2(CONFIG)
    total_time = time.time() - t_start

    logger.info("")
    logger.info("="*72)
    logger.info("  FINAL RESULTS")
    logger.info("="*72)
    logger.info(f"True θ:      {theta_true}")
    logger.info(f"Estimated θ: {results.theta_final}")
    logger.info(f"Error:       {results.theta_final - theta_true}")
    logger.info(f"RMSE:        {(( (results.theta_final - theta_true)**2 ).mean())**0.5:.4f}")
    logger.info(f"Total time:  {total_time:.1f} s")
    logger.info(f"Convergence: M1={results.tmcmc_M1.converged}, "
                f"M2={results.tmcmc_M2.converged}, M3={results.tmcmc_M3.converged}")
    logger.info("Calibration completed successfully!")

if __name__ == "__main__":
    main()
