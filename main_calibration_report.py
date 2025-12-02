# main_calibration.py
import time

from src.config import CONFIG, DEBUG, get_theta_true
from src.hierarchical import hierarchical_case2
from src.viz import BayesianVisualizer
from src.report import BayesianReport
from src.viz_paper import make_all_case2_figures
from src.posterior_simulator_tsm import generate_posterior_phi_tsm
from src.validation_m3 import generate_M3_validation
from src.utils_log import log  # タイムスタンプ付きログ

def main():
    # ============================================================
    #  ヘッダ
    # ============================================================
    log("=" * 72)
    log("Biofilm Case II: TSM + TMCMC + Hierarchical Bayesian Updating")
    log("MODULE STRUCTURE (re_numba-based)")
    log("=" * 72)
    log(f"DEBUG : {DEBUG}")
    log(f"Ndata : {CONFIG['Ndata']}, N0 = {CONFIG['N0']}")
    log("=" * 72)

    # 真のパラメータ
    theta_true = get_theta_true()
    log("θ_true = " + str(theta_true))

    # ============================================================
    #  階層ベイズ推定 (Case II: M1 -> M2 -> M3)
    # ============================================================
    log("Starting hierarchical_case2 ...")
    t_start = time.time()

    results = hierarchical_case2(CONFIG)

    total_time = time.time() - t_start
    log("hierarchical_case2 finished.")

    # ============================================================
    #  真値との比較
    # ============================================================
    log("=" * 72)
    log("FINAL RESULTS")
    log("=" * 72)
    log("True θ:      " + str(theta_true))
    log("Estimated θ: " + str(results.theta_final))
    err = results.theta_final - theta_true
    log("Error:       " + str(err))
    rmse = ((err ** 2).mean()) ** 0.5
    log(f"RMSE:        {rmse:.4f}")
    log(f"Total time:  {total_time:.1f} s")
    log(
        f"Convergence: M1={results.tmcmc_M1.converged}, "
        f"M2={results.tmcmc_M2.converged}, "
        f"M3={results.tmcmc_M3.converged}"
    )

    # ============================================================
    #  可視化用の真値・ラベルを設定
    # （ここで results にもくっつけておく）
    # ============================================================
    # M1: a11, a12, a22, b1, b2
    theta_true_M1 = theta_true[0:5]
    labels_M1 = [r"$a_{11}$", r"$a_{12}$", r"$a_{22}$", r"$b_{1}$", r"$b_{2}$"]

    # M2: a33, a34, a44, b3, b4
    theta_true_M2 = theta_true[5:10]
    labels_M2 = [r"$a_{33}$", r"$a_{34}$", r"$a_{44}$", r"$b_{3}$", r"$b_{4}$"]

    # M3: a13, a14, a23, a24
    theta_true_M3 = theta_true[10:14]
    labels_M3 = [r"$a_{13}$", r"$a_{14}$", r"$a_{23}$", r"$a_{24}$"]

    # results にも保存しておく（viz_paper などからも使えるように）
    results.theta_true_M1 = theta_true_M1
    results.theta_true_M2 = theta_true_M2
    results.theta_true_M3 = theta_true_M3
    results.labels_M1 = labels_M1
    results.labels_M2 = labels_M2
    results.labels_M3 = labels_M3
    results.theta_true_full = theta_true
    results.labels_full = labels_M1 + labels_M2 + labels_M3

    # ============================================================
    #  ① 通常の可視化 (viz.py)
    # ============================================================
    log("Generating standard figures (viz.py) ...")
    viz = BayesianVisualizer("results")

    # --- M1 ---
    viz.plot_posterior(results.tmcmc_M1.samples,
                       theta_true_M1,
                       labels_M1,
                       model_name="M1")
    viz.plot_trace(results.tmcmc_M1.samples,
                   labels_M1,
                   model_name="M1")
    viz.plot_beta_schedule(results.tmcmc_M1.beta_list, model_name="M1")
    viz.plot_logL(results.tmcmc_M1.logL_list, model_name="M1")

    # --- M2 ---
    viz.plot_posterior(results.tmcmc_M2.samples,
                       theta_true_M2,
                       labels_M2,
                       model_name="M2")
    viz.plot_trace(results.tmcmc_M2.samples,
                   labels_M2,
                   model_name="M2")
    viz.plot_beta_schedule(results.tmcmc_M2.beta_list, model_name="M2")
    viz.plot_logL(results.tmcmc_M2.logL_list, model_name="M2")

    # --- M3 ---
    viz.plot_posterior(results.tmcmc_M3.samples,
                       theta_true_M3,
                       labels_M3,
                       model_name="M3")
    viz.plot_trace(results.tmcmc_M3.samples,
                   labels_M3,
                   model_name="M3")
    viz.plot_beta_schedule(results.tmcmc_M3.beta_list, model_name="M3")
    viz.plot_logL(results.tmcmc_M3.logL_list, model_name="M3")

    # ============================================================
    #  ② PDF REPORT (report.py)
    # ============================================================
    log("Generating PDF report ...")
    report = BayesianReport("results")
    report.build_report("bayesian_report.pdf")

    # ============================================================
    #  ③ TSM による posterior time-series 生成
    #      → 論文 Fig.9,11,13 用
    # ============================================================
    log("Generating posterior time-series via TSM (M1/M2/M3) ...")

    # DEBUG のときは軽く、本番は多めに
    Ns_plot = 10 if DEBUG else 40

    # data_* は hierarchical_case2 側で
    # results.data_M1 / data_M2 / data_M3 に保存されている前提
    data_M1 = getattr(results, "data_M1", None)
    data_M2 = getattr(results, "data_M2", None)
    data_M3 = getattr(results, "data_M3", None)

    # --- M1 ---
    t_M1, phi_post_M1 = generate_posterior_phi_tsm(
        posterior_samples=results.tmcmc_M1.samples,
        CONFIG=CONFIG,
        model_id="M1",
        Ns=Ns_plot,
    )

    # --- M2 ---
    t_M2, phi_post_M2 = generate_posterior_phi_tsm(
        posterior_samples=results.tmcmc_M2.samples,
        CONFIG=CONFIG,
        model_id="M2",
        Ns=Ns_plot,
    )

    # --- M3 ---
    t_M3, phi_post_M3 = generate_posterior_phi_tsm(
        posterior_samples=results.tmcmc_M3.samples,
        CONFIG=CONFIG,
        model_id="M3",
        Ns=Ns_plot,
    )

    # ============================================================
    #  ④ M3 validation (posterior predictive check)
    #      → 論文 Fig.15 用
    # ============================================================
    log("Generating M3 validation via TSM ...")

    t_M3_val, phi_M3_val, phi_post_M3_val = generate_M3_validation(
        results,
        CONFIG,
        Ns=Ns_plot,
    )

    # validation data:
    # - results.data_M3_val があればそれを使う
    # - 無ければ data_M3 を代わりに使う
    data_M3_val = getattr(results, "data_M3_val", data_M3)

    # ============================================================
    #  ⑤ Paper-style 図 (viz_paper)
    #      → Fig.9, 11, 13, 15 相当
    # ============================================================
    log("Generating paper-style time-series figures (viz_paper) ...")

    make_all_case2_figures(
        results=results,
        time_M1=t_M1,
        phi_post_M1=phi_post_M1,
        data_M1=data_M1,
        time_M2=t_M2,
        phi_post_M2=phi_post_M2,
        data_M2=data_M2,
        time_M3=t_M3,
        phi_post_M3=phi_post_M3,
        data_M3=data_M3,
        time_M3_val=t_M3_val,
        phi_post_M3_val=phi_post_M3_val,
        data_M3_val=data_M3_val,
        outdir="results",
    )

    # ============================================================
    #  終了
    # ============================================================
    log("All tasks completed.")

if __name__ == "__main__":
    main()
