# src/viz.py
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 12,
    "axes.grid": True
})

class BayesianVisualizer:
    """
    通常の Matplotlib を使った可視化。
    ・posterior ヒストグラム
    ・traceplot
    ・beta スケジュール
    ・log-likelihood 推移
    """

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Posterior histogram
    # ------------------------------------------------------------
    def plot_posterior(self, samples: np.ndarray, true_values: np.ndarray,
                       labels: List[str], model_name="M1"):
        n_params = samples.shape[1]
        fig, axes = plt.subplots(n_params, 1, figsize=(6, 2 * n_params))

        if n_params == 1:
            axes = [axes]

        for i in range(n_params):
            ax = axes[i]
            ax.hist(samples[:, i], bins=30, alpha=0.7, color='steelblue')
            ax.axvline(true_values[i], color='red', linestyle='--', label="True")
            ax.set_title(f"{model_name}: Posterior of {labels[i]}")
            ax.legend()

        fig.tight_layout()
        filepath = os.path.join(self.results_dir, f"{model_name}_posterior.png")
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Saved: {filepath}")

    # ------------------------------------------------------------
    # Traceplot
    # ------------------------------------------------------------
    def plot_trace(self, samples: np.ndarray, labels: List[str], model_name="M1"):
        n_params = samples.shape[1]
        fig, axes = plt.subplots(n_params, 1, figsize=(6, 2 * n_params))

        if n_params == 1:
            axes = [axes]

        for i in range(n_params):
            ax = axes[i]
            ax.plot(samples[:, i], alpha=0.8)
            ax.set_title(f"{model_name}: Trace of {labels[i]}")

        fig.tight_layout()
        filepath = os.path.join(self.results_dir, f"{model_name}_trace.png")
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Saved: {filepath}")

    # ------------------------------------------------------------
    # β schedule
    # ------------------------------------------------------------
    def plot_beta_schedule(self, beta_list: List[float], model_name="M1"):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(beta_list, marker='o')
        plt.xlabel("Stage")
        plt.ylabel("Beta")
        plt.title(f"{model_name}: Beta schedule")
        filepath = os.path.join(self.results_dir, f"{model_name}_beta.png")
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filepath}")

    # ------------------------------------------------------------
    # log-likelihood curve
    # ------------------------------------------------------------
    def plot_logL(self, logL_list: List[float], model_name="M1"):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(logL_list, marker='o')
        plt.xlabel("Stage")
        plt.ylabel("Mean log-likelihood")
        plt.title(f"{model_name}: logL progression")
        filepath = os.path.join(self.results_dir, f"{model_name}_logL.png")
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filepath}")
