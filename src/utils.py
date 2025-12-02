# src/utils.py
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_biofilm_trajectories(
    t: np.ndarray,
    g: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = False,
):
    """
    g = [phi(1..N), phi0, psi(1..N), gamma] の時間変化をプロット
    """
    N = (g.shape[1] - 2) // 2  # φ と ψ の種数

    phi = g[:, 0:N]
    phi0 = g[:, N]
    psi = g[:, N + 1 : 2 * N + 1]
    gamma = g[:, -1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # φ
    for i in range(N):
        ax1.plot(t, phi[:, i], label=f"phi_{i+1}")
    ax1.plot(t, phi0, linestyle="--", label="phi0")
    ax1.set_title("Volume fractions (phi)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("phi")
    ax1.legend()

    # ψ
    for i in range(N):
        ax2.plot(t, psi[:, i], label=f"psi_{i+1}")
    ax2.set_title("Psi variables")
    ax2.set_xlabel("t")
    ax2.set_ylabel("psi")
    ax2.legend()

    # γ
    ax3.plot(t, gamma)
    ax3.set_title("Gamma")
    ax3.set_xlabel("t")
    ax3.set_ylabel("gamma")

    # φ 総和
    ax4.plot(t, np.sum(phi, axis=1), label="sum(phi_i)")
    ax4.set_title("Sum of phi_i")
    ax4.set_xlabel("t")
    ax4.set_ylabel("sum(phi)")
    ax4.legend()

    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"[utils] Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_numpy_data(
    t: np.ndarray,
    g: np.ndarray,
    output_dir: str,
    prefix: str = "biofilm",
):
    os.makedirs(output_dir, exist_ok=True)
    t_path = os.path.join(output_dir, f"{prefix}_t.npy")
    g_path = os.path.join(output_dir, f"{prefix}_g.npy")
    np.save(t_path, t)
    np.save(g_path, g)
    print(f"[utils] Saved t to {t_path}")
    print(f"[utils] Saved g to {g_path}")
