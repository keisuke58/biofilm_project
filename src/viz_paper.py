# src/viz_paper.py
import os
from typing import Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Paper-style global settings
# ============================================================

def set_paper_style():
    """
    論文と同じ雰囲気のスタイルに rcParams を設定する。
    - Times 系 serif フォント
    - cm 数式
    - 白背景、細グリッド
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "figure.constrained_layout.use": True,
    })


# カラーパレット（論文に近いイメージ）
COLOR_BLUE = "#1f77b4"   # species 1
COLOR_ORANGE = "#ff7f0e" # species 2
COLOR_GREEN = "#2ca02c"  # species 3
COLOR_RED = "#d62728"    # species 4
COLOR_HIST = "#d33682"   # corner plot のヒスト（マゼンタ系）
COLOR_SCAT = "#268bd2"   # corner plot の散布


# ============================================================
#  基本ユーティリティ
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _normalize_time(t: np.ndarray) -> np.ndarray:
    """t を [0, 1] に正規化（論文の "Normalized time t"）"""
    t = np.asarray(t, dtype=float)
    if t.max() == t.min():
        return np.zeros_like(t)
    return (t - t.min()) / (t.max() - t.min())


# ============================================================
#  Corner plot (= Fig. 3, 8, 10, 12 型)
# ============================================================

def corner_plot(
    samples: np.ndarray,
    true_values: Optional[np.ndarray],
    labels: Sequence[str],
    model_name: str,
    outdir: str = "results",
    filename: Optional[str] = None,
):
    """
    論文の Fig.3 / Fig.8 / Fig.10 / Fig.12 のような corner plot を描く。

    Parameters
    ----------
    samples : (Nsamples, Ndim)
        TMCMC のサンプル。
    true_values : (Ndim,) or None
        真値。None の場合は縦線を描かない。
    labels : list of str
        パラメータ名（r"$a_{11}$"$" など）。
    model_name : str
        "M1", "M2", "M3" など。
    outdir : str
        保存ディレクトリ。
    filename : str or None
        ファイル名。None の場合は f"case2_{model_name}_corner.png"
    """
    set_paper_style()
    _ensure_dir(outdir)

    samples = np.asarray(samples)
    n_params = samples.shape[1]
    assert len(labels) == n_params

    fig, axes = plt.subplots(n_params, n_params, figsize=(2.2 * n_params, 2.2 * n_params))

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # 対角：ヒストグラム
                ax.hist(samples[:, j], bins=40, color=COLOR_HIST, alpha=0.8, density=True)
                if true_values is not None:
                    ax.axvline(true_values[j], color="k", linestyle="--", linewidth=1.2)
                ax.set_ylabel("")  # 右肩のラベルは外で
                ax.set_yticks([])
            elif i > j:
                # 下三角：散布図
                ax.scatter(samples[:, j], samples[:, i], s=4,
                           color=COLOR_SCAT, alpha=0.4, edgecolor="none")
            else:
                # 上三角は空白
                ax.axis("off")

            # 軸ラベル
            if i == n_params - 1 and j <= i:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])

    title = f"{model_name}: Posterior samples"
    fig.suptitle(title, y=0.94)

    if filename is None:
        filename = f"case2_{model_name}_corner.png"
    path = os.path.join(outdir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz_paper] Saved corner plot: {path}")


# ============================================================
#  時系列図 (Fig. 4, 5, 9, 11, 13, 15 型)
# ============================================================

def time_series_posterior_band(
    t: np.ndarray,
    phi_samples: np.ndarray,
    phi_data: Optional[np.ndarray],
    model_name: str,
    species_labels: Sequence[str],
    outdir: str = "results",
    filename: Optional[str] = None,
    ylabel: str = r"$\bar{\varphi}_\ell(t)$",
    use_normalized_time: bool = True,
    colors: Optional[Sequence[str]] = None,
):
    """
    論文 Fig.9, Fig.11, Fig.13, Fig.15 のような
    「posterior の帯 + 実現曲線 (shaded) + データの点」を描画する。

    Parameters
    ----------
    t : (Nt,)
        時間ステップ配列
    phi_samples : (Nsamples, Nt, Nspecies)
        posterior サンプルから得られた時系列。TSM-ROM 等で生成したもの。
    phi_data : (Ndata, Nt_data, Nspecies) or (Nt_data, Nspecies) or None
        観測データ。None の場合はプロットしない。
        一般には、t に対応したインデックスだけ点を打つことを想定。
    model_name : str
        "M1", "M2", "M3", "M3_val" など。
    species_labels : list of str
        凡例用の species 名（例: ["Species 1", "Species 2"]）
    """
    set_paper_style()
    _ensure_dir(outdir)

    t = np.asarray(t)
    if use_normalized_time:
        t_plot = _normalize_time(t)
        x_label = "Normalized time $t$"
    else:
        t_plot = t
        x_label = "Time $t$"

    phi_samples = np.asarray(phi_samples)
    assert phi_samples.ndim == 3  # (Ns, Nt, Nspp)
    n_species = phi_samples.shape[2]

    if colors is None:
        base_colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED]
        colors = base_colors[:n_species]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    # Posterior band: 全サンプルを薄く、平均を太線に
    for s in range(n_species):
        # 全サンプル
        for i in range(phi_samples.shape[0]):
            ax.plot(
                t_plot,
                phi_samples[i, :, s],
                color=colors[s],
                alpha=0.02,
                linewidth=0.7,
            )
        mean_phi = phi_samples[:, :, s].mean(axis=0)
        ax.plot(t_plot, mean_phi, color=colors[s], linewidth=1.8,
                label=f"Realizations Species {s+1}")

    # データ点
    if phi_data is not None:
        phi_data = np.asarray(phi_data)
        # 形状を (Nt_data, Nspecies) に揃える
        if phi_data.ndim == 3:
            # (Nreal, Nt_data, Nspp) → とりあえず mean over realizations
            phi_data_plot = phi_data.mean(axis=0)
        else:
            phi_data_plot = phi_data  # (Nt_data, Nspp)

        # t_data はユーザ側で t と同じ index にしている前提：
        # ここでは t_plot の一部に marker を打つだけ
        nt_data = phi_data_plot.shape[0]
        idx = np.linspace(0, len(t_plot) - 1, nt_data).astype(int)
        for s in range(n_species):
            ax.scatter(
                t_plot[idx],
                phi_data_plot[:, s],
                color=colors[s],
                edgecolor="k",
                s=25,
                zorder=3,
                label=f"Data Species {s+1}" if s == 0 else None,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)

    # 凡例は論文風にまとめる
    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=True)

    ax.set_title(model_name)

    if filename is None:
        filename = f"case2_{model_name}_timeseries.png"
    path = os.path.join(outdir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz_paper] Saved time series: {path}")


# ============================================================
#  パラメータ比較 (Fig. 14 型)
# ============================================================

def parameter_comparison_bar(
    theta_true: np.ndarray,
    theta_est: np.ndarray,
    theta_std: Optional[np.ndarray],
    labels: Sequence[str],
    outdir: str = "results",
    filename: str = "case2_param_comparison.png",
):
    """
    論文 Fig.14 のような「真値 vs 推定値 + エラーバー」を描画。

    Parameters
    ----------
    theta_true : (Nparam,)
    theta_est : (Nparam,)
    theta_std : (Nparam,) or None
        posterior の標準偏差。None の場合はエラーバー無し。
    labels : list of str
        r"$a_{11}$" など。
    """
    set_paper_style()
    _ensure_dir(outdir)

    theta_true = np.asarray(theta_true)
    theta_est = np.asarray(theta_est)
    assert theta_true.shape == theta_est.shape
    n = theta_true.size
    x = np.arange(n)

    width = 0.35

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    ax.bar(x - width/2, theta_true, width=width,
           color="white", edgecolor="k", label="True")
    if theta_std is not None:
        theta_std = np.asarray(theta_std)
        ax.bar(
            x + width/2, theta_est, width=width,
            yerr=theta_std, capsize=3,
            color=COLOR_BLUE, edgecolor="k",
            label="Identified"
        )
    else:
        ax.bar(
            x + width/2, theta_est, width=width,
            color=COLOR_BLUE, edgecolor="k",
            label="Identified"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Parameter value [-]")
    ax.legend(frameon=True)
    ax.set_title("Case II: Identified vs true means")

    path = os.path.join(outdir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz_paper] Saved parameter comparison: {path}")


# ============================================================
#  Case II 用ワンショット関数
# ============================================================

def make_all_case2_figures(
    results,
    # 以下は必要に応じて None のままでもよい（その図だけスキップ）
    time_M1: Optional[np.ndarray] = None,
    phi_post_M1: Optional[np.ndarray] = None,
    data_M1: Optional[np.ndarray] = None,
    time_M2: Optional[np.ndarray] = None,
    phi_post_M2: Optional[np.ndarray] = None,
    data_M2: Optional[np.ndarray] = None,
    time_M3: Optional[np.ndarray] = None,
    phi_post_M3: Optional[np.ndarray] = None,
    data_M3: Optional[np.ndarray] = None,
    time_M3_val: Optional[np.ndarray] = None,
    phi_post_M3_val: Optional[np.ndarray] = None,
    data_M3_val: Optional[np.ndarray] = None,
    outdir: str = "results",
):
    """
    Case II 用に、論文 Fig.8–15 相当の図をまとめて描く。

    Parameters
    ----------
    results : HierarchicalResults
        hierarchical_case2(CONFIG) の戻り値（あなたのコードのオブジェクト）。
        少なくとも以下の属性を持っていることを想定:
        - tmcmc_M1.samples, beta_list, logL_list
        - tmcmc_M2.samples, ...
        - tmcmc_M3.samples, ...
        - theta_true_M1, theta_true_M2, theta_true_M3
        - labels_M1, labels_M2, labels_M3
        - theta_final, theta_std など（必要なら追加）
    """
    # Corner plots (Fig. 8, Fig.10, Fig.12)
    # Extract last stage samples from TMCMC result
    corner_plot(
        samples=results.tmcmc_M1.samples[-1],  # Last stage
        true_values=results.theta_true_M1,
        labels=results.labels_M1,
        model_name="M1",
        outdir=outdir,
        filename="case2_M1_corner.png",
    )
    corner_plot(
        samples=results.tmcmc_M2.samples[-1],  # Last stage
        true_values=results.theta_true_M2,
        labels=results.labels_M2,
        model_name="M2",
        outdir=outdir,
        filename="case2_M2_corner.png",
    )
    corner_plot(
        samples=results.tmcmc_M3.samples[-1],  # Last stage
        true_values=results.theta_true_M3,
        labels=results.labels_M3,
        model_name="M3",
        outdir=outdir,
        filename="case2_M3_corner.png",
    )

    # Time series (Fig. 9, 11, 13, 15)
    if time_M1 is not None and phi_post_M1 is not None:
        time_series_posterior_band(
            t=time_M1,
            phi_samples=phi_post_M1,
            phi_data=data_M1,
            model_name=r"$\mathcal{M}^1$",
            species_labels=["Species 1", "Species 2"],
            outdir=outdir,
            filename="case2_M1_timeseries.png",
            ylabel=r"$\bar{\varphi}_\ell(t)$",
        )
    if time_M2 is not None and phi_post_M2 is not None:
        time_series_posterior_band(
            t=time_M2,
            phi_samples=phi_post_M2,
            phi_data=data_M2,
            model_name=r"$\mathcal{M}^2$",
            species_labels=["Species 3", "Species 4"],
            outdir=outdir,
            filename="case2_M2_timeseries.png",
            ylabel=r"$\bar{\varphi}_\ell(t)$",
        )
    if time_M3 is not None and phi_post_M3 is not None:
        time_series_posterior_band(
            t=time_M3,
            phi_samples=phi_post_M3,
            phi_data=data_M3,
            model_name=r"$\mathcal{M}^3$",
            species_labels=["Species 1", "Species 2", "Species 3", "Species 4"],
            outdir=outdir,
            filename="case2_M3_timeseries.png",
            ylabel=r"$\bar{\varphi}_\ell(t)$",
        )
    if time_M3_val is not None and phi_post_M3_val is not None:
        time_series_posterior_band(
            t=time_M3_val,
            phi_samples=phi_post_M3_val,
            phi_data=data_M3_val,
            model_name=r"$\mathcal{M}^{3}_{\mathrm{val}}$",
            species_labels=["Species 1", "Species 2", "Species 3", "Species 4"],
            outdir=outdir,
            filename="case2_M3_validation.png",
            ylabel=r"$\bar{\varphi}_\ell(t)$",
        )

    # Parameter comparison (Fig. 14)
    if hasattr(results, "theta_final") and hasattr(results, "theta_true_full"):
        if hasattr(results, "theta_std"):
            theta_std = results.theta_std
        else:
            theta_std = None

        parameter_comparison_bar(
            theta_true=results.theta_true_full,
            theta_est=results.theta_final,
            theta_std=theta_std,
            labels=results.labels_full,
            outdir=outdir,
            filename="case2_param_comparison.png",
        )
