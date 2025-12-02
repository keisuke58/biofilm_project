# src/inference.py
from typing import Callable, Tuple

import numpy as np

from .config import (
    TMCMC_N_SAMPLES,
    TMCMC_N_STAGES,
    TMCMC_GAUSS_PROPOSAL_STD,
    TMCMC_RANDOM_SEED,
)
from .config import SIGMA_OBS


LogLikelihoodFn = Callable[[np.ndarray], float]
PriorSampleFn = Callable[[int], np.ndarray]


def gaussian_log_likelihood(residual: np.ndarray, sigma: float) -> float:
    """
    残差ベクトルに対するガウス尤度
    """
    r2 = np.dot(residual, residual)
    n = residual.size
    return -0.5 * (r2 / (sigma * sigma) + n * np.log(2.0 * np.pi * sigma * sigma))


def simple_tmcmc(
    log_like: LogLikelihoodFn,
    prior_sample: PriorSampleFn,
    n_samples: int = TMCMC_N_SAMPLES,
    n_stages: int = TMCMC_N_STAGES,
    proposal_std: float = TMCMC_GAUSS_PROPOSAL_STD,
    random_seed: int = TMCMC_RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    非常にシンプルな TMCMC もどき:
    - β_0 = 0 から β_{n_stages} = 1 まで線形に増やす
    - 各ステージで重要度重み付き resampling + ガウス提案

    引数:
        log_like: log p(data | theta)
        prior_sample: N サンプルを一括生成する関数
    戻り値:
        samples: (n_stages, n_samples, dim_theta)
        beta_seq: (n_stages,)
    """
    rng = np.random.default_rng(random_seed)

    # 最初は事前分布からサンプリング
    theta = prior_sample(n_samples)
    dim = theta.shape[1]

    # β スケジュール
    beta_seq = np.linspace(0.0, 1.0, n_stages)

    all_samples = np.zeros((n_stages, n_samples, dim))

    # 事前段階の log-like を計算
    logL = np.zeros(n_samples)
    for i in range(n_samples):
        logL[i] = log_like(theta[i])

    for s, beta in enumerate(beta_seq):
        print(f"[TMCMC] Stage {s+1}/{n_stages}, beta={beta:.3f}")

        # ウェイト計算（β によるスケーリング）
        w = np.exp(beta * logL - beta * np.max(logL))
        w /= np.sum(w)

        # リサンプリング
        idx = rng.choice(n_samples, size=n_samples, p=w)
        theta = theta[idx]

        # 提案 (ガウス)
        for i in range(n_samples):
            prop = theta[i] + proposal_std * rng.standard_normal(size=dim)
            logL_prop = log_like(prop)
            # Metropolis 受理判定（ここでは β 付き）
            log_acc = beta * (logL_prop - logL[i])
            if np.log(rng.random()) < log_acc:
                theta[i] = prop
                logL[i] = logL_prop

        all_samples[s, :, :] = theta

    return all_samples, beta_seq
