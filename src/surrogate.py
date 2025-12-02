# src/surrogate.py
from typing import Optional, Tuple

import numpy as np

from .config import (
    GP_KERNEL_CONSTANT,
    GP_KERNEL_RBF,
    GP_NOISE,
    LENGTH_SCALE,
)


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
    """
    シンプルな RBF カーネル。
    X1: (N, D), X2: (M, D)
    """
    N, D = X1.shape
    M, _ = X2.shape

    K = np.empty((N, M), dtype=float)
    ls2 = length_scale * length_scale

    for i in range(N):
        for j in range(M):
            sqdist = 0.0
            for d in range(D):
                diff = X1[i, d] - X2[j, d]
                sqdist += diff * diff
            K[i, j] = np.exp(-0.5 * sqdist / ls2)
    return K


class BiofilmSurrogateGP:
    """
    非常にシンプルな GP 回帰クラス。
    TMCMC などから高価な Forward Solver を置き換えるための土台。
    """

    def __init__(
        self,
        length_scale: float = LENGTH_SCALE,
        noise: float = GP_NOISE,
        kernel_constant: float = GP_KERNEL_CONSTANT,
        kernel_rbf: float = GP_KERNEL_RBF,
    ) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self.kernel_constant = kernel_constant
        self.kernel_rbf = kernel_rbf

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, D)
        y: (N,)
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)

        K_rbf = rbf_kernel(self.X_train, self.X_train, self.length_scale)
        K = (
            self.kernel_constant * np.ones_like(K_rbf)
            + self.kernel_rbf * K_rbf
        )
        K += (self.noise ** 2) * np.eye(K.shape[0])

        self.K_inv = np.linalg.inv(K)

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測値と分散を返す。
        X_star: (M, D)
        戻り値:
            mean: (M,)
            var:  (M,)
        """
        if self.X_train is None or self.y_train is None or self.K_inv is None:
            raise RuntimeError("GP is not fitted yet.")

        X_star = np.asarray(X_star, dtype=float)
        K_star = rbf_kernel(self.X_train, X_star, self.length_scale)
        K_star_T = K_star.T

        # mean
        alpha = self.K_inv @ self.y_train
        mean = K_star_T @ alpha

        # variance (対角のみ)
        K_ss = np.diag(
            self.kernel_constant + self.kernel_rbf * np.ones(X_star.shape[0])
        )
        v = self.K_inv @ K_star
        var = np.maximum(0.0, np.diag(K_ss - K_star_T @ v))

        return mean, var
