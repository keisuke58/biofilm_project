# src/solvers.py
import time
import numpy as np

from .numerics import assemble_system_numba_generic
from .config import (
    N_SPECIES,
    DT,
    DEFAULT_STEPS,
    ETA_VEC,
    ETA_PHI_VEC,
    C_CONST,
    ALPHA_CONST,
    KP1,
)


class FastBiofilmSolver:
    """
    Biofilm の時間発展を行うソルバークラス。

    - A, b_diag はランダムに生成（将来的には config / 引数から与える）
    - g = [phi(1..N), phi0, psi(1..N), gamma]
    """

    def __init__(
        self,
        N_species: int = N_SPECIES,
        dt: float = DT,
        steps: int = DEFAULT_STEPS,
        seed: int = 42,
    ) -> None:
        self.N = N_species
        self.dt = dt
        self.steps = steps

        # 相互作用行列 A の初期化
        rng = np.random.default_rng(seed)
        A_raw = rng.uniform(0.5, 5.0, size=(N_species, N_species))
        A_sym = 0.5 * (A_raw + A_raw.T)
        for i in range(N_species):
            A_sym[i, i] = 1.0
        self.A = A_sym

        # 対角項 b_diag
        self.b_diag = rng.uniform(0.5, 1.5, size=N_species)

    def _initial_state(self) -> np.ndarray:
        """
        初期状態ベクトル g の作成
        g = [phi(1..N), phi0, psi(1..N), gamma]
        """
        dim = 2 * self.N + 2
        g = np.zeros(dim)

        # φ: 小さい値からスタート
        phi0_val = 0.05
        g[0 : self.N] = phi0_val
        # φ0: 残り
        g[self.N] = 1.0 - np.sum(g[0 : self.N])
        # ψ: ほぼ 1
        g[self.N + 1 : 2 * self.N + 1] = 0.999
        # γ: 小さい値
        g[-1] = 1e-6
        return g

    def run(self):
        """
        シミュレーション本体。
        戻り値:
            t_hist: shape (T,)
            g_hist: shape (T, dim)
        """
        print(f"[FastBiofilmSolver] N={self.N}, steps={self.steps}, dt={self.dt}")

        g_curr = self._initial_state()
        g_prev = g_curr.copy()

        dim = g_curr.size

        t_hist = [0.0]
        g_hist = [g_curr.copy()]

        start_time = time.time()

        for step in range(self.steps):
            # Newton 法
            for _ in range(20):
                Q, K = assemble_system_numba_generic(
                    g_curr,
                    g_prev,
                    self.dt,
                    ETA_VEC,
                    ETA_PHI_VEC,
                    C_CONST,
                    ALPHA_CONST,
                    self.A,
                    self.b_diag,
                    KP1,
                    self.N,
                )

                max_res = np.max(np.abs(Q))
                if max_res < 1e-8:
                    break

                # 線形方程式 K * dg = -Q を解く
                try:
                    dg = np.linalg.solve(K, -Q)
                except np.linalg.LinAlgError:
                    # 特異行列になった場合は break
                    print("[FastBiofilmSolver] Warning: singular Jacobian.")
                    break

                g_curr += dg

            # 次ステップへ
            g_prev[:] = g_curr[:]

            # データ保存（間引き）
            if (step + 1) % 10 == 0:
                t_hist.append((step + 1) * self.dt)
                g_hist.append(g_curr.copy())

        elapsed = time.time() - start_time
        print(f"[FastBiofilmSolver] Simulation finished in {elapsed:.2f} s")

        t_arr = np.asarray(t_hist, dtype=float)
        g_arr = np.vstack(g_hist)
        return t_arr, g_arr
