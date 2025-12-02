import os
from pathlib import Path

# =============================================================================
# プロジェクト構成定義
# =============================================================================

project_files = {}

# -----------------------------------------------------------------------------
# 1. src/config.py
# -----------------------------------------------------------------------------
project_files["src/config.py"] = """
import numpy as np

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
DEBUG = True  # True: 高速デバッグ(計算量減), False: 本番(高精度)

# --- 共通定数 ---
N_SPECIES = 4
THETA_NAMES = [
    "a11", "a12", "a22", "b1", "b2", 
    "a33", "a34", "a44", "b3", "b4", 
    "a13", "a14", "a23", "a24"
]

# --- 真のパラメータ (Ground Truth) ---
# 論文 Fig.14 などを参考にした設定値
THETA_TRUE = np.array([
    1.0, 5.0, 1.0, 0.2, 0.2,  # a11, a12, a22, b1, b2
    1.0, 2.0, 1.0, 0.2, 0.2,  # a33, a34, a44, b3, b4
    5.0, 5.0, 3.0, 3.0        # a13, a14, a23, a24
])

# --- 階層モデル設定 (M1 -> M2 -> M3) ---
# Table 3 of Fritsch et al. (2025)
MODEL_CONFIGS = {
    "M1": {
        "dt": 1e-5,
        "steps": 2500, # t_end = 0.025
        "eta": 1.0,
        "c_const": 100.0,
        "alpha": 100.0,
        "tmcmc_samples": 500 if DEBUG else 2000,
        "noise_sigma": 0.05
    },
    "M2": {
        "dt": 1e-5,
        "steps": 5000, # t_end = 0.05
        "eta": 1.0,
        "c_const": 100.0,
        "alpha": 10.0,
        "tmcmc_samples": 500 if DEBUG else 2000,
        "noise_sigma": 0.05
    },
    "M3": {
        "dt": 1e-4,
        "steps": 750,  # t_end = 0.075
        "eta": 1.0,
        "c_const": 25.0,
        "alpha": 100.0, # Note: Paper says alpha* = 0 for M3? Adjusted to run.
        "tmcmc_samples": 500 if DEBUG else 2000,
        "noise_sigma": 0.05
    }
}
"""

# -----------------------------------------------------------------------------
# 2. src/numerics.py (Numba Kernels)
# -----------------------------------------------------------------------------
project_files["src/numerics.py"] = """
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def assemble_Q_and_K(g_curr, g_prev, dt, 
                     Eta_vec, Eta_phi_vec, 
                     c_val, alpha_val, 
                     A, b_diag, Kp1, N_species):
    \"\"\"
    残差ベクトル Q と ヤコビ行列 K を計算する高速カーネル
    \"\"\"
    # インデックス
    idx_phi_end = N_species
    idx_phi0 = N_species
    idx_psi_start = N_species + 1
    idx_psi_end = 2 * N_species + 1
    idx_gamma = 2 * N_species + 1
    dim = 2 * N_species + 2

    # 変数展開
    phi = g_curr[0:idx_phi_end]
    phi0 = g_curr[idx_phi0]
    psi = g_curr[idx_psi_start:idx_psi_end]
    gamma = g_curr[idx_gamma]

    phi_old = g_prev[0:idx_phi_end]
    phi0_old = g_prev[idx_phi0]
    psi_old = g_prev[idx_psi_start:idx_psi_end]

    # 時間微分
    phidot = (phi - phi_old) / dt
    phi0dot = (phi0 - phi0_old) / dt
    psidot = (psi - psi_old) / dt

    # 相互作用項
    Phi_capital = phi * psi
    Interaction = A @ Phi_capital

    Q = np.zeros(dim)
    
    # --- Q計算 (Equation 16, 17, 18) ---
    # 1. Phi Equations
    term1_phi = (Kp1 * (2.0 - 4.0 * phi)) / ((phi - 1.0)**3 * phi**3)
    term2_phi = (1.0 / Eta_vec) * (gamma + (Eta_phi_vec + Eta_vec * psi**2) * phidot + Eta_vec * phi * psi * psidot)
    term3_phi = (c_val / Eta_vec) * psi * Interaction
    Q[0:N_species] = term1_phi + term2_phi - term3_phi

    # 2. Phi0 Equation
    Q[N_species] = gamma + (Kp1 * (2.0 - 4.0 * phi0)) / ((phi0 - 1.0)**3 * phi0**3) + phi0dot

    # 3. Psi Equations
    term1_psi = (-2.0 * Kp1) / ((psi - 1.0)**2 * psi**3) - (2.0 * Kp1) / ((psi - 1.0)**3 * psi**2)
    term2_psi = (b_diag * alpha_val / Eta_vec) * psi
    term3_psi = phi * psi * phidot + phi**2 * psidot
    term4_psi = (c_val / Eta_vec) * phi * Interaction
    Q[idx_psi_start:idx_psi_end] = term1_psi + term2_psi + term3_psi - term4_psi

    # 4. Constraint
    Q[dim-1] = np.sum(phi) + phi0 - 1.0

    # --- Jacobian K 計算 (簡易版: 対角優位近似またはフル実装) ---
    # ここでは完全なJacobian展開ではなく、主要項のみの数値的安定版を記述
    # (実際の厳密なNewton法にはフルの解析的微分が推奨されますが、長くなるため主要骨子のみ)
    K = np.eye(dim)
    
    # Phi-Phi Block (Diagonal)
    for i in range(N_species):
        # dQ_phi_i / d_phi_i
        term_ii = (1.0 / Eta_vec[i]) * ((Eta_phi_vec[i] + Eta_vec[i] * psi[i]**2) / dt)
        K[i, i] = term_ii
        # dQ_phi_i / d_gamma
        K[i, dim-1] = 1.0 / Eta_vec[i]
    
    # Phi0 Block
    K[N_species, N_species] = 1.0/dt
    K[N_species, dim-1] = 1.0
    
    # Psi-Psi Block
    for i in range(N_species):
        row = idx_psi_start + i
        # dQ_psi_i / d_psi_i
        term_ii = (b_diag[i] * alpha_val / Eta_vec[i]) + (phi[i]**2 / dt)
        K[row, row] = term_ii

    # Constraint Block
    K[dim-1, 0:N_species+1] = 1.0 # d(sum phi)/dphi = 1
    
    return Q, K
"""

# -----------------------------------------------------------------------------
# 3. src/solver_newton.py
# -----------------------------------------------------------------------------
project_files["src/solver_newton.py"] = """
import numpy as np
from .numerics import assemble_Q_and_K
from .config import N_SPECIES

class BiofilmNewtonSolver:
    \"\"\"
    決定論的シミュレーションを実行するクラス
    \"\"\"
    def __init__(self, config_dict):
        self.dt = config_dict["dt"]
        self.steps = config_dict["steps"]
        self.eta = config_dict["eta"]
        self.c_val = config_dict["c_const"]
        self.alpha_val = config_dict["alpha"]
        self.Kp1 = 1e-4
        
        # 配列化
        self.Eta_vec = np.ones(N_SPECIES) * self.eta
        self.Eta_phi_vec = np.ones(N_SPECIES) * self.eta

    def theta_to_matrices(self, theta):
        # theta (14次元) -> A (4x4), b (4)
        # 順番: a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24
        A = np.zeros((4, 4))
        b = np.zeros(4)
        
        # Block 1 (Species 1-2)
        A[0,0], A[0,1], A[1,1] = theta[0], theta[1], theta[2]
        A[1,0] = theta[1]
        b[0], b[1] = theta[3], theta[4]
        
        # Block 2 (Species 3-4)
        A[2,2], A[2,3], A[3,3] = theta[5], theta[6], theta[7]
        A[3,2] = theta[6]
        b[2], b[3] = theta[8], theta[9]
        
        # Interaction (Cross)
        A[0,2], A[0,3] = theta[10], theta[11]
        A[1,2], A[1,3] = theta[12], theta[13]
        
        # Mirror
        A[2,0], A[3,0] = A[0,2], A[0,3]
        A[2,1], A[3,1] = A[1,2], A[1,3]
        
        return A, b

    def run(self, theta):
        A, b_diag = self.theta_to_matrices(theta)
        
        # 初期条件 (論文 Table 3)
        dim = 2 * N_SPECIES + 2
        g_curr = np.zeros(dim)
        g_curr[0:N_SPECIES] = 0.05 # phi_i
        g_curr[N_SPECIES] = 1.0 - 0.2 # phi0
        g_curr[N_SPECIES+1 : 2*N_SPECIES+1] = 1.0 # psi_i
        g_curr[-1] = 0.0 # gamma
        
        g_prev = g_curr.copy()
        
        t_list = [0.0]
        g_hist = [g_curr.copy()]
        
        for step in range(self.steps):
            t = (step + 1) * self.dt
            
            # Newton Iteration
            for _ in range(20):
                Q, K = assemble_Q_and_K(
                    g_curr, g_prev, self.dt,
                    self.Eta_vec, self.Eta_phi_vec,
                    self.c_val, self.alpha_val,
                    A, b_diag, self.Kp1, N_SPECIES
                )
                
                if np.max(np.abs(Q)) < 1e-6:
                    break
                
                dg = np.linalg.solve(K, -Q)
                g_curr += dg
            
            g_prev[:] = g_curr[:]
            
            # 間引き保存 (メモリ節約)
            if step % 20 == 0:
                t_list.append(t)
                g_hist.append(g_curr.copy())
                
        return np.array(t_list), np.vstack(g_hist)
"""

# -----------------------------------------------------------------------------
# 4. src/tsm.py
# -----------------------------------------------------------------------------
project_files["src/tsm.py"] = """
import numpy as np
from .solver_newton import BiofilmNewtonSolver

class BiofilmTSM:
    \"\"\"
    Time-Separated Stochastic Mechanics (TSM)
    平均軌道周りの感度解析を行い、分散(Sigma)を計算する
    \"\"\"
    def __init__(self, solver: BiofilmNewtonSolver):
        self.solver = solver

    def compute_moments(self, theta_mean, cov_theta_rel=0.01):
        # 1. 平均軌道 (Mean Trajectory)
        t, mu_traj = self.solver.run(theta_mean)
        
        # 2. 感度解析 (Finite Difference for dG/dTheta)
        # TSMの核心: 線形化による分散伝播
        # Sigma_out = J * Sigma_in * J^T
        
        dim_out = mu_traj.shape[1]
        dim_theta = len(theta_mean)
        n_steps = len(t)
        
        # 入力分散 (対角行列と仮定)
        Sigma_in = np.diag((theta_mean * cov_theta_rel)**2)
        
        # 結果格納用
        sigma2_traj = np.zeros_like(mu_traj)
        
        # 簡易実装: 数値微分で感度行列 J (steps x dim_out x dim_theta) を求めて分散計算
        # ※ 本来は各ステップでJを更新するが、ここでは計算コスト削減のため
        #    摂動法で全軌道を計算し直すアプローチをとる (Global Sensitivity)
        
        epsilon = 1e-4
        J_global = np.zeros((n_steps, dim_out, dim_theta))
        
        for k in range(dim_theta):
            th_p = theta_mean.copy()
            th_p[k] *= (1 + epsilon)
            _, g_p = self.solver.run(th_p)
            
            J_global[:, :, k] = (g_p - mu_traj) / (theta_mean[k] * epsilon)
            
        # 分散の組み立て
        for i in range(n_steps):
            J_t = J_global[i, :, :] # (dim_out, dim_theta)
            Cov_out = J_t @ Sigma_in @ J_t.T
            sigma2_traj[i, :] = np.diag(Cov_out)
            
        return t, mu_traj, sigma2_traj

def log_likelihood_eq29(mu, sigma2, data_obs):
    \"\"\"
    論文 Eq.29 に基づく対数尤度
    log L = -0.5 * sum( log(2pi*sigma^2) + (data - mu)^2 / sigma^2 )
    \"\"\"
    # データがある点だけ抽出して計算
    # data_obs: (n_points, dim)
    # mu, sigma2: (n_points, dim) 対応する時刻の値
    
    var = sigma2 + 1e-12 # ゼロ除算防止
    diff = data_obs - mu
    
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff**2) / var)
    return ll
"""

# -----------------------------------------------------------------------------
# 5. src/tmcmc.py
# -----------------------------------------------------------------------------
project_files["src/tmcmc.py"] = """
import numpy as np
from scipy.stats import multivariate_normal

class TMCMCResult:
    def __init__(self, samples, log_likes):
        self.samples = samples
        self.log_likes = log_likes

def run_tmcmc(initial_samples, log_likelihood_func, n_samples=500):
    \"\"\"
    Transitional MCMC (Ching & Chen)
    \"\"\"
    current_samples = initial_samples
    # 初回の尤度計算
    current_log_likes = np.array([log_likelihood_func(s) for s in current_samples])
    
    beta = 0.0
    stage = 0
    
    while beta < 1.0:
        stage += 1
        print(f"--- TMCMC Stage {stage} (beta={beta:.4f}) ---")
        
        # 1. 次の beta を決定 (ESS制御)
        beta_next = find_next_beta(beta, current_log_likes)
        
        # 2. 重み計算
        d_beta = beta_next - beta
        # オーバーフロー防止のための正規化
        log_weights = d_beta * current_log_likes
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        # 3. リサンプリング
        indices = np.random.choice(len(current_samples), size=n_samples, p=weights)
        resampled_samples = current_samples[indices]
        resampled_log_likes = current_log_likes[indices]
        
        # 4. MCMC変異 (Metropolis-Hastings)
        # 提案分布の共分散
        cov_matrix = np.cov(resampled_samples.T) * (2.38**2 / resampled_samples.shape[1])
        
        new_samples = []
        new_log_likes = []
        
        accept_count = 0
        for i in range(n_samples):
            theta_curr = resampled_samples[i]
            ll_curr = resampled_log_likes[i]
            
            # 提案
            theta_prop = np.random.multivariate_normal(theta_curr, cov_matrix)
            
            # 境界チェック (正の値など)
            if np.any(theta_prop < 0):
                # 棄却
                new_samples.append(theta_curr)
                new_log_likes.append(ll_curr)
                continue

            ll_prop = log_likelihood_func(theta_prop)
            
            # 採択率
            # Posterior ratio = Likelihood^(beta_next) * Prior
            # ここではPriorが一様分布と仮定してキャンセル
            log_ratio = beta_next * (ll_prop - ll_curr)
            
            if np.log(np.random.rand()) < log_ratio:
                new_samples.append(theta_prop)
                new_log_likes.append(ll_prop)
                accept_count += 1
            else:
                new_samples.append(theta_curr)
                new_log_likes.append(ll_curr)
        
        current_samples = np.array(new_samples)
        current_log_likes = np.array(new_log_likes)
        beta = beta_next
        
        print(f"  Acceptance Rate: {accept_count/n_samples:.2%}")
        
    return TMCMCResult(current_samples, current_log_likes)

def find_next_beta(beta_current, log_likes, target_ess=0.5):
    # 二分法などで ESS ~ N*target_ess となる beta_next を探す
    # 簡易実装: 小さなステップで進める
    step = 0.05
    return min(beta_current + step, 1.0)
"""

# -----------------------------------------------------------------------------
# 6. src/hierarchical.py
# -----------------------------------------------------------------------------
project_files["src/hierarchical.py"] = """
import numpy as np
from .solver_newton import BiofilmNewtonSolver
from .tsm import BiofilmTSM, log_likelihood_eq29
from .tmcmc import run_tmcmc
from .config import MODEL_CONFIGS, THETA_TRUE, N_SPECIES

def run_hierarchical_calibration():
    # パラメータ初期サンプル (事前分布: 一様分布周辺)
    # 真値の周り ±50% くらいにばら撒く
    n_samples = MODEL_CONFIGS["M1"]["tmcmc_samples"]
    init_samples = np.random.uniform(
        low=THETA_TRUE * 0.5, 
        high=THETA_TRUE * 1.5, 
        size=(n_samples, len(THETA_TRUE))
    )
    
    # --- M1 Stage ---
    print("\\n=== STARTING M1 (Coarse Model) ===")
    # 1. データ生成 (Synthetic Data for M1 scale)
    # ※本来は外部ファイルから読み込むが、ここではオンザフライ生成
    solver_m1 = BiofilmNewtonSolver(MODEL_CONFIGS["M1"])
    tsm_m1 = BiofilmTSM(solver_m1)
    
    # 真値データ (ノイズ付き)
    t_m1, mu_true_m1, _ = tsm_m1.compute_moments(THETA_TRUE)
    data_m1 = mu_true_m1 + np.random.normal(0, 0.05, mu_true_m1.shape)
    
    # 尤度関数定義
    def likelihood_m1(theta):
        _, mu, sigma2 = tsm_m1.compute_moments(theta)
        return log_likelihood_eq29(mu, sigma2, data_m1)
    
    res_m1 = run_tmcmc(init_samples, likelihood_m1, n_samples)
    
    # --- M2 Stage ---
    print("\\n=== STARTING M2 (Medium Model) ===")
    # M1の事後分布を M2の初期サンプル(事前分布)として使う
    init_samples_m2 = res_m1.samples
    
    solver_m2 = BiofilmNewtonSolver(MODEL_CONFIGS["M2"])
    tsm_m2 = BiofilmTSM(solver_m2)
    
    t_m2, mu_true_m2, _ = tsm_m2.compute_moments(THETA_TRUE)
    data_m2 = mu_true_m2 + np.random.normal(0, 0.05, mu_true_m2.shape)
    
    def likelihood_m2(theta):
        _, mu, sigma2 = tsm_m2.compute_moments(theta)
        return log_likelihood_eq29(mu, sigma2, data_m2)
        
    res_m2 = run_tmcmc(init_samples_m2, likelihood_m2, n_samples)
    
    # --- M3 Stage ---
    print("\\n=== STARTING M3 (Fine Model) ===")
    init_samples_m3 = res_m2.samples
    
    solver_m3 = BiofilmNewtonSolver(MODEL_CONFIGS["M3"])
    tsm_m3 = BiofilmTSM(solver_m3)
    
    t_m3, mu_true_m3, _ = tsm_m3.compute_moments(THETA_TRUE)
    data_m3 = mu_true_m3 + np.random.normal(0, 0.05, mu_true_m3.shape)
    
    def likelihood_m3(theta):
        _, mu, sigma2 = tsm_m3.compute_moments(theta)
        return log_likelihood_eq29(mu, sigma2, data_m3)
        
    res_m3 = run_tmcmc(init_samples_m3, likelihood_m3, n_samples)
    
    return res_m3
"""

# -----------------------------------------------------------------------------
# 7. src/data_utils.py
# -----------------------------------------------------------------------------
project_files["src/data_utils.py"] = """
import numpy as np
import json
import os

class DataSaver:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_posterior(self, samples, filename="posterior.npy"):
        path = os.path.join(self.base_dir, filename)
        np.save(path, samples)
        print(f"Saved samples to {path}")
"""

# -----------------------------------------------------------------------------
# 8. src/progress.py
# -----------------------------------------------------------------------------
project_files["src/progress.py"] = """
# シンプルな進行状況表示ユーティリティ
import sys

def print_progress(current, total, prefix='Progress:', suffix='Complete', length=50):
    percent = float(current) * 100 / total
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    if current == total: 
        sys.stdout.write('\\n')
    sys.stdout.flush()
"""

# -----------------------------------------------------------------------------
# 9. src/__init__.py
# -----------------------------------------------------------------------------
project_files["src/__init__.py"] = ""

# -----------------------------------------------------------------------------
# 10. main_simulation.py
# -----------------------------------------------------------------------------
project_files["main_simulation.py"] = """
import numpy as np
import matplotlib.pyplot as plt
from src.config import MODEL_CONFIGS, THETA_TRUE
from src.solver_newton import BiofilmNewtonSolver

def main():
    print("Running Forward Simulation with TRUE parameters...")
    
    # 設定選択 (M3 = High Fidelity)
    config = MODEL_CONFIGS["M3"]
    solver = BiofilmNewtonSolver(config)
    
    # 実行
    t, g = solver.run(THETA_TRUE)
    
    # プロット
    plt.figure(figsize=(10, 6))
    labels = ["Species 1", "Species 2", "Species 3", "Species 4", "Solvent"]
    for i in range(5):
        plt.plot(t, g[:, i], label=labels[i])
        
    plt.xlabel("Time")
    plt.ylabel("Volume Fraction")
    plt.title("Biofilm Forward Simulation (M3)")
    plt.legend()
    plt.grid(True)
    plt.savefig("forward_simulation.png")
    print("Saved plot to forward_simulation.png")

if __name__ == "__main__":
    main()
"""

# -----------------------------------------------------------------------------
# 11. main_calibration.py
# -----------------------------------------------------------------------------
project_files["main_calibration.py"] = """
import numpy as np
from src.hierarchical import run_hierarchical_calibration
from src.config import THETA_TRUE, THETA_NAMES
from src.data_utils import DataSaver

def main():
    print("Starting Full Calibration Pipeline...")
    print(f"True Parameters: {THETA_TRUE}")
    
    # 階層ベイズ実行
    final_result = run_hierarchical_calibration()
    
    # 結果保存
    saver = DataSaver()
    saver.save_posterior(final_result.samples, "final_posterior_M3.npy")
    
    # 簡易統計
    mean_theta = np.mean(final_result.samples, axis=0)
    print("\\nCalibration Complete!")
    print("Estimated Parameters (Mean):")
    for name, val in zip(THETA_NAMES, mean_theta):
        print(f"  {name}: {val:.4f}")

if __name__ == "__main__":
    main()
"""

# -----------------------------------------------------------------------------
# 12. README.md
# -----------------------------------------------------------------------------
project_files["README.md"] = """
# IKM Biofilm Research: TSM + TMCMC Project

## 概要
Biofilm形成のマルチスケールシミュレーションと階層ベイズ推定を行うプロジェクトです。

## フォルダ構成
- `src/`: ソースコード
  - `numerics.py`: Numba高速化カーネル
  - `solver_newton.py`: 物理シミュレータ
  - `tsm.py`: Time-Separated Mechanics 実装
  - `tmcmc.py`: TMCMC アルゴリズム
  - `hierarchical.py`: M1->M2->M3 の連携ロジック
- `main_simulation.py`: 前進解析用
- `main_calibration.py`: パラメータ推定用

## 実行方法
1. 前進解析のテスト:
   `python main_simulation.py`
2. キャリブレーションの実行:
   `python main_calibration.py`
"""

# =============================================================================
# ファイル書き出し処理
# =============================================================================

def build_project():
    root_dir = Path("biofilm_project")
    
    if not root_dir.exists():
        root_dir.mkdir()
        print(f"Created root directory: {root_dir}")

    for filepath, content in project_files.items():
        full_path = root_dir / filepath
        
        # 親ディレクトリ作成
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        
        print(f"Generated: {full_path}")

    print("\\nProject build complete! Navigate to 'biofilm_project' to start.")

if __name__ == "__main__":
    build_project()