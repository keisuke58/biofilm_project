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