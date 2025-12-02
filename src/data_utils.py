# src/data_utils.py
import numpy as np

def select_sparse_data_indices(n_total, n_data, skip_first=True):
    """
    re_numba.py の select_sparse_data_indices をコピペ
    """
    start_idx = int(n_total * 0.05) if skip_first else 0
    indices = np.linspace(start_idx, n_total - 1, n_data, dtype=int)
    return indices
