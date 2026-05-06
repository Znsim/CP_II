import numpy as np
from typing import List, Tuple


def compute_mean_std(paths: List[str], max_len: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std by accumulating frames from given .npy files."""
    sum_ = None
    sumsq = None
    count = 0
    for p in paths:
        arr = np.load(p).astype(np.float32)
        T, F = arr.shape
        # trim/pad similar to dataset behavior
        if T > max_len:
            start = max(0, (T - max_len) // 2)
            arr = arr[start:start + max_len]
        elif T < max_len:
            pad = np.zeros((max_len - T, F), dtype=arr.dtype)
            arr = np.vstack([arr, pad])

        if sum_ is None:
            sum_ = np.sum(arr, axis=0)
            sumsq = np.sum(arr * arr, axis=0)
        else:
            sum_ += np.sum(arr, axis=0)
            sumsq += np.sum(arr * arr, axis=0)
        count += arr.shape[0]

    mean = sum_ / count
    var = (sumsq / count) - (mean * mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def augment_time_warp(arr: np.ndarray, scale_range: float = 0.1) -> np.ndarray:
    # simple speed change by resampling frames
    import random
    scale = 1.0 + random.uniform(-scale_range, scale_range)
    T, F = arr.shape
    new_T = max(1, int(T * scale))
    idx = np.linspace(0, T - 1, new_T).astype(np.float32)
    left = np.floor(idx).astype(int)
    right = np.ceil(idx).astype(int)
    right = np.clip(right, 0, T - 1)
    alpha = idx - left
    res = arr[left] * (1 - alpha)[:, None] + arr[right] * alpha[:, None]
    return res
