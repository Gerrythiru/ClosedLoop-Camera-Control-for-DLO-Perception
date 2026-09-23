"""B3: synthetic D435 depth noise model (Mujoco Plan 0, Part B3)."""

from __future__ import annotations

import numpy as np


def add_d435_noise(
    depth_meters: np.ndarray,
    fx: float = 615.0,
    baseline: float = 0.05,
    dropout_rate: float = 0.02,
) -> np.ndarray:
    """Add D435-like stereo-matching noise, edge dropout, and quantization.

    - Gaussian noise with sigma_z(z) ~= z^2 / (fx * baseline), the standard
      D435 depth-error model.
    - Random dropout (1-3%) biased toward depth-gradient edges, simulating
      stereo-matching failure at object boundaries.
    - Quantization to 16-bit unsigned int at 1 mm resolution (D435's native
      depth format).
    """
    sigma_z = depth_meters**2 / (fx * baseline)
    noise = np.random.randn(*depth_meters.shape).astype(np.float32) * sigma_z
    noisy_depth = depth_meters + noise

    grad_x = np.abs(np.diff(depth_meters, axis=1, prepend=depth_meters[:, :1]))
    grad_y = np.abs(np.diff(depth_meters, axis=0, prepend=depth_meters[:1, :]))
    edge_mask = (grad_x + grad_y) > 0.02
    dropout_mask = np.random.rand(*depth_meters.shape) < dropout_rate
    dropout_mask = dropout_mask | (edge_mask & (np.random.rand(*depth_meters.shape) < 0.3))
    noisy_depth[dropout_mask] = 0.0

    noisy_depth_mm = np.round(noisy_depth * 1000).astype(np.uint16)
    noisy_depth = noisy_depth_mm.astype(np.float32) / 1000.0
    noisy_depth[noisy_depth_mm == 0] = 0.0

    return noisy_depth
