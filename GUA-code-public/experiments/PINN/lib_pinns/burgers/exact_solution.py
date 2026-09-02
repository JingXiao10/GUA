from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .simulation_paras import NU, T_TEST, X_TEST


DEFAULT_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "burgers"
    / "simulation_data_cole_hopf.npy"
)


def burgers_exact_reference(
    xs: np.ndarray = X_TEST,
    ts: np.ndarray = T_TEST,
    nu: float = NU,
    n_quad: int = 4096,
    chunk_x: int = 64,
) -> np.ndarray:
    """Evaluate the Cole-Hopf reference solution on a periodic [-1, 1] grid."""
    xs = np.asarray(xs, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    xi = -1.0 + (np.arange(n_quad, dtype=np.float64) + 0.5) * (2.0 / n_quad)
    log_phi0 = -50.0 * (1.0 + np.cos(np.pi * xi))
    images = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    reference = np.empty((ts.size, xs.size), dtype=np.float64)

    for idx_t, t in enumerate(ts):
        if abs(t) < 1e-15:
            reference[idx_t] = -np.sin(np.pi * xs)
            continue

        for start in range(0, xs.size, chunk_x):
            x_chunk = xs[start : start + chunk_x]
            distance = x_chunk[:, None, None] - xi[None, None, :] + 2.0 * images[None, :, None]
            log_weight = log_phi0[None, None, :] - distance**2 / (4.0 * nu * t)
            flat_log_weight = log_weight.reshape(x_chunk.size, -1)
            flat_distance = distance.reshape(x_chunk.size, -1)
            normalizer = logsumexp(flat_log_weight, axis=1)
            weights = np.exp(flat_log_weight - normalizer[:, None])
            reference[idx_t, start : start + x_chunk.size] = np.sum(weights * flat_distance, axis=1) / t

    return reference.astype(np.float32)


def load_burgers_reference(
    path: str | Path = DEFAULT_REFERENCE_PATH,
    xs: np.ndarray = X_TEST,
    ts: np.ndarray = T_TEST,
    regenerate: bool = False,
) -> np.ndarray:
    path = Path(path)
    if path.exists() and not regenerate:
        reference = np.load(path)
        if reference.shape == (len(ts), len(xs)):
            return reference

    reference = burgers_exact_reference(xs=xs, ts=ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reference)
    return reference
