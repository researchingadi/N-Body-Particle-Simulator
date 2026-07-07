"""GPU (Taichi) vs. NumPy direct vs. Barnes-Hut CPU: validation and benchmarking.

Compares three force backends on the same systems:
- numpy_direct: physics.forces.compute_accelerations -- the numerical
  reference every other backend is measured against.
- taichi_direct: physics.taichi_forces.taichi_direct_accelerations -- same
  O(N^2) sum as numpy_direct, evaluated on parallel threads. Any deviation
  from numpy_direct here should be floating-point noise, not a systematic
  error (see tests/test_taichi_forces.py).
- barnes_hut_cpu: physics.barnes_hut.barnes_hut_accelerations -- a genuine
  approximation, included so Stage 3's GPU numbers can be read against
  Stage 2's tree-approximation numbers on the same axes.

"Error" throughout always means "this backend's deviation from
numpy_direct", never the reverse.
"""
from __future__ import annotations

import csv
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is optional; CSV export still works without it
    pd = None  # type: ignore[assignment]

from initial_conditions.presets import random_cloud
from physics.barnes_hut import barnes_hut_accelerations
from physics.forces import compute_accelerations
from physics.taichi_forces import taichi_direct_accelerations

# factory(n, seed) -> (positions, masses)
PositionsMassesFactory = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def _positions_and_masses_only(preset_triple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    positions, _velocities, masses = preset_triple
    return positions, masses


GPU_VALIDATION_SYSTEMS: dict[str, tuple[PositionsMassesFactory, float]] = {
    "random_cloud": (
        lambda n, seed: _positions_and_masses_only(random_cloud(n=n, seed=seed)),
        0.05,
    ),
}


def _peak_rss_mb() -> float:
    """Process peak resident-set-size high-water-mark, in MB (Linux: ru_maxrss is KB).

    IMPORTANT caveat, stated here once rather than at every call site: this
    is a whole-process cumulative high-water mark, not a measurement
    isolated to a single solver call. Neither NumPy, Taichi, nor Python
    expose a clean per-call allocation delta without much heavier
    instrumentation (e.g. running each measurement in its own subprocess,
    which the accuracy/runtime sweeps deliberately avoid for speed). Treat
    this as "how has peak memory grown as N increases within this one
    script run", not "how much memory did backend X use for this one call".
    """
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0


@dataclass
class GPUValidationResult:
    """One row: one backend's accuracy/runtime/memory on one system/N."""

    system_name: str
    n_particles: int
    backend: str  # "numpy_direct", "taichi_direct", or "barnes_hut_cpu"
    theta: float  # only meaningful for barnes_hut_cpu; 0.0 elsewhere (unused)
    mean_relative_error: float  # vs. numpy_direct; exactly 0 for numpy_direct itself
    median_relative_error: float
    max_relative_error: float
    runtime_s: float
    speedup_vs_numpy_direct: float
    peak_rss_mb: float
    taichi_arch: str  # backend actually used by Taichi ("cpu", "cuda", ...); "" if not applicable

    def has_finite_values(self) -> bool:
        values = [
            self.mean_relative_error,
            self.median_relative_error,
            self.max_relative_error,
            self.runtime_s,
            self.speedup_vs_numpy_direct,
            self.peak_rss_mb,
        ]
        return all(np.isfinite(v) for v in values)


def warmup_taichi_backend(softening: float = 0.05, G: float = 1.0) -> float:
    """Run one throwaway Taichi call to absorb first-invocation kernel
    compilation overhead before any timed measurement happens.

    Taichi compiles a kernel once per unique argument *type signature*
    (dtype/ndim), not once per array size -- confirmed empirically while
    building this module: an N=100 call that includes compilation took
    ~0.68s, while an immediately following N=300 call (same kernel, already
    compiled) took ~0.002s. Without this warmup, whichever N happens to be
    benchmarked first would look artificially, misleadingly slow. This is
    exactly the pitfall this project's GPU documentation warns readers to
    watch for -- and it needs a real fix, not just a caveat, since letting
    it stand would silently distort every runtime/speedup number reported.

    Returns:
        The warmup call's wall-clock time in seconds -- discarded from all
        real measurements, returned only for optional logging/transparency.
    """
    warmup_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    warmup_masses = np.array([1.0, 1.0])
    t0 = time.perf_counter()
    taichi_direct_accelerations(warmup_positions, warmup_masses, softening, G)
    return time.perf_counter() - t0


def _timed_median(fn, args: tuple, kwargs: dict, repeats: int):
    """Call fn(*args, **kwargs) `repeats` times, return (last_result, median_runtime_s)."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def _relative_errors(reference: np.ndarray, other: np.ndarray) -> tuple[float, float, float]:
    ref_norms = np.linalg.norm(reference, axis=1)
    safe_norms = np.where(ref_norms > 0, ref_norms, 1.0)
    rel = np.linalg.norm(other - reference, axis=1) / safe_norms
    return float(np.mean(rel)), float(np.median(rel)), float(np.max(rel))


def evaluate_all_backends(
    system_name: str,
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    theta: float = 0.5,
    G: float = 1.0,
    repeats: int = 3,
) -> list[GPUValidationResult]:
    """Run numpy_direct, taichi_direct, and barnes_hut_cpu once each (with
    repeated timing) on the same system, comparing the latter two to
    numpy_direct.
    """
    from physics.taichi_forces import taichi_backend_info

    numpy_accel, t_numpy = _timed_median(
        compute_accelerations, (positions, masses, softening, G), {}, repeats
    )
    rss_numpy = _peak_rss_mb()

    taichi_accel, t_taichi = _timed_median(
        taichi_direct_accelerations, (positions, masses, softening, G), {}, repeats
    )
    rss_taichi = _peak_rss_mb()
    arch = taichi_backend_info()["arch"]

    bh_accel, t_bh = _timed_median(
        barnes_hut_accelerations, (positions, masses), {"G": G, "softening": softening, "theta": theta}, repeats
    )
    rss_bh = _peak_rss_mb()

    n = positions.shape[0]
    results = [
        GPUValidationResult(
            system_name=system_name,
            n_particles=n,
            backend="numpy_direct",
            theta=0.0,
            mean_relative_error=0.0,
            median_relative_error=0.0,
            max_relative_error=0.0,
            runtime_s=t_numpy,
            speedup_vs_numpy_direct=1.0,
            peak_rss_mb=rss_numpy,
            taichi_arch="",
        )
    ]

    mean_e, median_e, max_e = _relative_errors(numpy_accel, taichi_accel)
    results.append(
        GPUValidationResult(
            system_name=system_name,
            n_particles=n,
            backend="taichi_direct",
            theta=0.0,
            mean_relative_error=mean_e,
            median_relative_error=median_e,
            max_relative_error=max_e,
            runtime_s=t_taichi,
            speedup_vs_numpy_direct=t_numpy / max(t_taichi, 1e-12),
            peak_rss_mb=rss_taichi,
            taichi_arch=arch,
        )
    )

    mean_e, median_e, max_e = _relative_errors(numpy_accel, bh_accel)
    results.append(
        GPUValidationResult(
            system_name=system_name,
            n_particles=n,
            backend="barnes_hut_cpu",
            theta=theta,
            mean_relative_error=mean_e,
            median_relative_error=median_e,
            max_relative_error=max_e,
            runtime_s=t_bh,
            speedup_vs_numpy_direct=t_numpy / max(t_bh, 1e-12),
            peak_rss_mb=rss_bh,
            taichi_arch="",
        )
    )
    return results


def run_particle_scaling_validation(
    system_name: str,
    factory: PositionsMassesFactory,
    particle_counts: list[int],
    softening: float,
    theta: float = 0.5,
    repeats: int = 3,
    seed: int = 0,
) -> list[GPUValidationResult]:
    """Run all three backends across a range of particle counts N.

    Calls `warmup_taichi_backend` once, before any measured N, so Taichi's
    one-time kernel compilation cost never contaminates a real measurement
    (see that function's docstring for why this matters and how it was found).
    """
    warmup_time = warmup_taichi_backend(softening)
    print(f"  (Taichi warmup call: {warmup_time:.3f}s, excluded from all measurements below)")

    all_results: list[GPUValidationResult] = []
    for n in particle_counts:
        positions, masses = factory(n, seed)
        all_results.extend(
            evaluate_all_backends(system_name, positions, masses, softening, theta=theta, repeats=repeats)
        )
    return all_results


def results_to_dataframe(results: list[GPUValidationResult]):
    if pd is None:
        raise ImportError(
            "pandas is required for results_to_dataframe(); install it or use "
            "save_results_csv() instead, which has no pandas dependency."
        )
    return pd.DataFrame([asdict(r) for r in results])


def save_results_csv(results: list[GPUValidationResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("")
        return

    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
