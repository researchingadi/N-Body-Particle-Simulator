"""Barnes-Hut accuracy and performance validation against the direct solver.

This module answers two questions the README section explains in prose:
1. How much accuracy do we give up, as a function of opening angle theta?
2. What do we get back in runtime, as a function of particle count N?

The direct O(N^2) solver (physics.forces) is always the reference: "error"
here means "how far Barnes-Hut's acceleration is from direct summation's",
never the other way around.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is optional; CSV export still works without it
    pd = None  # type: ignore[assignment]

from initial_conditions.presets import disk_galaxy, plummer_sphere, random_cloud
from physics.barnes_hut import barnes_hut_accelerations
from physics.forces import compute_accelerations

# factory(n, seed) -> (positions, masses)
PositionsMassesFactory = Callable[[int, int], tuple[np.ndarray, np.ndarray]]


def _positions_and_masses_only(preset_triple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Drop the velocity array from a (positions, velocities, masses) preset.

    Force-solver accuracy/speed comparisons only need positions and masses;
    presets return velocities too since they're also used to launch actual
    dynamical simulations elsewhere.
    """
    positions, _velocities, masses = preset_triple
    return positions, masses


# Registry of systems used for Barnes-Hut validation, each with a
# recommended softening (same reasoning as validation.convergence's
# BENCHMARK_SYSTEMS: too-small softening on a clustered system produces
# huge, unrepresentative accelerations at close range).
BH_TEST_SYSTEMS: dict[str, tuple[PositionsMassesFactory, float]] = {
    "random_cloud": (
        lambda n, seed: _positions_and_masses_only(random_cloud(n=n, seed=seed)),
        0.05,
    ),
    "plummer_sphere": (
        lambda n, seed: _positions_and_masses_only(plummer_sphere(n=n, seed=seed)),
        0.1,
    ),
    "disk_galaxy": (
        lambda n, seed: _positions_and_masses_only(disk_galaxy(n=n, seed=seed)),
        0.1,
    ),
}


@dataclass
class AccuracyResult:
    """One row: accuracy + runtime of Barnes-Hut vs. direct on one system/N/theta."""

    system_name: str
    n_particles: int
    theta: float
    softening: float
    mean_relative_error: float
    median_relative_error: float
    max_relative_error: float
    direct_runtime_s: float
    barnes_hut_runtime_s: float
    speedup: float

    def has_finite_values(self) -> bool:
        values = [
            self.mean_relative_error,
            self.median_relative_error,
            self.max_relative_error,
            self.direct_runtime_s,
            self.barnes_hut_runtime_s,
            self.speedup,
        ]
        return all(np.isfinite(v) for v in values)


def evaluate_accuracy_and_speed(
    system_name: str,
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    theta: float,
    G: float = 1.0,
) -> AccuracyResult:
    """Run both solvers once on the same system and compare.

    Runtime is measured with a single call each, not an averaged benchmark
    loop -- fine for the accuracy sweeps (theta doesn't change runtime by
    the constant-factor amounts that matter for N-scaling claims). See
    `benchmark_particle_counts` for repeated-timing runtime benchmarks.
    """
    t0 = time.perf_counter()
    direct = compute_accelerations(positions, masses, softening, G)
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    bh = barnes_hut_accelerations(positions, masses, G=G, softening=softening, theta=theta)
    t_bh = max(time.perf_counter() - t0, 1e-9)  # guard against a zero-duration timer read

    direct_norms = np.linalg.norm(direct, axis=1)
    safe_norms = np.where(direct_norms > 0, direct_norms, 1.0)
    relative_error = np.linalg.norm(bh - direct, axis=1) / safe_norms

    return AccuracyResult(
        system_name=system_name,
        n_particles=positions.shape[0],
        theta=theta,
        softening=softening,
        mean_relative_error=float(np.mean(relative_error)),
        median_relative_error=float(np.median(relative_error)),
        max_relative_error=float(np.max(relative_error)),
        direct_runtime_s=t_direct,
        barnes_hut_runtime_s=t_bh,
        speedup=t_direct / t_bh,
    )


def sweep_theta(
    system_name: str,
    factory: PositionsMassesFactory,
    n: int,
    theta_values: list[float],
    softening: float,
    seed: int = 0,
) -> list[AccuracyResult]:
    """Fixed system, varying opening angle -- the core accuracy/speed tradeoff sweep."""
    positions, masses = factory(n, seed)
    return [
        evaluate_accuracy_and_speed(system_name, positions, masses, softening, theta)
        for theta in theta_values
    ]


def sweep_particle_count(
    system_name: str,
    factory: PositionsMassesFactory,
    particle_counts: list[int],
    theta: float,
    softening: float,
    seed: int = 0,
) -> list[AccuracyResult]:
    """Fixed opening angle, varying N -- for the runtime/speedup-vs-N plots."""
    results = []
    for n in particle_counts:
        positions, masses = factory(n, seed)
        results.append(evaluate_accuracy_and_speed(system_name, positions, masses, softening, theta))
    return results


def evaluate_accuracy_and_speed_repeated(
    system_name: str,
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float,
    theta: float,
    G: float = 1.0,
    repeats: int = 3,
) -> AccuracyResult:
    """Like `evaluate_accuracy_and_speed`, but times each solver `repeats`
    times and reports the median -- a single `perf_counter` read can be
    noisy for small/fast systems (OS scheduling jitter, cache effects), so
    this is the more reliable choice for runtime-scaling benchmarks where
    the actual timing numbers (not just pass/fail accuracy) get reported.
    """
    direct_times = []
    direct = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        direct = compute_accelerations(positions, masses, softening, G)
        direct_times.append(time.perf_counter() - t0)
    t_direct = float(np.median(direct_times))

    bh_times = []
    bh = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        bh = barnes_hut_accelerations(positions, masses, G=G, softening=softening, theta=theta)
        bh_times.append(time.perf_counter() - t0)
    t_bh = max(float(np.median(bh_times)), 1e-9)

    direct_norms = np.linalg.norm(direct, axis=1)
    safe_norms = np.where(direct_norms > 0, direct_norms, 1.0)
    relative_error = np.linalg.norm(bh - direct, axis=1) / safe_norms

    return AccuracyResult(
        system_name=system_name,
        n_particles=positions.shape[0],
        theta=theta,
        softening=softening,
        mean_relative_error=float(np.mean(relative_error)),
        median_relative_error=float(np.median(relative_error)),
        max_relative_error=float(np.max(relative_error)),
        direct_runtime_s=t_direct,
        barnes_hut_runtime_s=t_bh,
        speedup=t_direct / t_bh,
    )


def benchmark_particle_scaling(
    system_name: str,
    factory: PositionsMassesFactory,
    particle_counts: list[int],
    theta: float,
    softening: float,
    seed: int = 0,
    repeats: int = 3,
) -> list[AccuracyResult]:
    """Runtime/speedup vs. N, with repeated timing per N for stability.

    IMPORTANT (and worth stating explicitly, not just in the README): at
    small N, tree-construction overhead means Barnes-Hut can be SLOWER than
    direct summation despite its better asymptotic complexity. This
    function reports whatever the measured numbers are, including a
    speedup below 1.0 -- it does not filter or reframe an unfavorable
    result.
    """
    results = []
    for n in particle_counts:
        positions, masses = factory(n, seed)
        results.append(
            evaluate_accuracy_and_speed_repeated(system_name, positions, masses, softening, theta, repeats=repeats)
        )
    return results


def results_to_dataframe(results: list[AccuracyResult]):
    """Convert to a pandas DataFrame, if pandas is installed."""
    if pd is None:
        raise ImportError(
            "pandas is required for results_to_dataframe(); install it or use "
            "save_results_csv() instead, which has no pandas dependency."
        )
    return pd.DataFrame([asdict(r) for r in results])


def save_results_csv(results: list[AccuracyResult], path: str | Path) -> None:
    """Save accuracy/speed results to CSV without requiring pandas."""
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
