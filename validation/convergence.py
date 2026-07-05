"""Convergence and validation sweeps for the N-body engine.

This module runs the same benchmark system repeatedly under varying
timestep (dt) or softening, holding total simulated physical time fixed,
and reports how conservation diagnostics respond. It does not modify or
replace the `Simulation` class -- it is a thin harness around it, reusing
the existing engine, integrators, and diagnostics exactly as-is.

Scientific framing (read before interpreting results):

- Leapfrog and velocity Verlet are symplectic. They do NOT drive energy
  error to zero as a function of *time* -- they keep it bounded and
  oscillatory, conserving a nearby "shadow" Hamiltonian instead of the true
  one. What SHOULD improve as dt shrinks is the *size* of that bounded
  oscillation (local truncation error), which for leapfrog is O(dt^2) per
  step / O(dt^2) globally.
- Euler is not symplectic. Its energy error is expected to grow secularly
  (roughly monotonically) over time regardless of dt, just more slowly for
  smaller dt. It is included here only as a known-bad control, never as a
  candidate for production use.
- "Relative energy drift" (final-step) and "max relative energy drift"
  (envelope over the whole run) are both reported because they answer
  different questions -- see diagnostics.conservation for details.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is optional; CSV export still works without it
    pd = None  # type: ignore[assignment]

from diagnostics.conservation import (
    center_of_mass_drift,
    max_relative_energy_drift,
    relative_energy_drift,
    relative_vector_drift,
)
from initial_conditions.presets import (
    binary_orbit,
    disk_galaxy,
    plummer_sphere,
    three_body_figure_eight,
)
from simulation.engine import IntegratorName, Simulation, SimulationConfig

SystemFactory = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class BenchmarkSystem:
    """A named, reproducible test system plus recommended run parameters.

    The recommended values are starting points, not requirements -- sweeps
    override dt and/or softening explicitly; the recommendation is only
    used for whichever parameter a given sweep is holding fixed.
    """

    name: str
    factory: SystemFactory
    recommended_dt: float
    recommended_softening: float
    recommended_total_time: float
    G: float = 1.0


# Registry of benchmark systems used across the validation suite.
# Softening/dt values are chosen per-system by inspecting characteristic
# length and timescales (e.g. Plummer scale_radius=1 needs coarser softening
# than a binary orbit at separation=2 needs, or single-step accelerations
# become unphysically large -- see the Stage 1 collision-test bug for what
# happens when this isn't respected).
BENCHMARK_SYSTEMS: dict[str, BenchmarkSystem] = {
    "binary_orbit": BenchmarkSystem(
        name="binary_orbit",
        factory=lambda: binary_orbit(separation=2.0),
        recommended_dt=0.01,
        recommended_softening=0.01,
        recommended_total_time=20.0,  # several orbital periods
    ),
    "figure_eight": BenchmarkSystem(
        name="figure_eight",
        factory=three_body_figure_eight,
        recommended_dt=0.001,
        recommended_softening=0.001,
        recommended_total_time=6.28,  # ~ one period of the figure-eight solution
    ),
    "plummer_sphere": BenchmarkSystem(
        name="plummer_sphere",
        factory=lambda: plummer_sphere(n=100, seed=0),
        recommended_dt=0.01,
        recommended_softening=0.1,
        recommended_total_time=5.0,
    ),
    "disk_galaxy": BenchmarkSystem(
        name="disk_galaxy",
        factory=lambda: disk_galaxy(n=100, seed=0),
        recommended_dt=0.02,
        recommended_softening=0.2,
        recommended_total_time=10.0,
    ),
}


@dataclass
class ConvergenceResult:
    """One row of a convergence study: one (system, integrator, dt, softening) run."""

    system_name: str
    integrator: str
    dt: float
    softening: float
    n_steps: int
    actual_total_time: float
    initial_energy: float
    final_energy: float
    relative_energy_drift: float
    max_relative_energy_drift: float
    relative_angular_momentum_drift: float
    center_of_mass_drift: float

    def has_finite_values(self) -> bool:
        """True if none of the numeric diagnostic fields are NaN/inf."""
        numeric_fields = [
            self.initial_energy,
            self.final_energy,
            self.relative_energy_drift,
            self.max_relative_energy_drift,
            self.relative_angular_momentum_drift,
            self.center_of_mass_drift,
        ]
        return all(np.isfinite(v) for v in numeric_fields)


def run_single_case(
    system_name: str,
    factory: SystemFactory,
    dt: float,
    softening: float,
    integrator: IntegratorName,
    total_time: float,
    G: float = 1.0,
    seed: int = 0,
    record_every: int = 1,
) -> ConvergenceResult:
    """Run one simulation and summarize its conservation behavior.

    `total_time` is held fixed across a sweep by deriving n_steps = total_time
    / dt for each dt, so every run in a dt-sweep covers the same physical
    time span (the point of the sweep is isolating the effect of dt, not
    confounding it with "ran for a different amount of physical time").
    """
    positions, velocities, masses = factory()
    n_steps = max(1, round(total_time / dt))

    config = SimulationConfig(
        dt=dt,
        softening=softening,
        G=G,
        integrator=integrator,
        seed=seed,
        preset_name=system_name,
    )
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=n_steps, record_every=record_every)

    hist = sim.history.as_arrays()
    energies = hist["total_energy"]

    return ConvergenceResult(
        system_name=system_name,
        integrator=integrator,
        dt=dt,
        softening=softening,
        n_steps=sim.step_count,
        actual_total_time=sim.time,
        initial_energy=float(energies[0]),
        final_energy=float(energies[-1]),
        relative_energy_drift=relative_energy_drift(energies),
        max_relative_energy_drift=max_relative_energy_drift(energies),
        relative_angular_momentum_drift=relative_vector_drift(hist["angular_momentum"]),
        center_of_mass_drift=center_of_mass_drift(hist["center_of_mass"]),
    )


def sweep_timestep(
    system_name: str,
    factory: SystemFactory,
    dt_values: Sequence[float],
    total_time: float,
    softening: float,
    integrator: IntegratorName = "leapfrog",
    G: float = 1.0,
    seed: int = 0,
    record_every: int = 1,
) -> list[ConvergenceResult]:
    """Run the same system across multiple dt values, softening held fixed."""
    return [
        run_single_case(system_name, factory, dt, softening, integrator, total_time, G, seed, record_every)
        for dt in dt_values
    ]


def sweep_softening(
    system_name: str,
    factory: SystemFactory,
    softening_values: Sequence[float],
    dt: float,
    total_time: float,
    integrator: IntegratorName = "leapfrog",
    G: float = 1.0,
    seed: int = 0,
    record_every: int = 1,
) -> list[ConvergenceResult]:
    """Run the same system across multiple softening values, dt held fixed."""
    return [
        run_single_case(system_name, factory, dt, softening, integrator, total_time, G, seed, record_every)
        for softening in softening_values
    ]


def compare_integrators(
    system_name: str,
    factory: SystemFactory,
    integrators: Sequence[IntegratorName],
    dt: float,
    softening: float,
    total_time: float,
    G: float = 1.0,
    seed: int = 0,
    record_every: int = 1,
) -> list[ConvergenceResult]:
    """Run the same system/dt/softening across multiple integrators.

    The intended use is a direct leapfrog-vs-Euler comparison at matched
    dt, to show Euler's known-bad behavior is a property of the integrator,
    not of an unfairly coarse timestep.
    """
    return [
        run_single_case(system_name, factory, dt, softening, integrator, total_time, G, seed, record_every)
        for integrator in integrators
    ]


def results_to_dataframe(results: list[ConvergenceResult]):
    """Convert a list of ConvergenceResult to a pandas DataFrame, if pandas
    is installed. Raises ImportError with a clear message otherwise.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for results_to_dataframe(); install it or use "
            "save_results_csv() instead, which has no pandas dependency."
        )
    return pd.DataFrame([asdict(r) for r in results])


def save_results_csv(results: list[ConvergenceResult], path: str | Path) -> None:
    """Save convergence results to CSV without requiring pandas."""
    import csv

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
