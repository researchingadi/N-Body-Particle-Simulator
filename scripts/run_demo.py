"""Runnable demo: binary orbit + Plummer sphere, with plots and CSV export.

Usage:
    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from diagnostics.conservation import relative_energy_drift
from io_utils.export import save_config_json, save_trajectory_csv
from initial_conditions.presets import binary_orbit, plummer_sphere
from simulation.engine import Simulation, SimulationConfig
from visualization.plot import (
    plot_energy_diagnostics,
    plot_momentum_and_angular_momentum,
    plot_trajectories_2d,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run_binary_demo() -> None:
    positions, velocities, masses = binary_orbit(separation=2.0)
    config = SimulationConfig(dt=0.005, softening=0.01, integrator="leapfrog", preset_name="binary_orbit")
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=4000, record_every=10)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_trajectories_2d(sim, OUTPUT_DIR / "binary_trajectories.png", title="Binary orbit")
    plot_energy_diagnostics(sim, OUTPUT_DIR / "binary_energy.png")
    plot_momentum_and_angular_momentum(sim, OUTPUT_DIR / "binary_momentum.png")
    save_trajectory_csv(sim, OUTPUT_DIR / "binary_trajectory.csv")
    save_config_json(config, OUTPUT_DIR / "binary_config.json")

    drift = relative_energy_drift(np.array(sim.history.total_energy))
    print(f"[binary_orbit] steps={sim.step_count} relative energy drift={drift:.3e}")


def run_plummer_demo() -> None:
    positions, velocities, masses = plummer_sphere(n=200, seed=0)
    config = SimulationConfig(dt=0.01, softening=0.1, integrator="leapfrog", preset_name="plummer_sphere", seed=0)
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=500, record_every=5)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_trajectories_2d(sim, OUTPUT_DIR / "plummer_trajectories.png", title="Plummer sphere (200 bodies)")
    plot_energy_diagnostics(sim, OUTPUT_DIR / "plummer_energy.png")
    save_config_json(config, OUTPUT_DIR / "plummer_config.json")

    drift = relative_energy_drift(np.array(sim.history.total_energy))
    print(f"[plummer_sphere] steps={sim.step_count} relative energy drift={drift:.3e}")


if __name__ == "__main__":
    run_binary_demo()
    run_plummer_demo()
