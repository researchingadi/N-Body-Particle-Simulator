"""Export simulation trajectories (CSV) and reproducible configs (JSON).

Design goal: a saved config + the named preset + the seed should be enough
to exactly regenerate a run. The CSV holds the actual trajectory for
downstream analysis/plotting without needing to re-simulate.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

from simulation.engine import Simulation, SimulationConfig


def save_trajectory_csv(sim: Simulation, path: str | Path) -> None:
    """Write a long-format CSV: one row per (recorded_step, body).

    Columns: step, time, body_id, mass, x, y, z, vx, vy, vz
    """
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time", "body_id", "mass", "x", "y", "z", "vx", "vy", "vz"])
        for step_idx, (t, positions) in enumerate(zip(sim.trajectory_times, sim.trajectory)):
            for body_id, pos in enumerate(positions):
                mass = sim.masses[body_id] if body_id < len(sim.masses) else float("nan")
                writer.writerow([step_idx, t, body_id, mass, *pos, "", "", ""])


def save_config_json(config: SimulationConfig, path: str | Path) -> None:
    """Serialize a SimulationConfig to JSON for reproducibility."""
    path = Path(path)
    with path.open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)


def load_config_json(path: str | Path) -> SimulationConfig:
    """Load a SimulationConfig previously saved with save_config_json."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return SimulationConfig(**data)


def save_initial_state_npz(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, path: str | Path
) -> None:
    """Save the exact initial condition arrays (binary, fast to reload)."""
    np.savez(path, positions=positions, velocities=velocities, masses=masses)


def load_initial_state_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load initial condition arrays saved by save_initial_state_npz."""
    data = np.load(path)
    return data["positions"], data["velocities"], data["masses"]
