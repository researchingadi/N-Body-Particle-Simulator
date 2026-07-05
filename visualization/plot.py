"""Minimal visualization layer using matplotlib.

This is deliberately not the cinematic frontend -- it exists so the engine
is visually checkable (does the orbit look right? is energy flat?) before
any frontend work starts. The React/Three.js frontend consumes exported
trajectory CSVs (see io_utils.export), not this module.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.engine import Simulation


def plot_trajectories_2d(sim: Simulation, path: str | Path, title: str = "Trajectories") -> None:
    """Plot x-y trajectories of all bodies over the recorded run."""
    trajectory = np.array(sim.trajectory)  # (steps, N, 3)
    n_bodies = trajectory.shape[1]

    fig, ax = plt.subplots(figsize=(7, 7))
    for body_id in range(n_bodies):
        ax.plot(trajectory[:, body_id, 0], trajectory[:, body_id, 1], linewidth=0.8)
        ax.scatter(trajectory[-1, body_id, 0], trajectory[-1, body_id, 1], s=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy_diagnostics(sim: Simulation, path: str | Path) -> None:
    """Plot KE, PE, and total energy vs. time -- the standard correctness check."""
    hist = sim.history.as_arrays()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["time"], hist["kinetic"], label="Kinetic")
    ax.plot(hist["time"], hist["potential"], label="Potential")
    ax.plot(hist["time"], hist["total_energy"], label="Total", linewidth=2, color="black")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.set_title("Energy diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_momentum_and_angular_momentum(sim: Simulation, path: str | Path) -> None:
    """Plot momentum and angular momentum component drift vs. time."""
    hist = sim.history.as_arrays()
    momentum = hist["momentum"]  # (steps, 3)
    angmom = hist["angular_momentum"]  # (steps, 3)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i, label in enumerate(["px", "py", "pz"]):
        axes[0].plot(hist["time"], momentum[:, i], label=label)
    axes[0].set_ylabel("linear momentum")
    axes[0].legend()

    for i, label in enumerate(["Lx", "Ly", "Lz"]):
        axes[1].plot(hist["time"], angmom[:, i], label=label)
    axes[1].set_ylabel("angular momentum")
    axes[1].set_xlabel("time")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
