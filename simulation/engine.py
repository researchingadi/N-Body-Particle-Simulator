"""Simulation orchestration.

This module owns the mutable simulation state and the stepping loop. It
intentionally knows nothing about visualization or the API layer -- those
consume `Simulation` as a plain object. Three force backends are wired in:
direct O(N^2) (physics.forces, the validated reference), Barnes-Hut
(physics.barnes_hut, an approximation traded for scalability), and Taichi
direct (physics.taichi_forces, the same O(N^2) sum as the reference, just
evaluated on parallel GPU/CPU threads). All three implement the same
`accel_fn(positions, masses) -> accelerations` signature, selected via
`SimulationConfig.solver`, so a future backend can be added the same way
without touching the stepping loop below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from diagnostics.conservation import (
    center_of_mass,
    kinetic_energy,
    potential_energy,
    total_angular_momentum,
    total_momentum,
)
from physics.barnes_hut import barnes_hut_accelerations
from physics.forces import compute_accelerations
from physics.integrators import euler_step, leapfrog_kdk_step, rk4_step, velocity_verlet_step
from physics.taichi_forces import taichi_direct_accelerations

IntegratorName = Literal["euler", "leapfrog", "velocity_verlet", "rk4"]
SolverName = Literal["direct", "barnes_hut", "taichi_direct"]


@dataclass
class SimulationConfig:
    """Reproducible description of a simulation run.

    Storing this alongside output data (see io_utils.export) is what makes a
    run reproducible: same config + same seed => same trajectory.
    """

    dt: float = 0.01
    softening: float = 0.05
    G: float = 1.0
    integrator: IntegratorName = "leapfrog"
    solver: SolverName = "direct"
    theta: float = 0.5  # Barnes-Hut opening angle; unused for "direct"/"taichi_direct"
    seed: int = 0
    enable_collisions: bool = False
    merge_distance: float = 0.05
    preset_name: str = "custom"


@dataclass
class DiagnosticsHistory:
    """Time series of conserved quantities, recorded once per step."""

    time: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    momentum: list[np.ndarray] = field(default_factory=list)
    angular_momentum: list[np.ndarray] = field(default_factory=list)
    center_of_mass: list[np.ndarray] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time": np.array(self.time),
            "kinetic": np.array(self.kinetic),
            "potential": np.array(self.potential),
            "total_energy": np.array(self.total_energy),
            "momentum": np.array(self.momentum),
            "angular_momentum": np.array(self.angular_momentum),
            "center_of_mass": np.array(self.center_of_mass),
        }


class Simulation:
    """Mutable N-body simulation state plus a stepping loop.

    Example:
        sim = Simulation(positions, velocities, masses, SimulationConfig())
        sim.run(n_steps=1000, record_every=5)
        history = sim.history.as_arrays()
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        config: SimulationConfig | None = None,
    ) -> None:
        self.positions = np.array(positions, dtype=np.float64, copy=True)
        self.velocities = np.array(velocities, dtype=np.float64, copy=True)
        self.masses = np.array(masses, dtype=np.float64, copy=True)
        self.config = config or SimulationConfig()
        self.time: float = 0.0
        self.step_count: int = 0
        self.history = DiagnosticsHistory()
        self.trajectory: list[np.ndarray] = []  # snapshots of positions, one per record()
        self.trajectory_times: list[float] = []
        self._prev_accel: np.ndarray | None = None  # cache for KDK/Verlet

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = self.positions.shape[0]
        assert self.positions.shape == (n, 3), "positions must be (N, 3)"
        assert self.velocities.shape == (n, 3), "velocities must be (N, 3)"
        assert self.masses.shape == (n,), "masses must be (N,)"
        assert np.all(self.masses > 0), "masses must be positive"

    def _accel_fn(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        if self.config.solver == "direct":
            return compute_accelerations(positions, masses, self.config.softening, self.config.G)
        elif self.config.solver == "barnes_hut":
            return barnes_hut_accelerations(
                positions, masses, G=self.config.G, softening=self.config.softening, theta=self.config.theta
            )
        elif self.config.solver == "taichi_direct":
            return taichi_direct_accelerations(positions, masses, self.config.softening, self.config.G)
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")

    def step(self) -> None:
        """Advance the simulation by one timestep using the configured integrator."""
        dt = self.config.dt
        integrator = self.config.integrator

        if integrator == "euler":
            self.positions, self.velocities = euler_step(
                self.positions, self.velocities, self.masses, dt, self._accel_fn
            )
            self._prev_accel = None
        elif integrator == "leapfrog":
            self.positions, self.velocities, self._prev_accel = leapfrog_kdk_step(
                self.positions, self.velocities, self.masses, dt, self._accel_fn, self._prev_accel
            )
        elif integrator == "velocity_verlet":
            self.positions, self.velocities, self._prev_accel = velocity_verlet_step(
                self.positions, self.velocities, self.masses, dt, self._accel_fn, self._prev_accel
            )
        elif integrator == "rk4":
            self.positions, self.velocities = rk4_step(
                self.positions, self.velocities, self.masses, dt, self._accel_fn
            )
            self._prev_accel = None
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        if self.config.enable_collisions:
            self._resolve_collisions()

        self.time += dt
        self.step_count += 1

    def _resolve_collisions(self) -> None:
        """Detect close pairs and merge them inelastically (mass/momentum-conserving).

        Simple O(N^2) pairwise distance check. Merged body position/velocity
        are computed as the mass-weighted (center-of-mass) average, which
        exactly conserves total mass and total momentum by construction.
        This is a baseline collision model -- no fragmentation, no energy
        bookkeeping for the lost kinetic energy (assumed radiated away).
        """
        n = self.positions.shape[0]
        if n < 2:
            return

        merge_dist = self.config.merge_distance
        keep = np.ones(n, dtype=bool)

        i = 0
        while i < n:
            if not keep[i]:
                i += 1
                continue
            j = i + 1
            while j < n:
                if keep[j]:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < merge_dist:
                        m_i, m_j = self.masses[i], self.masses[j]
                        total_m = m_i + m_j
                        self.positions[i] = (m_i * self.positions[i] + m_j * self.positions[j]) / total_m
                        self.velocities[i] = (m_i * self.velocities[i] + m_j * self.velocities[j]) / total_m
                        self.masses[i] = total_m
                        keep[j] = False
                j += 1
            i += 1

        if not np.all(keep):
            self.positions = self.positions[keep]
            self.velocities = self.velocities[keep]
            self.masses = self.masses[keep]
            self._prev_accel = None  # shape changed, force recompute next step

    def record(self) -> None:
        """Append current diagnostics to `self.history`."""
        ke = kinetic_energy(self.masses, self.velocities)
        pe = potential_energy(self.positions, self.masses, self.config.softening, self.config.G)
        self.history.time.append(self.time)
        self.history.kinetic.append(ke)
        self.history.potential.append(pe)
        self.history.total_energy.append(ke + pe)
        self.history.momentum.append(total_momentum(self.masses, self.velocities))
        self.history.angular_momentum.append(
            total_angular_momentum(self.positions, self.masses, self.velocities)
        )
        self.history.center_of_mass.append(center_of_mass(self.positions, self.masses))
        self.trajectory.append(self.positions.copy())
        self.trajectory_times.append(self.time)

    def run(self, n_steps: int, record_every: int = 1) -> None:
        """Run `n_steps` integration steps, recording diagnostics periodically."""
        self.record()  # record initial state at t=0
        for step_idx in range(1, n_steps + 1):
            self.step()
            if step_idx % record_every == 0:
                self.record()
