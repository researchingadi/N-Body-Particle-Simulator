"""In-memory simulation registry, wrapping simulation.engine.Simulation.

Design decision worth stating explicitly: this manager calls `sim.step()`
directly in a loop, never `Simulation.run()`. `run()` (via `record()`)
accumulates a full diagnostics history and position trajectory in memory on
every recorded step -- exactly what a batch script (run_demo.py,
run_convergence_study.py, ...) wants for post-hoc plotting, but the wrong
behavior for a live API where a simulation might be stepped many times
across many small requests over a long-lived server process: that history
would grow without bound for as long as the simulation stays alive.
Diagnostics are instead computed fresh, on demand, directly from current
particle state via diagnostics.conservation -- O(1) memory per simulation
regardless of how many times it's been stepped or queried.
"""
from __future__ import annotations

import uuid

import numpy as np

from diagnostics.conservation import (
    center_of_mass,
    kinetic_energy,
    potential_energy,
    total_angular_momentum,
    total_momentum,
)
from simulation.engine import Simulation, SimulationConfig


class SimulationNotFoundError(Exception):
    """Raised when a simulation_id doesn't exist in the registry."""

    def __init__(self, simulation_id: str) -> None:
        self.simulation_id = simulation_id
        super().__init__(f"No simulation found with id={simulation_id!r}")


class SimulationManager:
    """Owns the in-memory simulation registry.

    Encapsulated in a class, rather than bare module-level globals, so the
    registry has exactly one clear owner, is trivially resettable in tests
    (just construct a new instance), and could later be swapped for a
    persistent backing store (Redis, a database) without changing any API
    route handler, which only ever calls this class's methods.
    """

    def __init__(self) -> None:
        self._simulations: dict[str, Simulation] = {}

    def create(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        config: SimulationConfig,
    ) -> str:
        """Create a new simulation, store it, and return its id.

        Simulation's own `_validate_shapes` (asserts on positions/velocities/
        masses shapes and positive masses) runs as part of construction, so
        malformed initial conditions surface here as an AssertionError --
        the API layer translates that into an HTTP 400 (see api/main.py).
        """
        sim = Simulation(positions, velocities, masses, config)
        simulation_id = str(uuid.uuid4())
        self._simulations[simulation_id] = sim
        return simulation_id

    def get(self, simulation_id: str) -> Simulation:
        """Look up a simulation by id, or raise SimulationNotFoundError."""
        sim = self._simulations.get(simulation_id)
        if sim is None:
            raise SimulationNotFoundError(simulation_id)
        return sim

    def step(self, simulation_id: str, n_steps: int) -> Simulation:
        """Advance a simulation by n_steps, calling Simulation.step directly
        (not Simulation.run) -- see this module's docstring for why.
        """
        sim = self.get(simulation_id)
        for _ in range(n_steps):
            sim.step()
        return sim

    def delete(self, simulation_id: str) -> None:
        """Remove a simulation from the registry, or raise SimulationNotFoundError."""
        if simulation_id not in self._simulations:
            raise SimulationNotFoundError(simulation_id)
        del self._simulations[simulation_id]

    def count(self) -> int:
        """Number of simulations currently held in memory."""
        return len(self._simulations)


def compute_live_diagnostics(sim: Simulation) -> dict:
    """Compute diagnostics from a simulation's CURRENT state (not its
    history), returning plain Python types ready to hand to a Pydantic
    response model.
    """
    ke = kinetic_energy(sim.masses, sim.velocities)
    pe = potential_energy(sim.positions, sim.masses, sim.config.softening, sim.config.G)
    momentum = total_momentum(sim.masses, sim.velocities)
    angular_momentum = total_angular_momentum(sim.positions, sim.masses, sim.velocities)
    com = center_of_mass(sim.positions, sim.masses)
    return {
        "time": sim.time,
        "kinetic_energy": ke,
        "potential_energy": pe,
        "total_energy": ke + pe,
        "momentum": momentum.tolist(),
        "angular_momentum": angular_momentum.tolist(),
        "center_of_mass": com.tolist(),
    }
