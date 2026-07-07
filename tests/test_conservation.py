"""Tests for conservation laws under integration.

This is where "is my physics actually correct" gets checked end-to-end:
momentum conservation, bounded energy drift for symplectic integrators, and
long-term stability of a known circular orbit.
"""
import numpy as np

from diagnostics.conservation import relative_energy_drift, total_momentum
from initial_conditions.presets import binary_orbit, three_body_figure_eight
from simulation.engine import Simulation, SimulationConfig


def test_momentum_conserved_leapfrog():
    """Total linear momentum should be ~exactly conserved (no external forces)."""
    positions, velocities, masses = binary_orbit(separation=2.0)
    config = SimulationConfig(dt=0.005, softening=0.01, integrator="leapfrog")
    sim = Simulation(positions, velocities, masses, config)

    p0 = total_momentum(sim.masses, sim.velocities)
    sim.run(n_steps=2000, record_every=100)
    p1 = total_momentum(sim.masses, sim.velocities)

    assert np.allclose(p0, p1, atol=1e-10)


def test_leapfrog_energy_drift_bounded():
    """Leapfrog should keep relative energy drift small over a moderate run."""
    positions, velocities, masses = binary_orbit(separation=2.0)
    config = SimulationConfig(dt=0.005, softening=0.01, integrator="leapfrog")
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=4000, record_every=20)

    energies = sim.history.as_arrays()["total_energy"]
    drift = abs(relative_energy_drift(energies))
    assert drift < 0.01, f"leapfrog energy drift too large: {drift}"


def test_euler_drifts_more_than_leapfrog():
    """Sanity check that Euler is measurably worse than leapfrog (known-bad baseline)."""
    positions, velocities, masses = binary_orbit(separation=2.0)

    cfg_euler = SimulationConfig(dt=0.005, softening=0.01, integrator="euler")
    sim_euler = Simulation(positions, velocities, masses, cfg_euler)
    sim_euler.run(n_steps=2000, record_every=20)
    euler_drift = abs(relative_energy_drift(sim_euler.history.as_arrays()["total_energy"]))

    cfg_leap = SimulationConfig(dt=0.005, softening=0.01, integrator="leapfrog")
    sim_leap = Simulation(positions, velocities, masses, cfg_leap)
    sim_leap.run(n_steps=2000, record_every=20)
    leap_drift = abs(relative_energy_drift(sim_leap.history.as_arrays()["total_energy"]))

    assert euler_drift > leap_drift


def test_two_body_orbit_stays_bounded():
    """A circular two-body orbit should not fly apart or collapse over many periods."""
    separation = 2.0
    positions, velocities, masses = binary_orbit(separation=separation)
    config = SimulationConfig(dt=0.005, softening=0.01, integrator="leapfrog")
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=6000, record_every=50)

    trajectory = np.array(sim.trajectory)  # (steps, 2, 3)
    relative_dist = np.linalg.norm(trajectory[:, 0] - trajectory[:, 1], axis=1)

    # separation should stay within a reasonable band of the initial value
    assert np.all(relative_dist > separation * 0.5)
    assert np.all(relative_dist < separation * 1.5)


def test_figure_eight_three_body_energy_drift_bounded():
    """The figure-eight solution is a strong integrator stress test."""
    positions, velocities, masses = three_body_figure_eight()
    config = SimulationConfig(dt=0.001, softening=0.001, integrator="leapfrog")
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=6000, record_every=50)

    energies = sim.history.as_arrays()["total_energy"]
    drift = abs(relative_energy_drift(energies))
    assert drift < 0.02, f"figure-eight energy drift too large: {drift}"
    positions, velocities, masses = three_body_figure_eight()
    config = SimulationConfig(dt=0.001, softening=0.001, integrator="leapfrog")
    sim = Simulation(positions, velocities, masses, config)
    sim.run(n_steps=6000, record_every=50)

    energies = sim.history.as_arrays()["total_energy"]
    drift = abs(relative_energy_drift(energies))
    assert drift < 0.02, f"figure-eight energy drift too large: {drift}"
