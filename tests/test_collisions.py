"""Tests for inelastic collision/merger handling."""
import numpy as np

from simulation.engine import Simulation, SimulationConfig


def test_merger_conserves_mass_and_momentum():
    """Two overlapping bodies should merge into one, conserving total mass
    and total momentum exactly.

    Collision resolution is tested in isolation from the force integrator:
    at very small separations the direct-solver acceleration is enormous
    (by design -- that's what softening is for), so a single force step
    would fling the bodies apart before merge-distance detection could act
    on their original separation. Calling `_resolve_collisions` directly
    isolates the merge logic from that unrelated numerical effect.
    """
    positions = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    velocities = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    masses = np.array([1.0, 2.0])

    config = SimulationConfig(
        dt=0.01, softening=0.01, integrator="leapfrog", enable_collisions=True, merge_distance=0.1
    )
    sim = Simulation(positions, velocities, masses, config)

    total_mass_before = np.sum(sim.masses)
    total_p_before = np.sum(masses[:, np.newaxis] * velocities, axis=0)

    sim._resolve_collisions()

    assert sim.masses.shape[0] == 1, "expected the two bodies to merge into one"
    assert np.isclose(np.sum(sim.masses), total_mass_before)
    total_p_after = np.sum(sim.masses[:, np.newaxis] * sim.velocities, axis=0)
    assert np.allclose(total_p_after, total_p_before, atol=1e-8)


def test_no_merger_when_far_apart():
    """Bodies well outside merge_distance should remain separate."""
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])

    config = SimulationConfig(
        dt=0.01, softening=0.01, integrator="leapfrog", enable_collisions=True, merge_distance=0.1
    )
    sim = Simulation(positions, velocities, masses, config)
    sim.step()
    assert sim.masses.shape[0] == 2
