"""Tests for initial-condition generators: shape correctness and physical sanity."""
import numpy as np

from diagnostics.conservation import center_of_mass_velocity
from initial_conditions.presets import (
    binary_orbit,
    disk_galaxy,
    plummer_sphere,
    ring_system,
    solar_system_like,
    three_body_figure_eight,
)


def test_binary_orbit_shapes_and_com_at_rest():
    positions, velocities, masses = binary_orbit(mass1=1.0, mass2=2.0, separation=3.0)
    assert positions.shape == (2, 3)
    assert velocities.shape == (2, 3)
    assert masses.shape == (2,)
    # COM should not be drifting for a symmetric two-body setup
    v_com = center_of_mass_velocity(velocities, masses)
    assert np.allclose(v_com, 0.0, atol=1e-10)


def test_three_body_figure_eight_shapes():
    positions, velocities, masses = three_body_figure_eight()
    assert positions.shape == (3, 3)
    assert velocities.shape == (3, 3)
    assert np.allclose(masses, 1.0)


def test_solar_system_like_shapes():
    positions, velocities, masses = solar_system_like()
    assert positions.shape[0] == velocities.shape[0] == masses.shape[0]
    assert masses[0] > np.max(masses[1:]), "central body should dominate mass"


def test_plummer_sphere_particle_count_and_finiteness():
    positions, velocities, masses = plummer_sphere(n=50, seed=42)
    assert positions.shape == (50, 3)
    assert np.all(np.isfinite(positions))
    assert np.all(np.isfinite(velocities))
    assert np.isclose(np.sum(masses), 100.0)  # default total_mass


def test_disk_galaxy_has_central_dominant_mass():
    positions, velocities, masses = disk_galaxy(n=100, seed=1)
    assert masses[0] > np.sum(masses[1:]), "central mass should dominate the disk"


def test_ring_system_particles_within_radius_band():
    positions, velocities, masses = ring_system(n=100, ring_radius=5.0, ring_width=0.3, seed=1)
    ring_positions = positions[1:]  # skip central body
    radii = np.linalg.norm(ring_positions[:, :2], axis=1)
    assert np.all(radii > 5.0 - 0.3 - 1e-6)
    assert np.all(radii < 5.0 + 0.3 + 1e-6)


def test_plummer_sphere_reproducible_with_seed():
    p1, v1, m1 = plummer_sphere(n=20, seed=7)
    p2, v2, m2 = plummer_sphere(n=20, seed=7)
    assert np.allclose(p1, p2)
    assert np.allclose(v1, v2)
