"""Tests for physics.forces: the ground-truth direct solver.

These must pass before anything else in the project is trusted -- Barnes-Hut
and GPU solvers will later be validated against this module's results.
"""
import numpy as np

from physics.forces import compute_accelerations, compute_potential_energy


def test_no_self_force():
    """A single isolated body feels zero acceleration."""
    positions = np.array([[0.0, 0.0, 0.0]])
    masses = np.array([5.0])
    acc = compute_accelerations(positions, masses, softening=0.01)
    assert np.allclose(acc, 0.0)


def test_force_symmetry_two_body():
    """Newton's third law: a_1 * m_1 == -a_2 * m_2 (equal and opposite force)."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([2.0, 3.0])
    acc = compute_accelerations(positions, masses, softening=0.01)
    force_on_1 = masses[0] * acc[0]
    force_on_2 = masses[1] * acc[1]
    assert np.allclose(force_on_1, -force_on_2, atol=1e-10)


def test_two_body_force_magnitude_matches_newton():
    """Direct solver should reproduce F = G*m1*m2/r^2 in the low-softening limit."""
    separation = 2.0
    positions = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    softening = 1e-6  # negligible vs separation
    acc = compute_accelerations(positions, masses, softening=softening, G=1.0)

    expected_accel_mag = 1.0 * masses[1] / separation**2  # a1 = G*m2/r^2
    assert np.isclose(np.linalg.norm(acc[0]), expected_accel_mag, rtol=1e-4)
    # body 0 should be pulled toward body 1 (+x direction)
    assert acc[0][0] > 0


def test_force_direction_toward_other_body():
    """Acceleration should point from a body toward a single attractor."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    masses = np.array([1.0, 10.0])
    acc = compute_accelerations(positions, masses, softening=0.001)
    assert acc[0][1] > 0  # pulled in +y toward body 1
    assert np.isclose(acc[0][0], 0.0, atol=1e-8)


def test_softening_prevents_singularity():
    """Coincident particles should not produce inf/nan acceleration."""
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    acc = compute_accelerations(positions, masses, softening=0.1)
    assert np.all(np.isfinite(acc))


def test_potential_energy_sign_is_negative_for_bound_system():
    """Gravitational PE of any multi-body system should be negative."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.array([1.0, 1.0, 1.0])
    pe = compute_potential_energy(positions, masses, softening=0.01)
    assert pe < 0
