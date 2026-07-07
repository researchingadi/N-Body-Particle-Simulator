"""Tests for physics.taichi_forces: the Taichi-accelerated direct solver.

Unlike Barnes-Hut, this backend is NOT an approximation -- it computes the
exact same O(N^2) sum as physics.forces.compute_accelerations, just on
parallel threads. So unlike tests/test_barnes_hut.py, these tests hold it
to the same tight numerical bar as the reference solver itself (agreement
to floating-point noise, not a tolerance band), while separately checking
that backend selection/fallback behaves safely regardless of what hardware
is actually present.
"""
import numpy as np
import pytest

from physics.forces import compute_accelerations
from physics.taichi_forces import init_taichi, taichi_backend_info, taichi_direct_accelerations


def _random_cloud(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-5.0, 5.0, size=(n, 3))
    masses = rng.uniform(0.5, 2.0, size=n)
    return positions, masses


def test_output_shape_matches_direct_solver():
    positions, masses = _random_cloud(50)
    ti_accel = taichi_direct_accelerations(positions, masses, softening=0.05)
    direct = compute_accelerations(positions, masses, softening=0.05)
    assert ti_accel.shape == direct.shape == (50, 3)


def test_no_nan_or_infinite_accelerations():
    positions, masses = _random_cloud(100)
    ti_accel = taichi_direct_accelerations(positions, masses, softening=0.05)
    assert np.all(np.isfinite(ti_accel))


def test_single_body_has_zero_acceleration():
    positions = np.array([[1.0, 2.0, 3.0]])
    masses = np.array([4.0])
    ti_accel = taichi_direct_accelerations(positions, masses, softening=0.05)
    assert np.allclose(ti_accel, 0.0)


def test_two_body_matches_direct_very_closely():
    """This is not an approximation, so tolerance is tight (float noise only)."""
    positions = np.array([[0.0, 0.0, 0.0], [1.3, -0.7, 0.4]])
    masses = np.array([1.0, 2.5])
    ti_accel = taichi_direct_accelerations(positions, masses, softening=0.05)
    direct = compute_accelerations(positions, masses, softening=0.05)
    assert np.allclose(ti_accel, direct, atol=1e-10, rtol=1e-10)


def test_agrees_with_direct_solver_within_numerical_tolerance():
    """Same O(N^2) sum, different execution strategy -- expect agreement to
    floating-point summation-order noise, not a Barnes-Hut-style error band.
    """
    positions, masses = _random_cloud(200, seed=1)
    softening = 0.05
    ti_accel = taichi_direct_accelerations(positions, masses, softening=softening)
    direct = compute_accelerations(positions, masses, softening=softening)

    direct_norms = np.linalg.norm(direct, axis=1)
    relative_error = np.linalg.norm(ti_accel - direct, axis=1) / direct_norms
    assert np.max(relative_error) < 1e-8, f"unexpectedly large discrepancy: {np.max(relative_error)}"


def test_agrees_with_direct_solver_on_clustered_system():
    """Distinct from the uniform random cloud above: a Plummer-like clustered
    distribution stresses close-range pairs (small separations, large
    individual force contributions) differently than a uniform cloud does.
    """
    from initial_conditions.presets import plummer_sphere

    positions, _velocities, masses = plummer_sphere(n=150, seed=2)
    softening = 0.1
    ti_accel = taichi_direct_accelerations(positions, masses, softening=softening)
    direct = compute_accelerations(positions, masses, softening=softening)

    direct_norms = np.linalg.norm(direct, axis=1)
    relative_error = np.linalg.norm(ti_accel - direct, axis=1) / direct_norms
    assert np.max(relative_error) < 1e-8


def test_backend_info_reports_taichi_installed_and_initialized():
    """Sanity check that the module can actually report its own state --
    used by the validation script/README to record what hardware ran the
    benchmark.
    """
    init_taichi()
    info = taichi_backend_info()
    assert info["taichi_installed"] is True
    assert info["initialized"] is True
    assert info["arch"] in ("cuda", "vulkan", "metal", "cpu")


def test_backend_fallback_does_not_crash_when_gpu_unavailable():
    """`init_taichi` must be safe to call regardless of what GPU hardware
    (if any) is actually present -- it should always return a valid arch
    name, never raise, and never crash the process. This is the direct
    regression test for the segfault-on-broken-GPU-probe issue found while
    building this module (see physics/taichi_forces.py's module docstring).
    """
    arch = init_taichi(prefer_gpu=True)
    assert arch in ("cuda", "vulkan", "metal", "cpu", "unavailable")

    # Once initialized, forcing prefer_gpu=False should just return the
    # cached result (Taichi doesn't support cheap re-init), not raise.
    arch_again = init_taichi(prefer_gpu=False)
    assert arch_again == arch


def test_simulation_can_run_with_taichi_backend():
    """End-to-end: Simulation should drive real dynamics with
    solver='taichi_direct', producing (for this non-approximating backend)
    a trajectory and energy-drift behavior matching the direct solver.
    """
    from diagnostics.conservation import relative_energy_drift
    from initial_conditions.presets import binary_orbit
    from simulation.engine import Simulation, SimulationConfig

    positions, velocities, masses = binary_orbit(separation=2.0)

    direct_sim = Simulation(
        positions, velocities, masses, SimulationConfig(dt=0.005, softening=0.01, solver="direct")
    )
    direct_sim.run(n_steps=500, record_every=10)

    taichi_sim = Simulation(
        positions, velocities, masses, SimulationConfig(dt=0.005, softening=0.01, solver="taichi_direct")
    )
    taichi_sim.run(n_steps=500, record_every=10)

    assert np.allclose(direct_sim.positions, taichi_sim.positions, atol=1e-6)

    taichi_drift = relative_energy_drift(np.array(taichi_sim.history.total_energy))
    assert abs(taichi_drift) < 1e-6, f"taichi_direct energy drift unexpectedly large: {taichi_drift}"
