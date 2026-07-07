"""Tests for physics.barnes_hut.

Barnes-Hut is an *approximation* -- these tests check it behaves like one
should: shape-compatible with the direct solver, finite everywhere, close
to direct summation at small theta, and monotonically less accurate as
theta grows. They deliberately do NOT test exact pairwise force symmetry
or exact momentum conservation the way tests/test_forces.py does for the
direct solver -- Barnes-Hut's per-particle approximation error breaks that
guarantee by construction, and asserting it anyway would be testing for
behavior we don't actually want.
"""
import numpy as np

from physics.barnes_hut import barnes_hut_accelerations, build_octree
from physics.forces import compute_accelerations


def _random_cloud(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-5.0, 5.0, size=(n, 3))
    masses = rng.uniform(0.5, 2.0, size=n)
    return positions, masses


def test_output_shape_matches_direct_solver():
    positions, masses = _random_cloud(50)
    bh = barnes_hut_accelerations(positions, masses, softening=0.05, theta=0.5)
    direct = compute_accelerations(positions, masses, softening=0.05)
    assert bh.shape == direct.shape == (50, 3)


def test_no_nan_or_infinite_accelerations():
    positions, masses = _random_cloud(100)
    bh = barnes_hut_accelerations(positions, masses, softening=0.05, theta=0.5)
    assert np.all(np.isfinite(bh))


def test_single_body_has_zero_acceleration():
    positions = np.array([[1.0, 2.0, 3.0]])
    masses = np.array([4.0])
    bh = barnes_hut_accelerations(positions, masses, softening=0.05, theta=0.5)
    assert np.allclose(bh, 0.0)


def test_two_body_matches_direct_exactly():
    """With only 2 particles the tree can't approximate anything -- both
    solvers should agree to floating-point precision."""
    positions = np.array([[0.0, 0.0, 0.0], [1.3, -0.7, 0.4]])
    masses = np.array([1.0, 2.5])
    bh = barnes_hut_accelerations(positions, masses, softening=0.05, theta=0.5)
    direct = compute_accelerations(positions, masses, softening=0.05)
    assert np.allclose(bh, direct, atol=1e-10)


def test_barnes_hut_close_to_direct_for_small_theta():
    """Small theta should approximate direct summation closely."""
    positions, masses = _random_cloud(200)
    softening = 0.05
    bh = barnes_hut_accelerations(positions, masses, softening=softening, theta=0.2)
    direct = compute_accelerations(positions, masses, softening=softening)

    direct_norms = np.linalg.norm(direct, axis=1)
    relative_error = np.linalg.norm(bh - direct, axis=1) / direct_norms
    assert np.mean(relative_error) < 0.01, f"mean relative error too high: {np.mean(relative_error)}"
    assert np.median(relative_error) < 0.01


def test_smaller_theta_gives_lower_or_equal_error():
    """Accuracy should degrade monotonically (in aggregate) as theta grows,
    on a fixed, stable test case."""
    positions, masses = _random_cloud(200, seed=1)
    softening = 0.05
    direct = compute_accelerations(positions, masses, softening=softening)
    direct_norms = np.linalg.norm(direct, axis=1)

    thetas = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean_errors = []
    for theta in thetas:
        bh = barnes_hut_accelerations(positions, masses, softening=softening, theta=theta)
        rel_err = np.linalg.norm(bh - direct, axis=1) / direct_norms
        mean_errors.append(np.mean(rel_err))

    for smaller_theta_err, larger_theta_err in zip(mean_errors, mean_errors[1:]):
        assert larger_theta_err >= smaller_theta_err - 1e-9, (
            f"error should not shrink as theta grows: {mean_errors}"
        )
    # and the overall trend from smallest to largest theta should be a clear increase
    assert mean_errors[-1] > mean_errors[0]


def test_tree_total_mass_matches_sum_of_masses():
    positions, masses = _random_cloud(150, seed=2)
    root = build_octree(positions, masses)
    assert np.isclose(root.mass, np.sum(masses))


def test_tree_center_of_mass_matches_direct_calculation():
    positions, masses = _random_cloud(150, seed=3)
    root = build_octree(positions, masses)
    expected_com = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
    assert np.allclose(root.com, expected_com, atol=1e-10)


def test_tree_mass_and_com_correct_for_simple_hand_checkable_case():
    """Small enough to verify the expected numbers by hand."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    masses = np.array([1.0, 1.0, 2.0])
    root = build_octree(positions, masses)

    assert np.isclose(root.mass, 4.0)
    expected_com = (1.0 * np.array([0, 0, 0]) + 1.0 * np.array([2, 0, 0]) + 2.0 * np.array([0, 2, 0])) / 4.0
    assert np.allclose(root.com, expected_com)


def test_coincident_positions_do_not_crash_or_produce_nan():
    """Degenerate case: many particles at (near-)identical positions should
    hit the MAX_TREE_DEPTH fallback gracefully, not infinite-recurse or NaN."""
    n = 20
    positions = np.zeros((n, 3)) + np.array([1.0, 2.0, 3.0])
    positions += np.random.default_rng(4).normal(scale=1e-14, size=(n, 3))  # sub-precision jitter
    masses = np.full(n, 1.0)

    bh = barnes_hut_accelerations(positions, masses, softening=0.05, theta=0.5)
    assert np.all(np.isfinite(bh))
    # coincident equal masses should feel ~zero net force by symmetry (all
    # softened separations are ~0), not blow up
    assert np.max(np.abs(bh)) < 1e6
