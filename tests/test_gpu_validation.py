"""Tests for validation.gpu_validation: the harness, not the solvers.

physics.taichi_forces's correctness is covered by tests/test_taichi_forces.py.
These tests check the harness built on top of it (and on top of the
existing direct/Barnes-Hut solvers) behaves as advertised.
"""
import numpy as np

from validation.gpu_validation import (
    GPU_VALIDATION_SYSTEMS,
    evaluate_all_backends,
    run_particle_scaling_validation,
    save_results_csv,
)


def test_evaluate_all_backends_returns_three_results():
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    positions, masses = factory(30, 0)
    results = evaluate_all_backends("random_cloud", positions, masses, softening, repeats=1)
    backends = {r.backend for r in results}
    assert backends == {"numpy_direct", "taichi_direct", "barnes_hut_cpu"}


def test_numpy_direct_has_zero_self_error():
    """numpy_direct is the reference it's compared against, so its own
    error columns should be exactly zero, not just small."""
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    positions, masses = factory(30, 1)
    results = evaluate_all_backends("random_cloud", positions, masses, softening, repeats=1)
    numpy_result = next(r for r in results if r.backend == "numpy_direct")
    assert numpy_result.mean_relative_error == 0.0
    assert numpy_result.speedup_vs_numpy_direct == 1.0


def test_taichi_direct_error_is_floating_point_noise():
    """taichi_direct is not an approximation -- its error vs. numpy_direct
    should be at the floating-point noise floor, not just "small"."""
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    positions, masses = factory(50, 2)
    results = evaluate_all_backends("random_cloud", positions, masses, softening, repeats=1)
    taichi_result = next(r for r in results if r.backend == "taichi_direct")
    assert taichi_result.max_relative_error < 1e-9


def test_results_are_finite():
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    positions, masses = factory(30, 3)
    results = evaluate_all_backends("random_cloud", positions, masses, softening, repeats=1)
    for r in results:
        assert r.has_finite_values()


def test_particle_scaling_validation_covers_every_n():
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    counts = [20, 40]
    results = run_particle_scaling_validation(
        "random_cloud", factory, particle_counts=counts, softening=softening, repeats=1
    )
    ns_seen = sorted({r.n_particles for r in results})
    assert ns_seen == counts
    assert len(results) == len(counts) * 3  # 3 backends per N


def test_save_results_csv_roundtrip(tmp_path):
    factory, softening = GPU_VALIDATION_SYSTEMS["random_cloud"]
    positions, masses = factory(25, 4)
    results = evaluate_all_backends("random_cloud", positions, masses, softening, repeats=1)

    csv_path = tmp_path / "gpu_results.csv"
    save_results_csv(results, csv_path)

    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per result
    assert "speedup_vs_numpy_direct" in lines[0]
