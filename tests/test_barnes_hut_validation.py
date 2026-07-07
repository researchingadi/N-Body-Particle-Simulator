"""Tests for validation.barnes_hut_validation: the harness, not the solver.

physics/barnes_hut.py's correctness is covered by tests/test_barnes_hut.py.
These tests check the harness built on top of it behaves as advertised:
well-formed output, no NaN/inf, and CSV export that round-trips.
"""
import numpy as np

from validation.barnes_hut_validation import (
    BH_TEST_SYSTEMS,
    benchmark_particle_scaling,
    save_results_csv,
    sweep_theta,
)


def test_sweep_theta_returns_one_result_per_theta():
    factory, softening = BH_TEST_SYSTEMS["random_cloud"]
    theta_values = [0.3, 0.6, 0.9]
    results = sweep_theta("random_cloud", factory, n=40, theta_values=theta_values, softening=softening)
    assert len(results) == len(theta_values)
    assert [r.theta for r in results] == theta_values


def test_sweep_theta_results_are_finite():
    factory, softening = BH_TEST_SYSTEMS["plummer_sphere"]
    results = sweep_theta("plummer_sphere", factory, n=40, theta_values=[0.3, 0.8], softening=softening)
    for r in results:
        assert r.has_finite_values()


def test_benchmark_particle_scaling_runs_for_each_n():
    factory, softening = BH_TEST_SYSTEMS["random_cloud"]
    counts = [20, 40]
    results = benchmark_particle_scaling(
        "random_cloud", factory, particle_counts=counts, theta=0.5, softening=softening, repeats=1
    )
    assert [r.n_particles for r in results] == counts
    for r in results:
        assert r.has_finite_values()
        assert r.direct_runtime_s > 0
        assert r.barnes_hut_runtime_s > 0


def test_save_results_csv_roundtrip(tmp_path):
    factory, softening = BH_TEST_SYSTEMS["random_cloud"]
    results = sweep_theta("random_cloud", factory, n=30, theta_values=[0.3, 0.6], softening=softening)
    csv_path = tmp_path / "bh_results.csv"
    save_results_csv(results, csv_path)

    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per result
    assert "mean_relative_error" in lines[0]
