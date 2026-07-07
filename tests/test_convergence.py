"""Tests for validation.convergence: the convergence/validation harness itself.

These tests check the harness's *claims* are true of the engine it wraps:
smaller dt should improve leapfrog's energy conservation, leapfrog should
beat Euler at matched dt, and the harness's own output should be well-formed
(no NaN/inf, expected columns present).
"""
import numpy as np
import pytest

from validation.convergence import (
    BENCHMARK_SYSTEMS,
    compare_integrators,
    results_to_dataframe,
    save_results_csv,
    sweep_timestep,
)

DT_VALUES = [0.02, 0.01, 0.005, 0.0025]


def test_smaller_dt_improves_leapfrog_energy_conservation():
    """As dt shrinks, leapfrog's energy-drift envelope should shrink too."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=DT_VALUES,
        total_time=system.recommended_total_time,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    results = sorted(results, key=lambda r: r.dt, reverse=True)  # largest dt first
    envelopes = [r.max_relative_energy_drift for r in results]

    # Each halving of dt should meaningfully shrink the envelope (allow some
    # slack since leapfrog is 2nd order, not exactly monotonically perfect
    # at every single sample, but the overall trend must hold).
    for coarse, fine in zip(envelopes, envelopes[1:]):
        assert fine < coarse, f"envelope did not shrink: {coarse} -> {fine}"

    # Finest dt should be dramatically better than coarsest (orders of magnitude,
    # consistent with ~2nd order convergence over a 8x range of dt).
    assert envelopes[-1] < envelopes[0] / 10


def test_leapfrog_beats_euler_at_matched_dt():
    """At the same dt, leapfrog's energy-drift envelope must be far smaller than Euler's."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    for dt in [0.01, 0.005]:
        results = compare_integrators(
            system_name=system.name,
            factory=system.factory,
            integrators=["euler", "leapfrog"],
            dt=dt,
            softening=system.recommended_softening,
            total_time=system.recommended_total_time,
        )
        by_integrator = {r.integrator: r for r in results}
        euler_drift = by_integrator["euler"].max_relative_energy_drift
        leapfrog_drift = by_integrator["leapfrog"].max_relative_energy_drift
        assert leapfrog_drift < euler_drift, f"leapfrog should beat euler at dt={dt}"
        # not just "better" but dramatically better -- this is the whole point
        # of using a symplectic integrator
        assert leapfrog_drift < euler_drift / 100


def test_figure_eight_convergence_order_is_roughly_second_order():
    """Empirical convergence order for leapfrog should be close to 2 (not 1, not 0)."""
    system = BENCHMARK_SYSTEMS["figure_eight"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=DT_VALUES,
        total_time=system.recommended_total_time,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    results = sorted(results, key=lambda r: r.dt)
    dts = np.array([r.dt for r in results])
    envelopes = np.array([r.max_relative_energy_drift for r in results])

    # Fit log(envelope) = order * log(dt) + const across the sweep.
    log_dt = np.log(dts)
    log_env = np.log(envelopes)
    order, _ = np.polyfit(log_dt, log_env, 1)

    # Expect roughly 2nd order; generous bounds since this is measured, not
    # asserted -- a real regression (e.g. an accidental switch to Euler
    # internals) would show order ~1, well outside this range.
    assert 1.5 < order < 2.7, f"unexpected empirical convergence order: {order}"


def test_validation_output_has_expected_columns():
    """DataFrame conversion should expose all the diagnostic fields the harness promises."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=[0.02, 0.01],
        total_time=2.0,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    df = results_to_dataframe(results)

    expected_columns = {
        "system_name",
        "integrator",
        "dt",
        "softening",
        "n_steps",
        "actual_total_time",
        "initial_energy",
        "final_energy",
        "relative_energy_drift",
        "max_relative_energy_drift",
        "relative_angular_momentum_drift",
        "center_of_mass_drift",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 2


def test_no_nan_or_infinite_values_in_results():
    """Every ConvergenceResult across sweeps of two systems should be fully finite."""
    for system_name in ["binary_orbit", "figure_eight"]:
        system = BENCHMARK_SYSTEMS[system_name]
        results = sweep_timestep(
            system_name=system.name,
            factory=system.factory,
            dt_values=[0.02, 0.01],
            total_time=min(2.0, system.recommended_total_time),
            softening=system.recommended_softening,
            integrator="leapfrog",
        )
        for r in results:
            assert r.has_finite_values(), f"non-finite value in {system_name} at dt={r.dt}"


def test_save_results_csv_roundtrip(tmp_path):
    """CSV export should produce a readable file with a header and one row per result."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=[0.02, 0.01],
        total_time=2.0,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    csv_path = tmp_path / "test_convergence.csv"
    save_results_csv(results, csv_path)

    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per result
    assert "relative_energy_drift" in lines[0]
        dt_values=DT_VALUES,
        total_time=system.recommended_total_time,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    results = sorted(results, key=lambda r: r.dt)
    dts = np.array([r.dt for r in results])
    envelopes = np.array([r.max_relative_energy_drift for r in results])

    # Fit log(envelope) = order * log(dt) + const across the sweep.
    log_dt = np.log(dts)
    log_env = np.log(envelopes)
    order, _ = np.polyfit(log_dt, log_env, 1)

    # Expect roughly 2nd order; generous bounds since this is measured, not
    # asserted -- a real regression (e.g. an accidental switch to Euler
    # internals) would show order ~1, well outside this range.
    assert 1.5 < order < 2.7, f"unexpected empirical convergence order: {order}"


def test_validation_output_has_expected_columns():
    """DataFrame conversion should expose all the diagnostic fields the harness promises."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=[0.02, 0.01],
        total_time=2.0,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    df = results_to_dataframe(results)

    expected_columns = {
        "system_name",
        "integrator",
        "dt",
        "softening",
        "n_steps",
        "actual_total_time",
        "initial_energy",
        "final_energy",
        "relative_energy_drift",
        "max_relative_energy_drift",
        "relative_angular_momentum_drift",
        "center_of_mass_drift",
    }
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 2


def test_no_nan_or_infinite_values_in_results():
    """Every ConvergenceResult across sweeps of two systems should be fully finite."""
    for system_name in ["binary_orbit", "figure_eight"]:
        system = BENCHMARK_SYSTEMS[system_name]
        results = sweep_timestep(
            system_name=system.name,
            factory=system.factory,
            dt_values=[0.02, 0.01],
            total_time=min(2.0, system.recommended_total_time),
            softening=system.recommended_softening,
            integrator="leapfrog",
        )
        for r in results:
            assert r.has_finite_values(), f"non-finite value in {system_name} at dt={r.dt}"


def test_save_results_csv_roundtrip(tmp_path):
    """CSV export should produce a readable file with a header and one row per result."""
    system = BENCHMARK_SYSTEMS["binary_orbit"]
    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=[0.02, 0.01],
        total_time=2.0,
        softening=system.recommended_softening,
        integrator="leapfrog",
    )
    csv_path = tmp_path / "test_convergence.csv"
    save_results_csv(results, csv_path)

    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == len(results) + 1  # header + one row per result
    assert "relative_energy_drift" in lines[0]
