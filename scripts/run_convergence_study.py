"""Stage 1.5 validation: convergence and integrator comparison study.

Runs before Barnes-Hut / GPU / frontend work, to establish that the direct
solver + leapfrog integrator behave the way the numerics predict: relative
energy error should shrink (in envelope, not necessarily monotonically at
every single final sample) as dt shrinks, and leapfrog should outperform
Euler at matched dt.

Usage:
    python scripts/run_convergence_study.py

Outputs (outputs/convergence/):
    dt_sweep_binary_orbit.csv
    dt_sweep_figure_eight.csv
    integrator_comparison_binary_orbit.csv
    *.png plots for each
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.convergence import (
    BENCHMARK_SYSTEMS,
    compare_integrators,
    save_results_csv,
    sweep_timestep,
)
from visualization.validation_plots import (
    plot_angular_momentum_drift_vs_dt,
    plot_com_drift_vs_dt,
    plot_energy_drift_vs_dt,
    plot_integrator_comparison,
    plot_loglog_convergence,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "convergence"

# Held fixed across the dt sweep so every run covers the same physical time
# span; only dt (and therefore step count) changes.
DT_VALUES = [0.02, 0.01, 0.005, 0.0025]


def run_dt_sweep_for_system(system_name: str) -> None:
    system = BENCHMARK_SYSTEMS[system_name]
    print(f"\n=== dt sweep: {system_name} ===")
    print(f"softening={system.recommended_softening}, total_time={system.recommended_total_time}")

    results = sweep_timestep(
        system_name=system.name,
        factory=system.factory,
        dt_values=DT_VALUES,
        total_time=system.recommended_total_time,
        softening=system.recommended_softening,
        integrator="leapfrog",
        G=system.G,
    )

    for r in results:
        print(
            f"  dt={r.dt:<8} steps={r.n_steps:<7} "
            f"final_drift={r.relative_energy_drift:+.3e}  "
            f"envelope_drift={r.max_relative_energy_drift:.3e}  "
            f"L_drift={r.relative_angular_momentum_drift:.3e}  "
            f"COM_drift={r.center_of_mass_drift:.3e}"
        )
        if not r.has_finite_values():
            print(f"  !! WARNING: non-finite diagnostic value at dt={r.dt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"dt_sweep_{system_name}.csv"
    save_results_csv(results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_energy_drift_vs_dt(results, OUTPUT_DIR / f"energy_drift_vs_dt_{system_name}.png", title=f"{system_name}: energy drift vs dt")
    plot_angular_momentum_drift_vs_dt(results, OUTPUT_DIR / f"angmom_drift_vs_dt_{system_name}.png", title=f"{system_name}: angular momentum drift vs dt")
    plot_com_drift_vs_dt(results, OUTPUT_DIR / f"com_drift_vs_dt_{system_name}.png", title=f"{system_name}: COM drift vs dt")
    plot_loglog_convergence(
        results,
        OUTPUT_DIR / f"loglog_convergence_{system_name}.png",
        metric="max_relative_energy_drift",
        reference_orders=(1, 2),
        title=f"{system_name}: leapfrog convergence order",
    )
    print(f"  saved plots -> {OUTPUT_DIR}")


def run_integrator_comparison(system_name: str) -> None:
    system = BENCHMARK_SYSTEMS[system_name]
    print(f"\n=== leapfrog vs. Euler comparison: {system_name} ===")

    results_by_integrator = {}
    for integrator in ("euler", "leapfrog"):
        results = sweep_timestep(
            system_name=system.name,
            factory=system.factory,
            dt_values=DT_VALUES,
            total_time=system.recommended_total_time,
            softening=system.recommended_softening,
            integrator=integrator,
            G=system.G,
        )
        results_by_integrator[integrator] = results
        for r in results:
            print(f"  [{integrator:>15}] dt={r.dt:<8} envelope_drift={r.max_relative_energy_drift:.3e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = results_by_integrator["euler"] + results_by_integrator["leapfrog"]
    csv_path = OUTPUT_DIR / f"integrator_comparison_{system_name}.csv"
    save_results_csv(all_results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_integrator_comparison(
        results_by_integrator,
        OUTPUT_DIR / f"integrator_comparison_{system_name}.png",
        metric="max_relative_energy_drift",
        title=f"{system_name}: leapfrog vs. Euler energy drift envelope",
    )
    print(f"  saved plot -> {OUTPUT_DIR / f'integrator_comparison_{system_name}.png'}")


if __name__ == "__main__":
    run_dt_sweep_for_system("binary_orbit")
    run_dt_sweep_for_system("figure_eight")
    run_integrator_comparison("binary_orbit")
    print("\nConvergence study complete.")
    save_results_csv(results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_energy_drift_vs_dt(results, OUTPUT_DIR / f"energy_drift_vs_dt_{system_name}.png", title=f"{system_name}: energy drift vs dt")
    plot_angular_momentum_drift_vs_dt(results, OUTPUT_DIR / f"angmom_drift_vs_dt_{system_name}.png", title=f"{system_name}: angular momentum drift vs dt")
    plot_com_drift_vs_dt(results, OUTPUT_DIR / f"com_drift_vs_dt_{system_name}.png", title=f"{system_name}: COM drift vs dt")
    plot_loglog_convergence(
        results,
        OUTPUT_DIR / f"loglog_convergence_{system_name}.png",
        metric="max_relative_energy_drift",
        reference_orders=(1, 2),
        title=f"{system_name}: leapfrog convergence order",
    )
    print(f"  saved plots -> {OUTPUT_DIR}")


def run_integrator_comparison(system_name: str) -> None:
    system = BENCHMARK_SYSTEMS[system_name]
    print(f"\n=== leapfrog vs. Euler comparison: {system_name} ===")

    results_by_integrator = {}
    for integrator in ("euler", "leapfrog"):
        results = sweep_timestep(
            system_name=system.name,
            factory=system.factory,
            dt_values=DT_VALUES,
            total_time=system.recommended_total_time,
            softening=system.recommended_softening,
            integrator=integrator,
            G=system.G,
        )
        results_by_integrator[integrator] = results
        for r in results:
            print(f"  [{integrator:>15}] dt={r.dt:<8} envelope_drift={r.max_relative_energy_drift:.3e}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = results_by_integrator["euler"] + results_by_integrator["leapfrog"]
    csv_path = OUTPUT_DIR / f"integrator_comparison_{system_name}.csv"
    save_results_csv(all_results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_integrator_comparison(
        results_by_integrator,
        OUTPUT_DIR / f"integrator_comparison_{system_name}.png",
        metric="max_relative_energy_drift",
        title=f"{system_name}: leapfrog vs. Euler energy drift envelope",
    )
    print(f"  saved plot -> {OUTPUT_DIR / f'integrator_comparison_{system_name}.png'}")


if __name__ == "__main__":
    run_dt_sweep_for_system("binary_orbit")
    run_dt_sweep_for_system("figure_eight")
    run_integrator_comparison("binary_orbit")
    print("\nConvergence study complete.")
