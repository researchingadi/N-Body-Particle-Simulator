"""Stage 2 validation: Barnes-Hut accuracy and performance vs. the direct solver.

Usage:
    python scripts/run_barnes_hut_validation.py

Outputs (outputs/barnes_hut/):
    theta_sweep_<system>.csv                 per-system theta sweep
    particle_scaling_random_cloud.csv         N-scaling sweep
    error_vs_theta_<system>.png
    accuracy_runtime_tradeoff_<system>.png
    runtime_vs_particle_count.png
    speedup_vs_particle_count.png

Scientific framing: the direct O(N^2) solver is the ground truth throughout.
"Error" always means "Barnes-Hut's deviation from direct summation", not the
other way around. Barnes-Hut's value proposition is scalability, not
accuracy -- see README.md's "Barnes-Hut Approximation" section.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.barnes_hut_validation import (
    BH_TEST_SYSTEMS,
    benchmark_particle_scaling,
    save_results_csv,
    sweep_theta,
)
from visualization.barnes_hut_plots import (
    plot_accuracy_runtime_tradeoff,
    plot_error_vs_theta,
    plot_runtime_vs_particle_count,
    plot_speedup_vs_particle_count,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "barnes_hut"

THETA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]
# 6000 is close to this machine's practical ceiling for the direct solver:
# at N=8000, direct summation's O(N^2) *memory* footprint (several N x N x 3
# float64 temporary arrays inside physics.forces.compute_accelerations) was
# observed to exceed available RAM and get OOM-killed on this machine,
# before Barnes-Hut's runtime crossover point was even reached (speedup was
# still only ~0.85x at N=7000). That's a real, useful finding, not just a
# benchmark inconvenience -- see the README's Barnes-Hut section.
PARTICLE_COUNTS = [100, 300, 1000, 3000, 6000]
N_FOR_THETA_SWEEP = 500  # moderate size: big enough for BH's approximation to matter, small enough to run fast


def run_theta_sweep_for_system(system_name: str) -> None:
    factory, softening = BH_TEST_SYSTEMS[system_name]
    print(f"\n=== theta sweep: {system_name} (N={N_FOR_THETA_SWEEP}, softening={softening}) ===")

    results = sweep_theta(
        system_name=system_name,
        factory=factory,
        n=N_FOR_THETA_SWEEP,
        theta_values=THETA_VALUES,
        softening=softening,
    )

    for r in results:
        print(
            f"  theta={r.theta:<4} mean_err={r.mean_relative_error:.3e}  "
            f"median_err={r.median_relative_error:.3e}  max_err={r.max_relative_error:.3e}  "
            f"t_direct={r.direct_runtime_s:.4f}s  t_bh={r.barnes_hut_runtime_s:.4f}s  "
            f"speedup={r.speedup:.3f}x"
        )
        if not r.has_finite_values():
            print(f"  !! WARNING: non-finite value at theta={r.theta}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"theta_sweep_{system_name}.csv"
    save_results_csv(results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_error_vs_theta(
        results, OUTPUT_DIR / f"error_vs_theta_{system_name}.png", title=f"{system_name}: error vs theta"
    )
    plot_accuracy_runtime_tradeoff(
        results,
        OUTPUT_DIR / f"accuracy_runtime_tradeoff_{system_name}.png",
        title=f"{system_name}: accuracy/runtime tradeoff",
    )
    print(f"  saved plots -> {OUTPUT_DIR}")


def run_particle_scaling_study() -> None:
    """Runtime/speedup vs. N, on the neutral random_cloud system.

    random_cloud is used here (rather than Plummer/disk) specifically
    because it has no preferred clustering scale -- we want the N-scaling
    plot to reflect the solvers' algorithmic complexity, not an artifact of
    a particular system's density profile.
    """
    system_name = "random_cloud"
    factory, softening = BH_TEST_SYSTEMS[system_name]
    print(f"\n=== particle-count scaling: {system_name}, theta=0.5 ===")

    results = benchmark_particle_scaling(
        system_name=system_name,
        factory=factory,
        particle_counts=PARTICLE_COUNTS,
        theta=0.5,
        softening=softening,
        repeats=3,
    )

    for r in results:
        note = "  <-- Barnes-Hut SLOWER than direct here" if r.speedup < 1.0 else ""
        print(
            f"  N={r.n_particles:<6} t_direct={r.direct_runtime_s:.4f}s  "
            f"t_bh={r.barnes_hut_runtime_s:.4f}s  speedup={r.speedup:.3f}x{note}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"particle_scaling_{system_name}.csv"
    save_results_csv(results, csv_path)
    print(f"  saved CSV -> {csv_path}")

    plot_runtime_vs_particle_count(results, OUTPUT_DIR / "runtime_vs_particle_count.png")
    plot_speedup_vs_particle_count(results, OUTPUT_DIR / "speedup_vs_particle_count.png")
    print(f"  saved plots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    for name in ["random_cloud", "plummer_sphere", "disk_galaxy"]:
        run_theta_sweep_for_system(name)

    run_particle_scaling_study()
    print("\nBarnes-Hut validation complete.")
