"""Stage 3 validation: Taichi (GPU/CPU) direct solver vs. NumPy direct vs. Barnes-Hut CPU.

Usage:
    python scripts/run_gpu_validation.py

Outputs (outputs/gpu_validation/):
    gpu_validation_results.csv
    runtime_vs_n.png
    speedup_vs_n.png
    taichi_error_vs_n.png
    solver_comparison.png

Scientific framing: numpy_direct is the ground-truth reference throughout.
taichi_direct is NOT an approximation of it (same O(N^2) sum, parallel
threads instead of a vectorized single-threaded loop) -- its "error" should
sit at the floating-point noise floor regardless of N. barnes_hut_cpu IS a
genuine approximation, included so Stage 3's numbers sit on the same axes
as Stage 2's.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.taichi_forces import init_taichi, taichi_backend_info
from validation.gpu_validation import GPU_VALIDATION_SYSTEMS, run_particle_scaling_validation, save_results_csv
from visualization.gpu_validation_plots import (
    plot_runtime_vs_n,
    plot_solver_comparison,
    plot_speedup_vs_n,
    plot_taichi_error_vs_n,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "gpu_validation"

# 6000 matches this project's established memory-safe ceiling for the
# direct solver on this machine (see the Barnes-Hut validation's finding:
# N=8000 gets OOM-killed here before any runtime crossover is reached).
PARTICLE_COUNTS = [100, 300, 1000, 3000, 6000]
THETA = 0.5
REPEATS = 3


def main() -> None:
    arch = init_taichi()
    info = taichi_backend_info()
    print(f"Taichi backend info: {info}")
    if arch in ("cuda", "vulkan", "metal"):
        print(f"Genuine GPU backend detected and in use: {arch}")
    elif arch == "cpu":
        print(
            "No GPU backend detected/usable on this machine -- Taichi initialized on its "
            "CPU backend. Results below still test the Taichi *code path* end-to-end and "
            "its multi-threaded CPU execution vs. NumPy's single-threaded loop; they do NOT "
            "demonstrate real GPU speedup, since there is no GPU here to demonstrate it on."
        )
    else:
        print("Taichi unavailable -- taichi_direct results below will not be produced.")
        return

    system_name = "random_cloud"
    factory, softening = GPU_VALIDATION_SYSTEMS[system_name]

    print(f"\n=== GPU validation: {system_name}, theta={THETA}, {REPEATS} repeats per N ===")
    results = run_particle_scaling_validation(
        system_name=system_name,
        factory=factory,
        particle_counts=PARTICLE_COUNTS,
        softening=softening,
        theta=THETA,
        repeats=REPEATS,
    )

    header = f"{'N':>6} {'backend':>15} {'mean_err':>10} {'runtime(s)':>11} {'speedup':>9} {'peak_rss(MB)':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.n_particles:>6} {r.backend:>15} {r.mean_relative_error:>10.2e} "
            f"{r.runtime_s:>11.4f} {r.speedup_vs_numpy_direct:>8.3f}x {r.peak_rss_mb:>13.1f}"
        )
        if not r.has_finite_values():
            print(f"  !! WARNING: non-finite value for backend={r.backend} at N={r.n_particles}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "gpu_validation_results.csv"
    save_results_csv(results, csv_path)
    print(f"\nsaved CSV -> {csv_path}")

    plot_runtime_vs_n(results, OUTPUT_DIR / "runtime_vs_n.png")
    plot_speedup_vs_n(results, OUTPUT_DIR / "speedup_vs_n.png")
    plot_taichi_error_vs_n(results, OUTPUT_DIR / "taichi_error_vs_n.png")
    plot_solver_comparison(results, OUTPUT_DIR / "solver_comparison.png")
    print(f"saved plots -> {OUTPUT_DIR}")

    print("\nGPU validation complete.")


if __name__ == "__main__":
    main()
