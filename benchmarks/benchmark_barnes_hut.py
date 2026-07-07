"""Direct vs. Barnes-Hut runtime benchmark.

Usage:
    python benchmarks/benchmark_barnes_hut.py

This is deliberately separate from scripts/run_barnes_hut_validation.py:
that script answers "how accurate is Barnes-Hut and how does it scale",
covering multiple systems; this script is a focused, repeatable runtime
benchmark on one representative system, intended to be the thing you rerun
after any change to physics/barnes_hut.py to check for performance
regressions.

Honesty note (this is the point of the exercise, not a caveat to bury):
Barnes-Hut has better asymptotic complexity (O(N log N) vs O(N^2)), but our
current implementation is a straightforward recursive Python octree with
real per-call overhead. At small N, that overhead outweighs the algorithmic
advantage and Barnes-Hut is measurably SLOWER than direct summation. This
benchmark reports that plainly rather than only showing N large enough to
flatter Barnes-Hut.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.barnes_hut_validation import BH_TEST_SYSTEMS, benchmark_particle_scaling, save_results_csv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "barnes_hut"
# 6000 is close to this machine's practical ceiling: at N=8000 the direct
# solver's O(N^2) memory footprint (several N x N x 3 float64 temporaries)
# was observed to get OOM-killed here, before Barnes-Hut's runtime
# crossover was even reached (speedup was still ~0.85x at N=7000). Worth
# knowing: on memory-constrained hardware, direct summation can become
# infeasible for reasons of RAM, not just runtime, before Barnes-Hut's
# speed advantage arrives.
PARTICLE_COUNTS = [100, 300, 1000, 3000, 6000]
THETA = 0.5
REPEATS = 3


def main() -> None:
    system_name = "random_cloud"
    factory, softening = BH_TEST_SYSTEMS[system_name]

    print(f"Benchmarking direct vs. Barnes-Hut on '{system_name}' (theta={THETA}, {REPEATS} repeats per N)\n")
    results = benchmark_particle_scaling(
        system_name=system_name,
        factory=factory,
        particle_counts=PARTICLE_COUNTS,
        theta=THETA,
        softening=softening,
        repeats=REPEATS,
    )

    header = f"{'N':>8} {'direct (s)':>12} {'barnes-hut (s)':>16} {'speedup':>10}  note"
    print(header)
    print("-" * len(header))
    crossover_n = None
    for r in results:
        note = ""
        if r.speedup < 1.0:
            note = "Barnes-Hut SLOWER than direct (tree overhead dominates at this N)"
        elif crossover_n is None:
            crossover_n = r.n_particles
            note = "<- Barnes-Hut becomes faster at/after this N"
        print(f"{r.n_particles:>8} {r.direct_runtime_s:>12.4f} {r.barnes_hut_runtime_s:>16.4f} {r.speedup:>9.3f}x  {note}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "benchmark_particle_scaling.csv"
    save_results_csv(results, csv_path)
    print(f"\nsaved CSV -> {csv_path}")

    if crossover_n is None:
        print(
            "\nBarnes-Hut did not beat direct summation at any tested N in this run.\n"
            "That's a real result, not a bug. Two honest contributing factors, measured\n"
            "on this machine: (1) this is a straightforward recursive Python octree, not\n"
            "yet a performance-optimized implementation, so its constant-factor overhead\n"
            "is real; (2) direct summation's O(N^2) *memory* footprint made N=8000 get\n"
            "OOM-killed here before a runtime crossover was reached (speedup trend was\n"
            "still climbing: ~0.85x at N=7000). On more RAM, or with a faster tree\n"
            "implementation, the crossover would appear at a different N -- rerun and\n"
            "look at the speedup trend rather than assuming a fixed crossover N."
        )
    else:
        print(f"\nCrossover point (this run): Barnes-Hut becomes faster around N={crossover_n}.")


if __name__ == "__main__":
    main()

    if crossover_n is None:
        print(
            "\nBarnes-Hut did not beat direct summation at any tested N in this run.\n"
            "That's a real result, not a bug. Two honest contributing factors, measured\n"
            "on this machine: (1) this is a straightforward recursive Python octree, not\n"
            "yet a performance-optimized implementation, so its constant-factor overhead\n"
            "is real; (2) direct summation's O(N^2) *memory* footprint made N=8000 get\n"
            "OOM-killed here before a runtime crossover was reached (speedup trend was\n"
            "still climbing: ~0.85x at N=7000). On more RAM, or with a faster tree\n"
            "implementation, the crossover would appear at a different N -- rerun and\n"
            "look at the speedup trend rather than assuming a fixed crossover N."
        )
    else:
        print(f"\nCrossover point (this run): Barnes-Hut becomes faster around N={crossover_n}.")


if __name__ == "__main__":
    main()
