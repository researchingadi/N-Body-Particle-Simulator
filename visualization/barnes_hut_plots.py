"""Plotting utilities for Barnes-Hut accuracy/performance validation.

Mirrors visualization/validation_plots.py's role for the convergence
harness: this module only renders results computed elsewhere
(validation.barnes_hut_validation), it doesn't compute anything itself.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.barnes_hut_validation import AccuracyResult

_STYLE = {
    "figure.figsize": (7, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}


def plot_error_vs_theta(
    results: list[AccuracyResult], path: str | Path, title: str | None = None
) -> None:
    """Mean/median/max relative acceleration error vs. opening angle theta.

    Log scale on the y-axis since error typically spans a couple of orders
    of magnitude across the usual theta range (~0.2 to ~1.0).
    """
    results = sorted(results, key=lambda r: r.theta)
    thetas = [r.theta for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(thetas, [r.mean_relative_error for r in results], marker="o", label="mean")
        ax.plot(thetas, [r.median_relative_error for r in results], marker="s", label="median")
        ax.plot(thetas, [r.max_relative_error for r in results], marker="^", label="max")
        ax.set_yscale("log")
        ax.set_xlabel("opening angle (theta)")
        ax.set_ylabel("relative acceleration error vs. direct solver")
        ax.set_title(title or "Barnes-Hut error vs. theta")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_runtime_vs_particle_count(
    results: list[AccuracyResult], path: str | Path, title: str | None = None
) -> None:
    """Direct vs. Barnes-Hut runtime vs. N, log-log (to show O(N^2) vs O(N log N))."""
    results = sorted(results, key=lambda r: r.n_particles)
    ns = [r.n_particles for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(ns, [r.direct_runtime_s for r in results], marker="o", label="direct O(N^2)")
        ax.plot(ns, [r.barnes_hut_runtime_s for r in results], marker="s", label="Barnes-Hut")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("runtime (s)")
        ax.set_title(title or "Runtime vs. particle count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_speedup_vs_particle_count(
    results: list[AccuracyResult], path: str | Path, title: str | None = None
) -> None:
    """Barnes-Hut speedup (direct_time / bh_time) vs. N.

    A horizontal reference line at speedup=1 marks the crossover point:
    below it, Barnes-Hut is actually SLOWER than direct summation (tree
    overhead dominates at small N) -- this is expected and reported
    honestly, not hidden.
    """
    results = sorted(results, key=lambda r: r.n_particles)
    ns = [r.n_particles for r in results]
    speedups = [r.speedup for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="break-even")
        ax.plot(ns, speedups, marker="o", color="darkorange")
        ax.set_xscale("log")
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("speedup (direct time / Barnes-Hut time)")
        ax.set_title(title or "Barnes-Hut speedup vs. particle count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_accuracy_runtime_tradeoff(
    results: list[AccuracyResult], path: str | Path, title: str | None = None
) -> None:
    """Scatter of mean relative error vs. runtime, one point per theta value,
    annotated with theta -- the direct visual answer to "what do I give up
    in accuracy for what I gain in speed".
    """
    results = sorted(results, key=lambda r: r.theta)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        errors = [r.mean_relative_error for r in results]
        runtimes = [r.barnes_hut_runtime_s for r in results]
        ax.plot(runtimes, errors, marker="o", color="black")
        for r in results:
            ax.annotate(
                f"θ={r.theta}",
                (r.barnes_hut_runtime_s, r.mean_relative_error),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
        ax.set_xlabel("Barnes-Hut runtime (s)")
        ax.set_ylabel("mean relative acceleration error")
        ax.set_yscale("log")
        ax.set_title(title or "Accuracy / runtime tradeoff across theta")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
