"""Plotting utilities for GPU/CPU solver-backend validation.

Mirrors the role of visualization/validation_plots.py and
visualization/barnes_hut_plots.py: this module only renders results
computed elsewhere (validation.gpu_validation), it doesn't compute anything.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.gpu_validation import GPUValidationResult

_STYLE = {
    "figure.figsize": (7, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}

_BACKEND_COLORS = {
    "numpy_direct": "tab:blue",
    "taichi_direct": "tab:orange",
    "barnes_hut_cpu": "tab:green",
}
_BACKEND_LABELS = {
    "numpy_direct": "NumPy direct (reference)",
    "taichi_direct": "Taichi direct",
    "barnes_hut_cpu": "Barnes-Hut (CPU)",
}


def _by_backend(results: list[GPUValidationResult]) -> dict[str, list[GPUValidationResult]]:
    grouped: dict[str, list[GPUValidationResult]] = {}
    for r in results:
        grouped.setdefault(r.backend, []).append(r)
    for backend_results in grouped.values():
        backend_results.sort(key=lambda r: r.n_particles)
    return grouped


def plot_runtime_vs_n(results: list[GPUValidationResult], path: str | Path, title: str | None = None) -> None:
    """Runtime vs. N for all three backends, log-log."""
    grouped = _by_backend(results)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        for backend, backend_results in grouped.items():
            ns = [r.n_particles for r in backend_results]
            runtimes = [r.runtime_s for r in backend_results]
            ax.plot(ns, runtimes, marker="o", label=_BACKEND_LABELS.get(backend, backend), color=_BACKEND_COLORS.get(backend))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("runtime (s)")
        ax.set_title(title or "Runtime vs. particle count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_speedup_vs_n(results: list[GPUValidationResult], path: str | Path, title: str | None = None) -> None:
    """Speedup vs. numpy_direct, vs. N, for taichi_direct and barnes_hut_cpu.

    numpy_direct itself is omitted (trivially always 1.0x by definition) --
    a break-even reference line at 1.0 is drawn instead.
    """
    grouped = _by_backend(results)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="break-even (numpy_direct)")
        for backend in ("taichi_direct", "barnes_hut_cpu"):
            if backend not in grouped:
                continue
            backend_results = grouped[backend]
            ns = [r.n_particles for r in backend_results]
            speedups = [r.speedup_vs_numpy_direct for r in backend_results]
            ax.plot(ns, speedups, marker="o", label=_BACKEND_LABELS.get(backend, backend), color=_BACKEND_COLORS.get(backend))
        ax.set_xscale("log")
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("speedup (numpy_direct time / backend time)")
        ax.set_title(title or "Speedup vs. particle count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_taichi_error_vs_n(results: list[GPUValidationResult], path: str | Path, title: str | None = None) -> None:
    """Taichi's mean/median/max relative error vs. numpy_direct, across N.

    Expected shape: flat, at the floating-point noise floor (~1e-15), with
    no growth as N increases -- since taichi_direct computes the exact same
    sum as numpy_direct, not an approximation of it. A rising trend here
    would indicate a real bug, not expected approximation behavior.
    """
    grouped = _by_backend(results)
    taichi_results = grouped.get("taichi_direct", [])
    ns = [r.n_particles for r in taichi_results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(ns, [r.mean_relative_error for r in taichi_results], marker="o", label="mean")
        ax.plot(ns, [r.median_relative_error for r in taichi_results], marker="s", label="median")
        ax.plot(ns, [r.max_relative_error for r in taichi_results], marker="^", label="max")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("relative acceleration error vs. numpy_direct")
        ax.set_title(title or "Taichi direct: error vs. numpy_direct (should stay at floating-point noise floor)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_solver_comparison(results: list[GPUValidationResult], path: str | Path, title: str | None = None) -> None:
    """Grouped bar chart: runtime per backend, per N.

    A complementary view to plot_runtime_vs_n's log-log line plot -- bars
    make the per-N relative ordering of backends easier to read at a glance,
    at the cost of not showing the overall scaling trend as clearly.
    """
    grouped = _by_backend(results)
    all_ns = sorted({r.n_particles for r in results})
    backends = [b for b in ("numpy_direct", "taichi_direct", "barnes_hut_cpu") if b in grouped]

    x = np.arange(len(all_ns))
    bar_width = 0.8 / max(len(backends), 1)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        for i, backend in enumerate(backends):
            backend_by_n = {r.n_particles: r.runtime_s for r in grouped[backend]}
            heights = [backend_by_n.get(n, 0.0) for n in all_ns]
            offset = (i - (len(backends) - 1) / 2) * bar_width
            ax.bar(x + offset, heights, width=bar_width, label=_BACKEND_LABELS.get(backend, backend), color=_BACKEND_COLORS.get(backend))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in all_ns])
        ax.set_xlabel("number of particles (N)")
        ax.set_ylabel("runtime (s)")
        ax.set_title(title or "Solver comparison: runtime by backend and N")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
