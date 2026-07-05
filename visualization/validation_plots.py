"""Plotting utilities for convergence/validation studies.

Separate from visualization/plot.py (which plots a single simulation's
trajectories/energy) -- this module plots *across* multiple runs, e.g. how
a diagnostic changes as dt varies. Kept in `visualization/` rather than
`validation/` since its only job is rendering, matching the existing
separation between physics/diagnostics modules and the visualization layer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validation.convergence import ConvergenceResult

# A small, consistent style so convergence figures look like they belong
# to the same report rather than ad hoc matplotlib defaults.
_STYLE = {
    "figure.figsize": (7, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}


def _sorted_by_dt(results: list[ConvergenceResult]) -> list[ConvergenceResult]:
    return sorted(results, key=lambda r: r.dt)


def plot_energy_drift_vs_dt(
    results: list[ConvergenceResult], path: str | Path, title: str | None = None
) -> None:
    """Linear-scale plot of final relative energy drift vs. timestep.

    Shows the actual signed drift (can be positive or negative for a
    symplectic integrator sampled at a particular final phase), which is
    why 0 is drawn as a reference line rather than treating "small" as
    automatically "good in one direction".
    """
    results = _sorted_by_dt(results)
    dts = [r.dt for r in results]
    drifts = [r.relative_energy_drift for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.plot(dts, drifts, marker="o")
        ax.set_xlabel("timestep (dt)")
        ax.set_ylabel("relative energy drift (final step)")
        ax.set_title(title or "Relative energy drift vs. timestep")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_angular_momentum_drift_vs_dt(
    results: list[ConvergenceResult], path: str | Path, title: str | None = None
) -> None:
    """Linear-scale plot of relative angular momentum drift vs. timestep."""
    results = _sorted_by_dt(results)
    dts = [r.dt for r in results]
    drifts = [r.relative_angular_momentum_drift for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(dts, drifts, marker="o", color="darkorange")
        ax.set_xlabel("timestep (dt)")
        ax.set_ylabel("relative angular momentum drift")
        ax.set_title(title or "Angular momentum drift vs. timestep")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_com_drift_vs_dt(
    results: list[ConvergenceResult], path: str | Path, title: str | None = None
) -> None:
    """Linear-scale plot of center-of-mass drift vs. timestep."""
    results = _sorted_by_dt(results)
    dts = [r.dt for r in results]
    drifts = [r.center_of_mass_drift for r in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(dts, drifts, marker="o", color="seagreen")
        ax.set_xlabel("timestep (dt)")
        ax.set_ylabel("center-of-mass drift (max displacement)")
        ax.set_title(title or "Center-of-mass drift vs. timestep")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_loglog_convergence(
    results: list[ConvergenceResult],
    path: str | Path,
    metric: str = "max_relative_energy_drift",
    reference_orders: tuple[int, ...] = (1, 2),
    title: str | None = None,
) -> None:
    """Log-log plot of |metric| vs. dt, with reference slope lines.

    Reference lines (dt^1, dt^2, ...) let you read the empirical convergence
    order directly off the plot: if the data points are roughly parallel to
    the dt^2 line, the method is behaving like a 2nd-order integrator, which
    is the expected result for leapfrog/velocity Verlet. Only points with a
    strictly positive metric value are plotted (log scale can't show zero or
    negative values); if a run's drift underflows to exactly 0.0 it is
    dropped from this plot with a note, not silently misrepresented.
    """
    results = _sorted_by_dt(results)
    dts = np.array([r.dt for r in results])
    values = np.array([abs(getattr(r, metric)) for r in results])

    mask = values > 0
    dropped = int(np.sum(~mask))
    dts_plot, values_plot = dts[mask], values[mask]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        if len(dts_plot) > 0:
            ax.plot(dts_plot, values_plot, marker="o", label=f"measured |{metric}|", color="black")

            # anchor reference lines at the finest dt / smallest measured value
            anchor_dt = dts_plot[0]
            anchor_val = values_plot[0]
            for order in reference_orders:
                ref = anchor_val * (dts_plot / anchor_dt) ** order
                ax.plot(dts_plot, ref, linestyle="--", alpha=0.6, label=f"dt^{order} reference")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("timestep (dt)")
        ax.set_ylabel(f"|{metric}|")
        subtitle = title or f"Convergence: |{metric}| vs. dt (log-log)"
        if dropped:
            subtitle += f"\n({dropped} point(s) with exactly-zero value omitted)"
        ax.set_title(subtitle)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_integrator_comparison(
    results_by_integrator: dict[str, list[ConvergenceResult]],
    path: str | Path,
    metric: str = "max_relative_energy_drift",
    title: str | None = None,
) -> None:
    """Overlay multiple integrators' |metric| vs. dt on one log-log plot.

    Intended for the leapfrog-vs-Euler comparison: Euler's curve should sit
    well above leapfrog's at every matched dt, showing the gap is due to the
    integrator, not an unfair choice of timestep.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        for integrator, results in results_by_integrator.items():
            results = _sorted_by_dt(results)
            dts = np.array([r.dt for r in results])
            values = np.array([abs(getattr(r, metric)) for r in results])
            mask = values > 0
            if np.any(mask):
                ax.plot(dts[mask], values[mask], marker="o", label=integrator)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("timestep (dt)")
        ax.set_ylabel(f"|{metric}|")
        ax.set_title(title or f"Integrator comparison: |{metric}| vs. dt")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
