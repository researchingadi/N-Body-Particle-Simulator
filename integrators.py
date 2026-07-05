"""Time integrators for the N-body system.

Force is assumed velocity-independent (pure gravity), so leapfrog
kick-drift-kick (KDK) and velocity Verlet are algebraically identical here.
Both are provided because they are conventionally named differently in the
astrophysics vs. molecular-dynamics literature; keeping both makes the code
readable to either audience. Euler is included only as a known-bad baseline
to demonstrate energy drift in tests and docs -- it should never be used for
production runs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

AccelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
# AccelFn(positions, masses) -> accelerations, softening/G bound via closure


def euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit (forward) Euler step. Unstable / energy-drifting -- baseline only."""
    acc = accel_fn(positions, masses)
    new_positions = positions + velocities * dt
    new_velocities = velocities + acc * dt
    return new_positions, new_velocities


def leapfrog_kdk_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    prev_accel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kick-drift-kick leapfrog step (symplectic, time-reversible).

        v_(1/2) = v_0 + a(x_0) * dt/2      [kick]
        x_1     = x_0 + v_(1/2) * dt       [drift]
        v_1     = v_(1/2) + a(x_1) * dt/2  [kick]

    Args:
        prev_accel: acceleration at positions, if already computed by the
            caller (avoids a redundant force evaluation across steps).

    Returns:
        (new_positions, new_velocities, accel_at_new_positions) so the caller
        can reuse the last acceleration on the next call.
    """
    a0 = prev_accel if prev_accel is not None else accel_fn(positions, masses)
    v_half = velocities + 0.5 * dt * a0
    new_positions = positions + dt * v_half
    a1 = accel_fn(new_positions, masses)
    new_velocities = v_half + 0.5 * dt * a1
    return new_positions, new_velocities, a1


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
    prev_accel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity Verlet step. Algebraically equivalent to leapfrog KDK for
    velocity-independent forces; provided as a distinct named entry point
    for API clarity and future extension (e.g. velocity-dependent drag).
    """
    a0 = prev_accel if prev_accel is not None else accel_fn(positions, masses)
    new_positions = positions + velocities * dt + 0.5 * a0 * dt**2
    a1 = accel_fn(new_positions, masses)
    new_velocities = velocities + 0.5 * (a0 + a1) * dt
    return new_positions, new_velocities, a1


def rk4_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    accel_fn: AccelFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic 4th-order Runge-Kutta. Not symplectic -- energy drifts slowly
    over long integrations despite high per-step accuracy. Provided as an
    optional reference integrator, not the recommended default.
    """
    def deriv(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return v, accel_fn(x, masses)

    k1x, k1v = deriv(positions, velocities)
    k2x, k2v = deriv(positions + 0.5 * dt * k1x, velocities + 0.5 * dt * k1v)
    k3x, k3v = deriv(positions + 0.5 * dt * k2x, velocities + 0.5 * dt * k2v)
    k4x, k4v = deriv(positions + dt * k3x, velocities + dt * k3v)

    new_positions = positions + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_velocities = velocities + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_positions, new_velocities
